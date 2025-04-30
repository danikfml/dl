import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from metrics import recall_at_k, mean_reciprocal_rank
import gc
import numpy as np
from tqdm import tqdm

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class OptimizedDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def generate_data_for_triplet_loss_hard_negative(df, model, device, all_answers_embeddings=None):
    examples = []
    all_answers = df['answer'].tolist()
    answer_to_index = {answer: idx for idx, answer in enumerate(all_answers)}

    batch_size = 32
    query_embeddings = []

    for i in tqdm(range(0, len(df), batch_size), desc="Encoding queries"):
        batch_queries = df['query'].iloc[i:i + batch_size].tolist()
        with torch.no_grad():
            batch_emb = model.encode(batch_queries, convert_to_tensor=True, device=device)
            query_embeddings.append(batch_emb)
            del batch_emb
        torch.cuda.empty_cache()

    query_embeddings = torch.cat(query_embeddings, dim=0).to(device)

    if all_answers_embeddings is None:
        all_answers_embeddings = []
        for i in tqdm(range(0, len(all_answers), batch_size), desc="Encoding answers"):
            batch_answers = all_answers[i:i + batch_size]
            with torch.no_grad():
                batch_emb = model.encode(batch_answers, convert_to_tensor=True, device=device)
                all_answers_embeddings.append(batch_emb.cpu())
                del batch_emb
            torch.cuda.empty_cache()
        all_answers_embeddings = torch.cat(all_answers_embeddings)

    all_answers_embeddings = all_answers_embeddings.to(device)

    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_query_embs = query_embeddings[i:i + batch_size].to(device)
        batch_answers_embs = all_answers_embeddings.to(device)

        with torch.no_grad():
            cos_sim = torch.mm(batch_query_embs, batch_answers_embs.T)

        for j in range(cos_sim.size(0)):
            idx = i + j
            correct_answer = df.iloc[idx]['answer']
            correct_index = answer_to_index[correct_answer]

            masked_sim = cos_sim[j].clone()
            masked_sim[correct_index] = -torch.inf
            hard_neg_index = torch.argmax(masked_sim).item()

            examples.append(InputExample(texts=[df.iloc[idx]['query'], correct_answer, all_answers[hard_neg_index]]))

        del batch_query_embs, cos_sim
        torch.cuda.empty_cache()

    return examples


def train_model_on_triplet_loss(model, train_loader):
    model._modules['0'].auto_model.gradient_checkpointing_enable()

    model.fit(
        train_objectives=[(train_loader, losses.TripletLoss(model))],
        epochs=1,
        warmup_steps=50,
        optimizer_params={'lr': 1e-5},
        show_progress_bar=True,
        use_amp=True,
    )


def main():
    torch.cuda.set_per_process_memory_fraction(0.7, device=0)

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        'intfloat/multilingual-e5-base',
        device=device,
        trust_remote_code=True
    )

    triplet_loss_examples = generate_data_for_triplet_loss_hard_negative(train_df, model, device)

    triplet_loss_loader = DataLoader(
        OptimizedDataset(triplet_loss_examples),
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )

    train_model_on_triplet_loss(model, triplet_loss_loader)

    torch.cuda.empty_cache()

    test_queries = test_df['query'].tolist()
    test_answers = test_df['answer'].tolist()

    query_emb = model.encode(test_queries, batch_size=32, show_progress_bar=True)
    answer_emb = model.encode(test_answers, batch_size=32, show_progress_bar=True)

    cos_sim = cosine_similarity(query_emb, answer_emb)
    predictions = np.argsort(-cos_sim, axis=1)[:, :10]

    print(f"Recall@1: {recall_at_k(test_answers, predictions, 1):.4f}")
    print(f"Recall@3: {recall_at_k(test_answers, predictions, 3):.4f}")
    print(f"Recall@10: {recall_at_k(test_answers, predictions, 10):.4f}")
    print(f"MRR: {mean_reciprocal_rank(test_answers, predictions):.4f}")

    model.save('e5_finetuned_model_hard_negatives')


if __name__ == "__main__":
    main()
