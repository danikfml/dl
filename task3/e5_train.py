from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
import torch
import gc
from sklearn.metrics.pairwise import cosine_similarity
from metrics import recall_at_k, mean_reciprocal_rank

import os

os.environ["WANDB_DISABLED"] = "true"

print(torch.cuda.is_available())


class OptimizedDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def generate_data_for_contrastive_loss(df):
    examples = []
    all_answers = df['answer'].tolist()

    for _, row in df.iterrows():
        query = row['query']
        correct_answer = row['answer']

        examples.append(InputExample(
            texts=[query, correct_answer],
            label=1.0
        ))

        neg_answer = random.choice(all_answers)
        while neg_answer == correct_answer:
            neg_answer = random.choice(all_answers)
        examples.append(InputExample(
            texts=[query, neg_answer],
            label=0.0
        ))

    return examples


def generate_data_for_triplet_loss(df):
    examples = []
    all_answers = df['answer'].tolist()

    for _, row in df.iterrows():
        query = row['query']
        correct_answer = row['answer']

        neg_answer = random.choice(all_answers)
        while neg_answer == correct_answer:
            neg_answer = random.choice(all_answers)

        examples.append(InputExample(
            texts=[query, correct_answer, neg_answer]
        ))

    return examples


def train_model_on_contrastive_loss(model, train_loader):
    model.fit(
        train_objectives=[(train_loader, losses.CosineSimilarityLoss(model))],
        epochs=1,
        warmup_steps=50,
        optimizer_params={'lr': 1e-5},
        show_progress_bar=True,
        use_amp=True
    )


def train_model_on_triplet_loss(model, train_loader):
    model.fit(
        train_objectives=[(train_loader, losses.TripletLoss(model))],
        epochs=1,
        warmup_steps=50,
        optimizer_params={'lr': 1e-5},
        show_progress_bar=True,
        use_amp=True
    )


def main():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        'intfloat/multilingual-e5-base',
        device=device,
        trust_remote_code=True
    )

    contrastive_loss_examples = generate_data_for_contrastive_loss(train_df)
    triplet_loss_examples = generate_data_for_triplet_loss(train_df)

    contrastive_loss_loader = DataLoader(
        OptimizedDataset(contrastive_loss_examples),
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    triplet_loss_loader = DataLoader(
        OptimizedDataset(triplet_loss_examples),
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    train_model_on_contrastive_loss(model, contrastive_loss_loader)
    del contrastive_loss_loader, contrastive_loss_examples
    gc.collect()
    torch.cuda.empty_cache()

    train_model_on_triplet_loss(model, triplet_loss_loader)

    del triplet_loss_loader, triplet_loss_examples
    gc.collect()
    torch.cuda.empty_cache()

    def safe_encode(texts, batch_size=4):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.append(model.encode(
                batch,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            ).cpu())
        return torch.cat(embeddings)

    test_queries = test_df['query'].tolist()
    test_answers = test_df['answer'].tolist()

    query_emb = safe_encode(test_queries)
    answer_emb = safe_encode(test_answers)

    cos_sim = cosine_similarity(query_emb, answer_emb)

    predictions = []
    for scores in cos_sim:
        top_indices = scores.argsort()[::-1]
        predictions.append([test_answers[i] for i in top_indices[:10]])

    print(f"Recall@1: {recall_at_k(test_answers, predictions, 1):.4f}")
    print(f"Recall@3: {recall_at_k(test_answers, predictions, 3):.4f}")
    print(f"Recall@10: {recall_at_k(test_answers, predictions, 10):.4f}")
    print(f"MRR: {mean_reciprocal_rank(test_answers, predictions):.4f}")

    model.save('e5_finetuned_model')


if __name__ == "__main__":
    main()
