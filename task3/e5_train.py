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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class StreamingDataset(Dataset):
    def __init__(self, df, loss_type):
        self.df = df.reset_index(drop=True)
        self.loss_type = loss_type
        self.all_answers = df['answer'].tolist()

    def __len__(self):
        return len(self.df) * 2 if self.loss_type == 'contrastive' else len(self.df)

    def __getitem__(self, idx):
        if self.loss_type == 'contrastive':
            row_idx = idx // 2
            row = self.df.iloc[row_idx]
            query = row['query']
            correct_answer = row['answer']

            if idx % 2 == 0:
                return InputExample(texts=[query, correct_answer], label=1)
            else:
                neg_answer = random.choice(self.all_answers)
                while neg_answer == correct_answer:
                    neg_answer = random.choice(self.all_answers)
                return InputExample(texts=[query, neg_answer], label=0)

        elif self.loss_type == 'triplet':
            row = self.df.iloc[idx]
            query = row['query']
            correct_answer = row['answer']
            neg_answer = random.choice(self.all_answers)
            while neg_answer == correct_answer:
                neg_answer = random.choice(self.all_answers)
            return InputExample(texts=[query, correct_answer, neg_answer])


class FastEvaluator:
    def __init__(self, test_samples=100):
        self.test_samples = test_samples

    def __call__(self, model, output_path: str, epoch: int, steps: int):
        sample_queries = test_df['query'][:self.test_samples].tolist()
        sample_answers = test_df['answer'][:self.test_samples].tolist()

        query_emb = model.encode(sample_queries,
                                 convert_to_tensor=True,
                                 device=device,
                                 batch_size=16,
                                 show_progress_bar=False)

        answer_emb = model.encode(sample_answers,
                                  convert_to_tensor=True,
                                  device=device,
                                  batch_size=16,
                                  show_progress_bar=False)

        cos_sim = cosine_similarity(query_emb.cpu(), answer_emb.cpu())

        predictions = []
        for scores in cos_sim:
            top_indices = scores.argsort()[::-1]
            predictions.append([sample_answers[i] for i in top_indices[:10]])

        print(f"\nStep {steps}:")
        print(f"Recall@1: {recall_at_k(sample_answers, predictions, 1):.3f}")
        print(f"MRR: {mean_reciprocal_rank(sample_answers, predictions):.3f}")


def train_model(model, df, loss_type, batch_size=8):
    dataset = StreamingDataset(df, loss_type)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=2,
                        prefetch_factor=2)

    loss = losses.ContrastiveLoss(model) if loss_type == 'contrastive' else losses.TripletLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=1,
        warmup_steps=50,
        optimizer_params={'lr': 1e-5},
        show_progress_bar=True,
        use_amp=True,
        evaluator=FastEvaluator(),
        evaluation_steps=500
    )


def main():
    global test_df, device

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    train_df = pd.read_csv("train.csv", usecols=['query', 'answer'])
    test_df = pd.read_csv("test.csv", usecols=['query', 'answer'])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training Contrastive Model...")
    model_contrastive = SentenceTransformer(
        'intfloat/multilingual-e5-base',
        device=device,
        trust_remote_code=True
    )
    model_contrastive.to(device).half()

    train_model(model_contrastive, train_df, 'contrastive', batch_size=16)
    model_contrastive.save('contrastive_model')

    del model_contrastive
    gc.collect()
    torch.cuda.empty_cache()

    print("\nTraining Triplet Model...")
    model_triplet = SentenceTransformer(
        'intfloat/multilingual-e5-base',
        device=device,
        trust_remote_code=True
    )
    model_triplet.to(device).half()

    train_model(model_triplet, train_df, 'triplet', batch_size=16)
    model_triplet.save('triplet_model')


if __name__ == "__main__":
    main()
