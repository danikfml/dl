from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from metrics import recall_at_k, mean_reciprocal_rank

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(
    'intfloat/multilingual-e5-base',
    trust_remote_code=True,
    device=device
)

test_data = pd.read_csv("test.csv")
questions = test_data['query'].tolist()
documents = test_data['answer'].tolist()


def optimized_encode(texts, batch_size=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size),
                  desc="Векторизация",
                  unit="батч"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            emb = model.encode(batch,
                               convert_to_tensor=True,
                               device=device,
                               show_progress_bar=False)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings).numpy()


question_embeddings = optimized_encode(questions)

document_embeddings = optimized_encode(documents)

cos_sim = cosine_similarity(question_embeddings, document_embeddings)

predictions = []
for scores in tqdm(cos_sim, desc="Обработка вопросов"):
    top_indices = np.argsort(-scores)[:10]
    predictions.append([documents[i] for i in top_indices])

true_answers = test_data['answer'].tolist()

print(f"Recall@1: {recall_at_k(true_answers, predictions, 1):.4f}")
print(f"Recall@3: {recall_at_k(true_answers, predictions, 3):.4f}")
print(f"Recall@10: {recall_at_k(true_answers, predictions, 10):.4f}")
print(f"MRR: {mean_reciprocal_rank(true_answers, predictions):.4f}")
