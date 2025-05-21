from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from metrics import recall_at_k, mean_reciprocal_rank

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

corpus = pd.concat([train_data['query'], train_data['answer']]).tolist()
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(corpus)

test_queries = vectorizer.transform(test_data['query'])
test_answers = vectorizer.transform(test_data['answer'])

cosine_sim = cosine_similarity(test_queries, test_answers)

predictions = []
for scores in cosine_sim:
    top_indices = scores.argsort()[::-1][:10]
    predictions.append(test_data['answer'].iloc[top_indices].tolist())

print(f"Recall@1: {recall_at_k(test_data['answer'].tolist(), predictions, 1):.4f}")
print(f"Recall@3: {recall_at_k(test_data['answer'].tolist(), predictions, 3):.4f}")
print(f"Recall@10: {recall_at_k(test_data['answer'].tolist(), predictions, 10):.4f}")
print(f"MRR: {mean_reciprocal_rank(test_data['answer'].tolist(), predictions):.4f}")
