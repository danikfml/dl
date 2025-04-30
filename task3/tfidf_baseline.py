from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from metrics import recall_at_k, mean_reciprocal_rank

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

vectorizer = TfidfVectorizer(stop_words='english')

train_vectors = vectorizer.fit_transform(train_data['query'])
test_vectors = vectorizer.transform(test_data['query'])

cosine_similarities = cosine_similarity(test_vectors, train_vectors)

predict = [train_data.iloc[cosine_similarities[i].argsort()[::-1]]['answer'].values for i in range(len(test_data))]

recall_1 = recall_at_k(test_data['answer'].values, predict, k=1)
recall_3 = recall_at_k(test_data['answer'].values, predict, k=3)
recall_10 = recall_at_k(test_data['answer'].values, predict, k=10)
mrr = mean_reciprocal_rank(test_data['answer'].values, predict)

print(f"Recall@1: {recall_1}")
print(f"Recall@3: {recall_3}")
print(f"Recall@10: {recall_10}")
print(f"MRR: {mrr}")
