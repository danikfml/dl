from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = load_dataset("sentence-transformers/natural-questions")

train_df = pd.DataFrame(dataset['train'])

train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=42)

train_data = dataset['train'].select(train_data.index.tolist())
test_data = dataset['train'].select(test_data.index.tolist())

train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)
