from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LoanDataset(Dataset):
    def __init__(self, data: pd.DataFrame, num_features: list[str], cat_features: list[str]):
        self.data = data
        self.num_features = num_features
        self.cat_features = cat_features
        self.ids = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        target = torch.tensor(row['loan_status'], dtype=torch.float32) if 'loan_status' in self.data.columns else None

        numeric = {feat: torch.tensor(row[feat], dtype=torch.float32) for feat in self.num_features}

        categorical = {feat: torch.tensor(row[feat], dtype=torch.long) for feat in self.cat_features}

        return {'target': target, 'numeric_features': numeric, 'cat_features': categorical}


class LoanCollator:
    def __call__(self, batch: list[dict]) -> dict:
        targets = None
        if batch[0]['target'] is not None:
            targets = torch.stack([x['target'] for x in batch])
        numeric_features = {}
        for feat in batch[0]['numeric_features']:
            numeric_features[feat] = torch.stack([x['numeric_features'][feat] for x in batch])
        cat_features = {}
        for feat in batch[0]['cat_features']:
            cat_features[feat] = torch.stack([x['cat_features'][feat] for x in batch])
        return {'target': targets, 'numeric_features': numeric_features, 'cat_features': cat_features}


def load_loan_data(file: Path) -> tuple[LoanDataset, LoanDataset]:
    df = pd.read_csv(file)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    num_features = ['person_age', 'person_income', 'person_emp_length',
                    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                    'cb_person_cred_hist_length']
    cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

    for feat in num_features:
        if df[feat].isna().any():
            df[feat] = df[feat].fillna(df[feat].mean())
    for feat in cat_features:
        if df[feat].isna().any():
            df[feat] = df[feat].fillna("missing")

    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0, 'missing': -1})
    from sklearn.preprocessing import LabelEncoder
    for feat in ['person_home_ownership', 'loan_intent', 'loan_grade']:
        le = LabelEncoder()
        df[feat] = le.fit_transform(df[feat])

    scaler = StandardScaler().fit(df[num_features])
    df[num_features] = scaler.transform(df[num_features])

    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['loan_status'])
    return LoanDataset(df_train, num_features, cat_features), LoanDataset(df_val, num_features, cat_features)


def load_test_data(file: Path) -> LoanDataset:
    df = pd.read_csv(file)
    ids = None
    if 'id' in df.columns:
        ids = df['id']
        df = df.drop(columns=['id'])
    if 'loan_status' not in df.columns:
        df['loan_status'] = 0

    num_features = ['person_age', 'person_income', 'person_emp_length',
                    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                    'cb_person_cred_hist_length']
    cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

    for feat in num_features:
        if df[feat].isna().any():
            df[feat] = df[feat].fillna(df[feat].mean())
    for feat in cat_features:
        if df[feat].isna().any():
            df[feat] = df[feat].fillna("missing")

    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0, 'missing': -1})
    from sklearn.preprocessing import LabelEncoder
    for feat in ['person_home_ownership', 'loan_intent', 'loan_grade']:
        le = LabelEncoder()
        df[feat] = le.fit_transform(df[feat])

    scaler = StandardScaler().fit(df[num_features])
    df[num_features] = scaler.transform(df[num_features])

    dataset = LoanDataset(df, num_features, cat_features)
    if ids is not None:
        dataset.ids = ids
    return dataset
