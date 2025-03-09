import pickle
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

NUMERIC_FEATURES = ['person_age', 'person_income', 'person_emp_length',
                    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                    'cb_person_cred_hist_length']
CATEGORICAL_FEATURES = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
LABEL_ENC_FEATURES = ['person_home_ownership', 'loan_intent', 'loan_grade']


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
        categorical = {}
        for feat in self.cat_features:
            if feat in row:
                categorical[feat] = torch.tensor(row[feat], dtype=torch.long)
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


def save_standard_scaler(scaler, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(scaler, f)


def load_standard_scaler(filename: str) -> StandardScaler:
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_label_encoder(le, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(le, f)


def load_label_encoder(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def _fill_missing(df: pd.DataFrame, numeric_features: list[str], cat_features: list[str],
                  numeric_means: list[float] = None) -> pd.DataFrame:
    for i, feat in enumerate(numeric_features):
        if df[feat].isna().any():
            fill_value = numeric_means[i] if numeric_means is not None else df[feat].mean()
            df[feat] = df[feat].fillna(fill_value)
    for feat in cat_features:
        if df[feat].isna().any():
            df[feat] = df[feat].fillna('missing')
    return df


def load_loan_data(file: Path, label_encoder_path: str, scaler_path: str) -> tuple[LoanDataset, LoanDataset]:
    df = pd.read_csv(file)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df = _fill_missing(df, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0, 'missing': -1})
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['loan_status'])
    label_encoders = {}
    for feat in LABEL_ENC_FEATURES:
        le = LabelEncoder()
        df_train[feat] = le.fit_transform(df_train[feat])
        df_val[feat] = le.transform(df_val[feat])
        label_encoders[feat] = le
    save_label_encoder(label_encoders, label_encoder_path)
    scaler = StandardScaler().fit(df_train[NUMERIC_FEATURES])
    save_standard_scaler(scaler, scaler_path)
    df_train[NUMERIC_FEATURES] = scaler.transform(df_train[NUMERIC_FEATURES])
    df_val[NUMERIC_FEATURES] = scaler.transform(df_val[NUMERIC_FEATURES])
    train_dataset = LoanDataset(df_train.reset_index(drop=True), NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    val_dataset = LoanDataset(df_val.reset_index(drop=True), NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    return train_dataset, val_dataset


def load_test_data(file: Path, label_encoder_path: str, scaler_path: str) -> LoanDataset:
    df = pd.read_csv(file)
    ids = None
    if 'id' in df.columns:
        ids = df['id']
        df = df.drop(columns=['id'])
    if 'loan_status' not in df.columns:
        df['loan_status'] = 0
    scaler = load_standard_scaler(scaler_path)
    label_encoders = load_label_encoder(label_encoder_path)
    train_means = scaler.mean_ if hasattr(scaler, 'mean_') else None
    df = _fill_missing(df, NUMERIC_FEATURES, CATEGORICAL_FEATURES, numeric_means=train_means)
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0, 'missing': -1})
    for feat in LABEL_ENC_FEATURES:
        df[feat] = label_encoders[feat].transform(df[feat])
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    test_dataset = LoanDataset(df.reset_index(drop=True), NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    if ids is not None:
        test_dataset.ids = ids
    return test_dataset
