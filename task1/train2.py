from pathlib import Path
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchmetrics import AUROC, MeanMetric
from tqdm import tqdm
import pandas as pd
from dataset import load_loan_data, load_test_data, LoanCollator
from model import LoanApprovalModel


def _to_device(batch: dict, device: torch.device) -> dict:
    if 'target' in batch and batch['target'] is not None:
        batch['target'] = batch['target'].to(device)
    for key in batch['numeric_features']:
        batch['numeric_features'][key] = batch['numeric_features'][key].to(device)
    for key in batch['cat_features']:
        batch['cat_features'][key] = batch['cat_features'][key].to(device)
    return batch


def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    hidden_size = 128
    n_blocks = 3
    use_skip = False
    dropout_p = 0.0
    lr = 0.01
    weight_decay = 0.0
    batch_size = 32
    n_epochs = 10
    seed = 42

    torch.manual_seed(seed)

    label_encoder_path = 'label_encoder.pkl'
    train_dataset, val_dataset = load_loan_data(Path('loan_train.csv'), label_encoder_path)
    collator = LoanCollator()
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collator)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collator)

    num_cat = {feat: train_dataset.data[feat].nunique() for feat in
               ['person_home_ownership', 'loan_intent', 'loan_grade']}
    num_cat['cb_person_default_on_file'] = 2

    input_numeric_dim = len([
        'person_age', 'person_income', 'person_emp_length',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length'
    ])

    model = LoanApprovalModel(
        input_numeric_dim=input_numeric_dim,
        num_cat=num_cat,
        hidden_size=hidden_size,
        n_blocks=n_blocks,
        use_skip=use_skip,
        dropout_p=dropout_p
    )
    model = model.to(device)

    loss_fn = BCEWithLogitsLoss()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        model.train()
        train_loss_metric = MeanMetric().to(device)
        train_auroc = AUROC(task='binary').to(device)
        for batch in tqdm(train_dl, desc=f"Epoch {epoch + 1}"):
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(
                cat_features=batch['cat_features'],
                numeric_features=batch['numeric_features']
            )
            loss = loss_fn(outputs, batch['target'])
            loss.backward()
            optimizer.step()

            train_loss_metric.update(loss)
            train_auroc.update(torch.sigmoid(outputs), batch['target'])

        train_loss = train_loss_metric.compute().item()
        train_auc = train_auroc.compute().item()
        print(f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train AUROC {train_auc:.4f}")

        model.eval()
        val_loss_metric = MeanMetric().to(device)
        val_auroc = AUROC(task='binary').to(device)
        with torch.no_grad():
            for batch in val_dl:
                batch = _to_device(batch, device)
                outputs = model(
                    cat_features=batch['cat_features'],
                    numeric_features=batch['numeric_features']
                )
                loss = loss_fn(outputs, batch['target'])
                val_loss_metric.update(loss)
                val_auroc.update(torch.sigmoid(outputs), batch['target'])
        val_loss = val_loss_metric.compute().item()
        val_auc = val_auroc.compute().item()
        print(f"Epoch {epoch + 1}: Val Loss {val_loss:.4f}, Val AUROC {val_auc:.4f}")

    test_dataset = load_test_data(Path('loan_test.csv'), label_encoder_path)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collator)

    all_predictions = []
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            for key in batch['numeric_features']:
                batch['numeric_features'][key] = batch['numeric_features'][key].to(device)
            for key in batch['cat_features']:
                batch['cat_features'][key] = batch['cat_features'][key].to(device)
            outputs = model(
                cat_features=batch['cat_features'],
                numeric_features=batch['numeric_features']
            )
            preds = torch.sigmoid(outputs)
            all_predictions.extend(preds.cpu().tolist())

    test_df = pd.read_csv(Path('loan_test.csv'))
    if 'id' in test_df.columns:
        test_ids = test_df['id']
    else:
        test_ids = list(range(len(test_dataset)))

    submission = pd.DataFrame({
        'id': test_ids,
        'loan_status': all_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Предсказания сохранены в submission.csv")


if __name__ == '__main__':
    train()
