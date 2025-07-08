import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.metrics import classification_report, fbeta_score
import numpy as np
import optuna

import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # kalau pakai multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # tradeoff: lebih lambat tapi deterministik

    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

# Cek ketersediaan GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class KmerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class KmerClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden1=512, hidden2=256, dropout=0.3):
        super(KmerClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_size),
        )

    def forward(self, x):
        return self.model(x)

def build_objective(X_train, y_train):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    # Hitung pos_weight dari keseluruhan y_train
    label_counts = y_train.sum(axis=0)
    total = len(y_train)
    pos_weights = torch.tensor((total - label_counts) / (label_counts + 1e-5), dtype=torch.float32).to(device)

    def objective(trial):
        hidden1 = trial.suggest_categorical("hidden1", [256, 512, 1024])
        hidden2 = trial.suggest_categorical("hidden2", [128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        num_epochs = trial.suggest_int("num_epochs", 10, 40)

        dataset = KmerDataset(X_train, y_train)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = KmerClassifier(input_size, output_size, hidden1, hidden2, dropout).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                y_true.append(yb.cpu().numpy())
                y_probs.append(preds.cpu().numpy())

        y_true = np.vstack(y_true)
        y_probs = np.vstack(y_probs)

        best_score = 0
        for t in np.arange(0.4, 0.7, 0.01):
            predictions = (torch.sigmoid(torch.tensor(y_probs)) > t).int().numpy()
            score = fbeta_score(y_true, predictions, average='macro', beta=2, zero_division=0)
            if score > best_score:
                best_score = score

        return best_score

    return objective

train_df = pd.read_csv('./data/for-7mer/train.csv')
val_df = pd.read_csv('./data/for-7mer/val.csv')
test_df = pd.read_csv('./data/for-7mer/test.csv')

X_train = train_df.iloc[:, 1:-11]
y_train = train_df.iloc[:, -11:]
X_val = val_df.iloc[:, 1:-11]
y_val = val_df.iloc[:, -11:]
X_test = test_df.iloc[:, 1:-11]
y_test = test_df.iloc[:, -11:]

input_size = X_train.shape[1]
output_size = y_train.shape[1]

# Hyperparameter tuning
objective_fn = build_objective(X_train, y_train)
study = optuna.create_study(direction="maximize")
study.optimize(objective_fn, n_trials=50)

print("Best Trial Parameters:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")

print(f"\nBest F2-score (macro): {study.best_value:.4f}")

# Gunakan best trial untuk training final model
params = study.best_trial.params