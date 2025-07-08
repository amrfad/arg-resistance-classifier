import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import classification_report, fbeta_score
import numpy as np
import random
import os

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ["PYTHONHASHSEED"] = str(seed)

# set_seed(42)

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

# Load full dataset
full_df = pd.read_csv('./data/for-7mer/7mer_features_clean.csv')
X_full = full_df.iloc[:, 1:-11]
y_full = full_df.iloc[:, -11:]

input_size = X_full.shape[1]
output_size = y_full.shape[1]

params = {'hidden1': 1024, 'hidden2': 128, 'dropout': 0.38365961722627806, 'lr': 0.0011145769086119111, 'batch_size': 128, 'num_epochs': 15}
model = KmerClassifier(input_size, output_size, params['hidden1'], params['hidden2'], params['dropout']).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_full) - y_full.sum()) / (y_full.sum() + 1e-5), dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

dataset = KmerDataset(X_full, y_full)
dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

model.train()
for epoch in range(params['num_epochs']):
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{params['num_epochs']} | Loss: {avg_loss:.4f}")

# Simpan model final
torch.save(model.state_dict(), "./model/final/bok_model.pt")
print("✓ Final model saved.")