# final_training_retrain.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import classification_report, fbeta_score
import numpy as np
import random
import os

# ======== ⚙️ Konfigurasi ========
USE_VALIDATION_FOR_TRAINING = True  # ← Jika True, gabungkan val ke train setelah memilih threshold

# ======== 🧪 Deterministik untuk eksperimen reproducible ========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

# ======== 🚀 Setup Device ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======== 📦 Dataset dan Model ========
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

# ======== 📂 Load Data ========
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

params = {
    'hidden1': 1024, 'hidden2': 128, 'dropout': 0.38365961722627806,
    'lr': 0.0011145769086119111, 'batch_size': 128, 'num_epochs': 38
}

# ======== 🏋️‍♀️ Training Awal (pakai val) ========
model = KmerClassifier(input_size, output_size, params['hidden1'], params['hidden2'], params['dropout']).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5), dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

train_loader = DataLoader(KmerDataset(X_train, y_train), batch_size=params['batch_size'], shuffle=True)
val_loader = DataLoader(KmerDataset(X_val, y_val), batch_size=params['batch_size'])

best_f2_score = 0
best_state_dict = None
best_epoch = 0  # ⬅️ menyimpan epoch terbaik

for epoch in range(params['num_epochs']):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    all_outputs, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = torch.sigmoid(model(X_batch))
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())

    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)

    thresholds = np.zeros(output_size)
    for i in range(output_size):
        best_t, best_s = 0.5, 0
        for t in np.arange(0.3, 0.71, 0.01):
            pred = (all_outputs[:, i] > t).astype(int)
            score = fbeta_score(all_targets[:, i], pred, average='binary', beta=2, zero_division=0)
            if score > best_s:
                best_s, best_t = score, t
        thresholds[i] = best_t

    preds = (all_outputs > thresholds).astype(int)
    macro_f2 = fbeta_score(all_targets, preds, average='macro', beta=2, zero_division=0)
    print(f"Epoch {epoch+1}: F2-score (macro) = {macro_f2:.4f}")

    if macro_f2 > best_f2_score:
        best_f2_score = macro_f2
        best_state_dict = model.state_dict()
        best_epoch = epoch + 1  # ⬅️ update best epoch
        torch.save(best_state_dict, "./model/best_model.pt")

# ======== 💾 Simpan Threshold dan Model ========
np.save("./model/bag-of-kmers/thresholds.npy", thresholds)

# ======== 🔁 Retraining Opsional pada train+val ========
if USE_VALIDATION_FOR_TRAINING:
    RETRAIN_EPOCHS = best_epoch  # ⬅️ retrain sesuai epoch terbaik
    print(f"\n🔁 Retraining on full (train + val) for {RETRAIN_EPOCHS} epochs...")
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    full_loader = DataLoader(KmerDataset(X_trainval, y_trainval), batch_size=params['batch_size'], shuffle=True)
    model = KmerClassifier(input_size, output_size, params['hidden1'], params['hidden2'], params['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    for epoch in range(RETRAIN_EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in full_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Retrain Epoch {epoch+1}] Loss: {total_loss / len(full_loader):.4f}")

    torch.save(model.state_dict(), "./model/bag-of-kmers/model_retrained.pt")
else:
    model.load_state_dict(torch.load("./model/best_model.pt"))

# ======== 🧪 Evaluasi pada Test Set ========
model.eval()
thresholds = np.load("./model/bag-of-kmers/thresholds.npy")
test_loader = DataLoader(KmerDataset(X_test, y_test), batch_size=64)

y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = torch.sigmoid(model(X_batch)).cpu().numpy()
        pred = np.zeros_like(outputs)
        for i in range(output_size):
            pred[:, i] = (outputs[:, i] > thresholds[i]).astype(int)
        y_true.extend(y_batch.numpy())
        y_pred.extend(pred)

print("\n📊 Final Evaluation on Test Set:")
print("F2-score (macro):", fbeta_score(y_true, y_pred, average='macro', beta=2, zero_division=0))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=y_train.columns.tolist(), zero_division=0))

f2_per_label = fbeta_score(y_true, y_pred, average=None, beta=2, zero_division=0)
print("\nF2-score per label:")
for i, score in enumerate(f2_per_label):
    print(f"{y_train.columns.tolist()[i]}: {score:.4f}")