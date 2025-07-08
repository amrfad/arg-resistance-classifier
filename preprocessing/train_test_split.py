import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Load dataset
df = pd.read_csv("./data/final/sequences_and_labels_final.csv")

# Pisahkan fitur dan label
y = df.iloc[:, -11:]
X = df.iloc[:, :-11]

# Konversi label ke array
y_np = y.values

# Langkah 1: Bagi jadi 90% trainval dan 10% test
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
trainval_idx, test_idx = next(mskf.split(X, y_np))

X_trainval = X.iloc[trainval_idx]
y_trainval = y.iloc[trainval_idx]
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Langkah 2: Bagi trainval jadi 90% train dan 10% val (berarti 81% dan 9% dari total)
mskf_inner = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=123)
train_idx, val_idx = next(mskf_inner.split(X_trainval, y_trainval))

X_train = X_trainval.iloc[train_idx]
y_train = y_trainval.iloc[train_idx]
X_val = X_trainval.iloc[val_idx]
y_val = y_trainval.iloc[val_idx]

# ✅ Output info
print(f"Total data: {len(df)}")
print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

# Kalau mau simpan jadi file
X_train.join(y_train).to_csv("./data/for-bpe/train.csv", index=False)
X_val.join(y_val).to_csv("./data/for-bpe/val.csv", index=False)
X_test.join(y_test).to_csv("./data/for-bpe/test.csv", index=False)
