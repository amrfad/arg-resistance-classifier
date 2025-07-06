import pandas as pd

# Misal df adalah DataFrame yang sudah dibaca
df = pd.read_csv("./data/processed/sequences_and_labels_wo_ambiguous.csv")

# Kolom non-label yang ingin dikecualikan
non_label_cols = ['model_id', 'dna_sequence']

# Pisahkan kolom label
label_cols = [col for col in df.columns if col not in non_label_cols]

# Hitung jumlah positif (1) per label
label_counts = df[label_cols].sum(axis=0)

# Filter label dengan >=5 sampel positif
selected_labels = label_counts[label_counts >= 5].index.tolist()
removed_labels = label_counts[label_counts < 5].index.tolist()
print(f"{len(removed_labels)} kolom yang dihapus karena terlalu jarang:", removed_labels)

# Buat DataFrame baru dengan label yang lolos saja (dan kolom metadata)
df = df[non_label_cols + selected_labels]

df.to_csv('./data/final/sequences_and_labels_final.csv', index=False)
