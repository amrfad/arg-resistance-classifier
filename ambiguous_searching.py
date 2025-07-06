import pandas as pd

df = pd.read_csv("./data/processed/sequences_and_labels.csv")

# set huruf ambiguous
ambiguous_letters = set("NRYWSKMBDHV")

# fungsi pengecekan
def has_ambiguous(seq):
    return any(base in ambiguous_letters for base in seq)

# tambahkan kolom baru untuk flag
df["has_ambiguous"] = df["dna_sequence"].apply(has_ambiguous)

# filter baris yang mengandung huruf ambiguous
ambiguous_samples = df[df["has_ambiguous"]]

print(ambiguous_samples['dna_sequence'])

ambiguous_samples.to_csv("./data/processed/ambiguous.csv", index=False)