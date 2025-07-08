import random
import pandas as pd

# Peta IUPAC kode → pilihan nukleotida yang mungkin
IUPAC_MAP = {
    'A': ['A'],
    'C': ['C'],
    'G': ['G'],
    'T': ['T'],
    'R': ['A', 'G'],
    'Y': ['C', 'T'],
    'S': ['G', 'C'],
    'W': ['A', 'T'],
    'K': ['G', 'T'],
    'M': ['A', 'C'],
    'B': ['C', 'G', 'T'],
    'D': ['A', 'G', 'T'],
    'H': ['A', 'C', 'T'],
    'V': ['A', 'C', 'G'],
    'N': ['A', 'C', 'G', 'T']
}

def resolve_ambiguous_bases(seq):
    """Ganti semua huruf ambigu dengan satu huruf acak dari opsi IUPAC-nya."""
    return ''.join(random.choice(IUPAC_MAP.get(base.upper(), ['A'])) for base in seq)

# Contoh DataFrame dengan sekuens
df = pd.read_csv("./data/processed/sequences_and_labels.csv")

# Buat kolom baru berisi sekuens yang sudah dibersihkan
df['dna_sequence'] = df['dna_sequence'].apply(resolve_ambiguous_bases)

df.to_csv('./data/processed/sequences_and_labels_wo_ambiguous.csv', index=False)
