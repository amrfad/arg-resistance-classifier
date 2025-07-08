import pandas as pd
from collections import defaultdict
from typing import Dict, List
from itertools import product

# === Konstanta ===
K = 7
NUCLEOTIDES = ['A', 'T', 'C', 'G']

# === Fungsi Reverse Complement ===
def reverse_complement(seq: str) -> str:
    complement = str.maketrans('ATCG', 'TAGC')
    return seq.translate(complement)[::-1]

# === Fungsi Canonical K-mer ===
def canonical_kmer(kmer: str) -> str:
    rev_kmer = reverse_complement(kmer)
    return min(kmer, rev_kmer)

# === Generate Semua Canonical K-mer ===
def generate_all_canonical_kmers(k: int) -> List[str]:
    all_kmers = {canonical_kmer(''.join(p)) for p in product(NUCLEOTIDES, repeat=k)}
    return sorted(all_kmers)

# Precompute: urutan k-mer tetap
CANONICAL_KMER_ORDER = generate_all_canonical_kmers(K)

# === Hitung Frekuensi K-mer Canonical dari 2 Arah ===
def compute_canonical_kmer_freq_vector(seq: str, k: int = K) -> List[int]:
    freq = defaultdict(int)
    rev_seq = reverse_complement(seq)

    for s in [seq, rev_seq]:
        for i in range(len(s) - k + 1):
            kmer = s[i:i+k]
            if all(nuc in NUCLEOTIDES for nuc in kmer):
                can_kmer = canonical_kmer(kmer)
                freq[can_kmer] += 1

    # Bangun vektor sesuai urutan tetap
    return [freq.get(kmer, 0) for kmer in CANONICAL_KMER_ORDER]

# === Main Processing Row ===
def process_row(row: pd.Series) -> List[int]:
    return compute_canonical_kmer_freq_vector(row['dna_sequence'])

# === Proses Dataset Secara Rapi ===
if __name__ == "__main__":
    # Baca file
    df = pd.read_csv("./data/final/sequences_and_labels_final.csv") 

    # Hitung fitur k-mer
    feature_matrix = df.apply(process_row, axis=1, result_type='expand')
    feature_matrix.columns = CANONICAL_KMER_ORDER  # beri nama kolom sesuai urutan

    # Gabungkan dengan model_id dan label
    final_df = pd.concat([df[['model_id']], feature_matrix, df.drop(columns=['model_id', 'dna_sequence'])], axis=1)

    # Simpan hasil
    final_df.to_csv("./data/for-7mer/7mer_features_clean.csv", index=False)

    print("✅ K-mer features dengan urutan reproducible selesai dibuat.")
