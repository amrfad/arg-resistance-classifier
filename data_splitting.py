import pandas as pd

# Load the full dataset from CSV
input_csv_path = "./data/processed/extracted_data.csv"
df = pd.read_csv(input_csv_path)

# Identify antibiotic columns (assume they are all columns after 'dna_sequence')
antibiotic_columns = df.columns.tolist()
first_abx_index = antibiotic_columns.index("dna_sequence") + 1
antibiotic_columns = antibiotic_columns[first_abx_index:]

# Split into two DataFrames
df_sequences = df[["model_id", "dna_sequence"] + antibiotic_columns]
df_metadata = df[["NCBI_taxonomy_name", "fmin", "fmax"]]

# Save to separate CSV files
df_sequences.to_csv("./data/processed/sequences_and_labels.csv", index=False)
df_metadata.to_csv("./data/processed/taxonomy_and_positions.csv", index=False)

print("✅ Files saved:")
print("- sequences_and_labels.csv")
print("- taxonomy_and_positions.csv")
