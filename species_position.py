import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("./data/processed/taxonomy_and_positions.csv")

# Calculate median fmin and fmax per taxonomy
median_df = df.groupby("NCBI_taxonomy_name")[["fmin", "fmax"]].median().reset_index()

# Apply rounding: floor for fmin, ceil for fmax
median_df["fmin"] = np.floor(median_df["fmin"]).astype(int)
median_df["fmax"] = np.ceil(median_df["fmax"]).astype(int)

# Show the result
print(median_df)

# Save to CSV
median_df.to_csv("./data/final/taxonomy_median_positions.csv", index=False)
