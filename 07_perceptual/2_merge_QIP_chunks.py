import os
import pandas as pd

# Merges the chunk CSVs written by 1_QIP_extraction_parallel.py.
chunk_dir = "results/QIP/chunks/"
output_dir = "results/QIP/"
output_base_name = "AVA_QIP"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Collect all chunk CSV files
csv_files = sorted([
    os.path.join(chunk_dir, f)
    for f in os.listdir(chunk_dir)
    if f.startswith("AVA_QIP_chunk") and f.endswith(".csv")
])

# Merge all CSV files into one DataFrame
merged_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Save
csv_output_path = os.path.join(output_dir, f"{output_base_name}.csv")

merged_df.to_csv(csv_output_path, index=False)

print(f"Merged {len(csv_files)} CSV files into {csv_output_path}")
