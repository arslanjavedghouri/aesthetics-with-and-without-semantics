import os
import pandas as pd

# Converts the raw Pavlovia files (one CSV per participant, as downloaded from Pavlovia) into one formatted
# CSV per participant. The raw files are not shared; the formatted trials of all participants are in the
# data repository (2AFC_trials_anonymised.csv), so 2_fit_sigmoid_thresholds.py can be run without this step.
input_folder = r"data/2AFC_pavlovia_raw"
output_folder = r"results/2AFC/formatted"
comparison_file = r"data/05_human_similarity_2AFC/2AFC_stimulus_pairs.csv"   # image pairs, CLIP similarity, AVA bin

os.makedirs(output_folder, exist_ok=True)

# === LOAD IMAGE PAIRS ===
df_comp = pd.read_csv(comparison_file).rename(columns={
    "original_image": "Original Image", "generated_image": "Generated Image",
    "clip_cosine_similarity": "CLIP Cosine Similarity", "bin_10": "bin"})


def normalize(path):
    return os.path.basename(str(path)).strip()


df_comp['Original Image'] = df_comp['Original Image'].apply(normalize)
df_comp['Generated Image'] = df_comp['Generated Image'].apply(normalize)

# === PROCESS EACH PAVLOVIA FILE ===
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        pav_path = os.path.join(input_folder, filename)
        df_pav = pd.read_csv(pav_path)

        # Check necessary columns exist
        required_cols = ['original_image', 'generated_image', 'response', 'reaction_time', 'age', 'gender', 'original_left', 'key_pressed']
        if not all(col in df_pav.columns for col in required_cols):
            print(f"Skipping {filename}: missing required columns.")
            continue

        # Normalize image names
        df_pav['Original Image'] = df_pav['original_image'].apply(normalize)
        df_pav['Generated Image'] = df_pav['generated_image'].apply(normalize)

        # Keep only rows with responses
        df_pav = df_pav[~df_pav['response'].isna()]
        df_pav['Response_Binary'] = df_pav['response'].astype(int)
        df_pav['Response'] = df_pav['Response_Binary'].map({1: 'Similar', 0: 'Not Similar'})

        # Merge with comparison data to get similarity and bin
        merged = pd.merge(
            df_pav,
            df_comp[['Original Image', 'Generated Image', 'CLIP Cosine Similarity', 'bin']],
            on=['Original Image', 'Generated Image'],
            how='inner'
        )

        # Final structure
        final_df = pd.DataFrame({
            'Age': merged['age'],
            'Gender': merged['gender'],
            'Original Image': merged['Original Image'],
            'Generated Image': merged['Generated Image'],
            'CLIP Cosine Similarity': merged['CLIP Cosine Similarity'],
            'Bin': merged['bin'],
            'Original_on_Left': merged['original_left'],
            'Response': merged['Response'],
            'Reaction Time (s)': merged['reaction_time'],
            'Response_Binary': merged['Response_Binary'],
            'Key_Pressed': merged['key_pressed']
        })

        # Save to output folder
        output_path = os.path.join(output_folder, f'{filename}')
        final_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
