# Chooses the replacement images: among the candidates of a bin whose caption was accepted
# (7_replacement_candidates_AVA.py, then 02_captions/caption_AVA.py, 1_generate_images_*.py,
# 2_clip_similarity.py and 3_compare_models.py), keep those with the highest CLIP similarity that are
# not already in the subset. N_REPLACE = number of images to replace in each bin.
import os
import pandas as pd

CANDIDATE_SCORES = "results/caption_validation/clip_scores_round_5.csv"   # from 3_compare_models.py, ROUND = 5
CAPTIONS_CSV = "results/AVA_replacement_candidates/captions_AVA.csv"      # from 02_captions/caption_AVA.py
SUBSET_DIR = "results/AVA_subset"         # images already in the subset (01_dataset_preparation/2_ and 3_)
OUT_CSV = "results/caption_validation/selected_replacements.csv"

N_REPLACE = {4: 179, 5: 91, 6: 73, 7: 40, 8: 30}

scores = pd.read_csv(CANDIDATE_SCORES, dtype={"Image ID": str})
in_subset = {os.path.splitext(f)[0] for _, _, files in os.walk(SUBSET_DIR) for f in files}
captions = pd.read_csv(CAPTIONS_CSV)
captions["Image ID"] = captions["Image_Name"].str.replace(r"\.\w+$", "", regex=True)

scores = scores[(scores["Above threshold"] == "yes") & (~scores["Image ID"].isin(in_subset))]
selected = []
for b, n in N_REPLACE.items():
    cand = scores[scores["Bin"] == f"Bin-{b}"].sort_values("Highest CCS", ascending=False)
    if len(cand) < n:
        print(f"Bin-{b}: only {len(cand)} accepted candidates for {n} replacements")
    selected.append(cand.head(n))
    print(f"Bin-{b}: {min(n, len(cand))} selected, lowest CCS {cand.head(n)['Highest CCS'].min():.3f}")

out = pd.concat(selected).merge(captions[["Image ID", "Challenge_Name", "Caption"]], on="Image ID", how="left")
out.to_csv(OUT_CSV, index=False)
print(f"saved {len(out)} images to {OUT_CSV}")
