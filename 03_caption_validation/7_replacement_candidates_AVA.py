# Candidate replacements for the images that still failed validation with the second caption.
# For one bin, takes N_CANDIDATES images of that bin (in table order), leaving out the failed images.
# The candidates are then captioned (02_captions/caption_AVA.py) and validated like the other images;
# 8_select_replacements.py keeps, per bin, the accepted candidates with the highest
# CLIP similarity that are not already in the subset.
import os
import re
import shutil
import pandas as pd

BIN = 4                 # run once per bin (4-8)
N_CANDIDATES = 200

excel_file = "results/AVA_ratings_with_MU.xlsx"   # from 01_dataset_preparation/1_truncated_gaussian_mu_AVA.py
image_folder = "data/AVA/images"
category_file = "data/AVA/challenges.txt"
failed_csv = "results/caption_validation/failed_after_round_4.csv"   # from 3_compare_models.py, ROUND = 4
out_dir = f"results/AVA_replacement_candidates/Bin-{BIN}"

df = pd.read_excel(excel_file)
excluded = set(pd.read_csv(failed_csv)["Image ID"].astype(str))

category_mapping = {}
with open(category_file, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            challenge_id, challenge_name = parts
            category_mapping[int(challenge_id)] = re.sub(r"[^\w\s]", "_", challenge_name)

rows = df[df["MU"].apply(lambda x: int(x) + 1 == BIN)].copy()
rows["id_str"] = rows["Image ID"].astype(int).astype(str)
rows = rows[~rows["id_str"].isin(excluded)].head(N_CANDIDATES)
print(f"Bin-{BIN}: {len(rows)} candidates")

copied = []
for _, row in rows.iterrows():
    image_id = row["id_str"]
    challenge_id = int(row["Challenge ID"])
    if challenge_id not in category_mapping:
        print(f"Challenge ID {challenge_id} not found")
        continue
    challenge_folder = os.path.join(out_dir, category_mapping[challenge_id])
    os.makedirs(challenge_folder, exist_ok=True)
    image_path = os.path.join(image_folder, f"{image_id}.jpg")
    if os.path.exists(image_path):
        shutil.copy(image_path, os.path.join(challenge_folder, f"{image_id}.jpg"))
        copied.append(image_id)
    else:
        print(f"Image not found: {image_id}.jpg")

pd.DataFrame({"Image ID": copied}).to_csv(os.path.join(out_dir, "candidates.csv"), index=False)
print(f"{len(copied)} images copied to {out_dir}")
