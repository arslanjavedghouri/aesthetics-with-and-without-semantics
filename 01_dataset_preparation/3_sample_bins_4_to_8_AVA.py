# Bins 4-8: 1,860 random images per bin (random_state=42).
# Images are copied to results/AVA_subset/Bin-<bin>/<challenge name>/<image id>.jpg
import os
import re
import shutil
import pandas as pd

BINS = [4, 5, 6, 7, 8]
N_PER_BIN = 1860

excel_file = "results/AVA_ratings_with_MU.xlsx"    # from 1_truncated_gaussian_mu_AVA.py
image_folder = "data/AVA/images"
category_file = "data/AVA/challenges.txt"
out_root = "results/AVA_subset"

df = pd.read_excel(excel_file)

category_mapping = {}
with open(category_file, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            challenge_id, challenge_name = parts
            category_mapping[int(challenge_id)] = re.sub(r"[^\w\s]", "_", challenge_name)


def in_bin(x, b):
    try:
        return int(float(x)) + 1 == b
    except (ValueError, TypeError):
        return False


for b in BINS:
    rows = df[df["MU"].apply(lambda x: in_bin(x, b))]
    sampled = rows.sample(n=min(N_PER_BIN, len(rows)), random_state=42)
    print(f"Bin-{b}: {len(sampled)} of {len(rows)} images")

    for _, row in sampled.iterrows():
        image_id = str(int(row["Image ID"]))
        challenge_id = int(row["Challenge ID"])
        if challenge_id not in category_mapping:
            print(f"Challenge ID {challenge_id} not found")
            continue
        challenge_folder = os.path.join(out_root, f"Bin-{b}", category_mapping[challenge_id])
        os.makedirs(challenge_folder, exist_ok=True)
        image_path = os.path.join(image_folder, f"{image_id}.jpg")
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(challenge_folder, f"{image_id}.jpg"))
        else:
            print(f"Image not found: {image_id}.jpg")

print("done")
