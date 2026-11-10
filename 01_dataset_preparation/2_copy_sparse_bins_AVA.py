# Bins 1-3 and 9-10 have few images, so all of them go into the subset.
# Images are copied to results/AVA_subset/Bin-<bin>/<challenge name>/<image id>.jpg
import os
import re
import shutil
import pandas as pd

SPARSE_BINS = [1, 2, 3, 9, 10]

excel_file = "results/AVA_ratings_with_MU.xlsx"    # from 1_truncated_gaussian_mu_AVA.py
image_folder = "data/AVA/images"
category_file = "data/AVA/challenges.txt"          # challenge ID and name, one per line
out_root = "results/AVA_subset"

df = pd.read_excel(excel_file)

category_mapping = {}
with open(category_file, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            challenge_id, challenge_name = parts
            category_mapping[int(challenge_id)] = re.sub(r"[^\w\s]", "_", challenge_name)

for b in SPARSE_BINS:
    rows = df[df["MU"].apply(lambda x: int(x) + 1 == b)]
    print(f"Bin-{b}: {len(rows)} images")
    for _, row in rows.iterrows():
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
