# Images whose caption failed validation in all three generation rounds get a second, more detailed caption.
# Copies them to results/AVA_second_caption/Bin-<bin>/<challenge name>/<image id>.jpg
import os
import re
import shutil
import pandas as pd

failed_csv = "results/caption_validation/failed_after_round_3.csv"   # from 4_remaining_failed_images.py
excel_file = "results/AVA_ratings_with_MU.xlsx"   # from 01_dataset_preparation/1_truncated_gaussian_mu_AVA.py
image_folder = "data/AVA/images"
category_file = "data/AVA/challenges.txt"
out_root = "results/AVA_second_caption"

failed_ids = set(pd.read_csv(failed_csv)["Image ID"].astype(int))
print(f"{len(failed_ids)} images")
df = pd.read_excel(excel_file)
df = df[df["Image ID"].astype(int).isin(failed_ids)]

category_mapping = {}
with open(category_file, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            challenge_id, challenge_name = parts
            category_mapping[int(challenge_id)] = re.sub(r"[^\w\s]", "_", challenge_name)

copied = 0
for _, row in df.iterrows():
    image_id = str(int(row["Image ID"]))
    b = int(float(row["MU"])) + 1
    challenge_id = int(row["Challenge ID"])
    if challenge_id not in category_mapping:
        print(f"Challenge ID {challenge_id} not found")
        continue
    challenge_folder = os.path.join(out_root, f"Bin-{b}", category_mapping[challenge_id])
    os.makedirs(challenge_folder, exist_ok=True)
    image_path = os.path.join(image_folder, f"{image_id}.jpg")
    if os.path.exists(image_path):
        shutil.copy(image_path, os.path.join(challenge_folder, f"{image_id}.jpg"))
        copied += 1
    else:
        print(f"Image not found: {image_id}.jpg")

print(f"copied {copied} images")
