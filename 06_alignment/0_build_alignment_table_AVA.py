# Table for the AVA alignment scripts: one row per image with the human rating
# histogram (AVA-Ratings) and the GPT-4o rating histogram (LLM-Rating).
# LLM ratings come from 05_llm_rating/rate_captions_AVA.py, or from the data repository if that file is not there.
import os
import pandas as pd

LLM_FILE = r"results/AVA_LLM_ratings.xlsx"                   # from 05_llm_rating/rate_captions_AVA.py
LLM_CSV  = r"data/04_llm_ratings/AVA_LLM_ratings.csv"        # data repository
IMAGES_CSV   = r"data/01_image_sets/AVA_subset_images.csv"   # human rating counts human_r1..human_r10
CAPTIONS_CSV = r"data/02_captions/AVA_captions.csv"
OUT_FILE = r"results/combined-human-llm-rating-AVA.xlsx"

HUMAN = [f"human_r{i}" for i in range(1, 11)]
im = pd.read_csv(IMAGES_CSV).set_index("image_id")

if os.path.exists(LLM_FILE):
    llm = pd.read_excel(LLM_FILE)
    ids = llm["Image ID"]
    llm_hist = llm[[f"Rating {i}" for i in range(1, 11)]].astype(int).values
else:
    llm = pd.read_csv(LLM_CSV)
    ids = llm["image_id"]
    llm_hist = llm[[f"llm_r{i}" for i in range(1, 11)]].astype(int).values

cap = pd.read_csv(CAPTIONS_CSV).set_index("image_id")
out = pd.DataFrame({
    "Image ID": ids.values,
    "Challenge ID": im.loc[ids, "challenge_id"].values,
    "Caption": cap.loc[ids, "caption"].values,
    "Challenge Name": cap.loc[ids, "challenge_name"].values,
    "Challenge Description": cap.loc[ids, "challenge_description"].values,
    "Bin": ["Bin-" + str(b - 1) for b in im.loc[ids, "bin_10"]],      # Bin-0 ... Bin-9 = bins 1-10
    "Total Votes Humans": im.loc[ids, "human_n"].values,
    "LLM Ratings Received": llm_hist.sum(axis=1),
    "AVA-Ratings": [str(h.tolist()) for h in im.loc[ids, HUMAN].astype(int).values],
    "LLM-Rating": [str(h.tolist()) for h in llm_hist],
})

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
out.to_excel(OUT_FILE, index=False)
print(f"{len(out)} images saved to {OUT_FILE}")
