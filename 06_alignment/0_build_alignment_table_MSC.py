# Table for the MSC alignment scripts from the data repository (same format as the output of
# 05_llm_rating/rate_captions_MSC.py). Not needed if the ratings were collected with that script.
import os
import pandas as pd

IMAGES_CSV   = r"data/01_image_sets/MSC_images.csv"
CAPTIONS_CSV = r"data/02_captions/MSC_captions.csv"
LLM_CSV      = r"data/04_llm_ratings/MSC_LLM_ratings.csv"
OUT_FILE     = r"results/MSC_LLM_ratings.xlsx"

im = pd.read_csv(IMAGES_CSV).merge(pd.read_csv(CAPTIONS_CSV), on="image_id").merge(pd.read_csv(LLM_CSV), on="image_id")
hum = im[[f"human_r{k}" for k in range(1, 6)]].astype(int).values
llm = im[[f"llm_r{k}" for k in range(1, 6)]].astype(int).values
out = pd.DataFrame({"Image ID": im["image_id"], "Bin": im["bin_5"], "Caption": im["caption"],
                    "Total Votes": hum.sum(axis=1), "MSC-Ratings": [str(h.tolist()) for h in hum],
                    "LLM-Rating": [str(h.tolist()) for h in llm]})

if os.path.exists(OUT_FILE):
    print(f"{OUT_FILE} already exists, not overwritten")
else:
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    out.to_excel(OUT_FILE, index=False)
    print(f"{len(out)} images saved to {OUT_FILE}")
