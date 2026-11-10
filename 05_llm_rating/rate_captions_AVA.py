# GPT-4o aesthetic ratings (1-10) for the AVA captions, with the challenge name and description.
# Each rating is a separate API call. Results are saved regularly; if the script is stopped,
# running it again continues from OUT_FILE.
# OPENAI_API_KEY has to be set.

import os, re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import openai   # old API (openai < 1.0)

openai.api_key = os.environ["OPENAI_API_KEY"]

# ratings per image: None = same number as human ratings, or a fixed number
NUM_RATINGS = None
# NUM_RATINGS = 100
MAX_WORKERS = 10
SAVE_EVERY = 25


def ask(prompt):
    for _ in range(5):
        try:
            r = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            m = re.findall(r"\b\d+\b", r["choices"][0]["message"]["content"].strip())
            return int(m[-1]) if m else None
        except openai.error.RateLimitError:
            time.sleep(10)
        except Exception as e:
            print("Error:", e)
            return None
    return None


def get_ratings(prompt, n, scale):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        out = [f.result() for f in as_completed([ex.submit(ask, prompt) for _ in range(n)])]
    return [r for r in out if r is not None and 1 <= r <= scale]

IMAGES_CSV   = r"data/01_image_sets/AVA_subset_images.csv"
CAPTIONS_CSV = r"data/02_captions/AVA_captions.csv"
OUT_FILE = r"results/AVA_LLM_ratings.xlsx"
COLS = [f"Rating {i}" for i in range(1, 11)]


def make_prompt(caption, challenge_name, challenge_description):
    return (
        f"You are an expert in aesthetics and will rate captions generated from images.\n"
        f"Challenge Name: {challenge_name}\n"
        f"Challenge Description: {challenge_description}\n\n"
        f"Task: Evaluate the caption based on two independent criteria:\n"
        f"1. Alignment: Does the caption accurately reflect the Challenge Name and Challenge Description?\n"
        f"2. Aesthetic Appeal: Independently assess the aesthetic beauty or ugliness described by the caption.\n\n"
        f"Important: Even if a caption is perfectly aligned, it should not automatically receive a high aesthetic score.\n"
        f"Strictly rate the overall aesthetic appeal of the scenario described in the caption on a scale from 1 to 10 (Ugly to Beautiful), "
        f"considering alignment as a necessary condition but NOT sufficient on its own for a high rating.\n\n"
        f"Provide only a single numeric score as your response. Do not include any explanation, reasoning, or additional text.\n\n"
        f"Caption:\n{caption}"
    )


if __name__ == "__main__":
    if os.path.exists(OUT_FILE):
        df = pd.read_excel(OUT_FILE)
    else:
        im = pd.read_csv(IMAGES_CSV); cap = pd.read_csv(CAPTIONS_CSV)
        m = im.drop(columns=["challenge_name"], errors="ignore").merge(cap, on="image_id")
        df = pd.DataFrame({"Image ID": m["image_id"], "Bin": "Bin-" + (m["bin_10"] - 1).astype(str), "Challenge ID": m["challenge_id"],
                           "Challenge Name": m["challenge_name"], "Challenge Description": m["challenge_description"],
                           "Caption": m["caption"], "Total Votes Humans": m["human_n"]})
    for c in COLS:
        if c not in df.columns:
            df[c] = 0
    target = df["Total Votes Humans"] if NUM_RATINGS is None else pd.Series(NUM_RATINGS, index=df.index)
    remaining = [i for i in df.index if df.loc[i, COLS].sum() < target[i]]
    print(f"{len(remaining)} of {len(df)} images still need ratings")
    for k, i in enumerate(remaining, 1):
        need = int(target[i] - df.loc[i, COLS].sum())
        p = make_prompt(df.at[i, "Caption"], df.at[i, "Challenge Name"], df.at[i, "Challenge Description"])
        cnt = Counter(get_ratings(p, need, 10))
        for r in range(1, 11):
            df.at[i, f"Rating {r}"] = int(df.at[i, f"Rating {r}"]) + cnt.get(r, 0)
        print(f"[{k}/{len(remaining)}] {df.at[i, 'Image ID']}: {sum(cnt.values())} ratings")
        if k % SAVE_EVERY == 0:
            df.to_excel(OUT_FILE, index=False)
    df.to_excel(OUT_FILE, index=False)
    print("saved", OUT_FILE)
