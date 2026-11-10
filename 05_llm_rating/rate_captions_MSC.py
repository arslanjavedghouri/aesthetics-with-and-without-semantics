# GPT-4o aesthetic ratings (1-5) for the MSC captions.
# Each rating is a separate API call. Results are saved regularly; if the script is stopped,
# running it again continues from OUT_FILE.
# OPENAI_API_KEY has to be set.

import os, re, time, ast
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

IMAGES_CSV   = r"data/01_image_sets/MSC_images.csv"
CAPTIONS_CSV = r"data/02_captions/MSC_captions.csv"
OUT_FILE     = r"results/MSC_LLM_ratings.xlsx"


def make_prompt(caption):
    return (
        f"You are an expert in aesthetics and will rate captions generated from images.\n"
        f"Independently assess the aesthetic beauty or ugliness described by the caption.\n\n"
        f"Strictly rate the overall aesthetic appeal of the scenario described in the caption on a scale from 1 to 5 (Ugly to Beautiful), "
        f"Provide only a single numeric score as your response. Do not include any explanation, reasoning, or additional text.\n\n"
        f"Caption:\n{caption}"
    )


def build_table():
    im = pd.read_csv(IMAGES_CSV).merge(pd.read_csv(CAPTIONS_CSV), on="image_id")
    hum = im[[f"human_r{k}" for k in range(1, 6)]].values.astype(int)
    return pd.DataFrame({"Image ID": im["image_id"], "Bin": im["bin_5"], "Caption": im["caption"],
                         "Total Votes": hum.sum(axis=1), "MSC-Ratings": [str(h.tolist()) for h in hum],
                         "LLM-Rating": [str([0] * 5)] * len(im)})


if __name__ == "__main__":
    df = pd.read_excel(OUT_FILE) if os.path.exists(OUT_FILE) else build_table()
    llm = [ast.literal_eval(x) for x in df["LLM-Rating"]]
    target = list(df["Total Votes"]) if NUM_RATINGS is None else [NUM_RATINGS] * len(df)
    remaining = [i for i in range(len(df)) if sum(llm[i]) < target[i]]
    print(f"{len(remaining)} of {len(df)} images still need ratings")
    for k, i in enumerate(remaining, 1):
        cnt = Counter(get_ratings(make_prompt(df.at[i, "Caption"]), int(target[i] - sum(llm[i])), 5))
        llm[i] = [llm[i][j] + cnt.get(j + 1, 0) for j in range(5)]
        print(f"[{k}/{len(remaining)}] {df.at[i, 'Image ID']}: {sum(cnt.values())} ratings")
        if k % SAVE_EVERY == 0:
            df["LLM-Rating"] = [str(x) for x in llm]; df.to_excel(OUT_FILE, index=False)
    df["LLM-Rating"] = [str(x) for x in llm]
    df.to_excel(OUT_FILE, index=False)
    print("saved", OUT_FILE)
