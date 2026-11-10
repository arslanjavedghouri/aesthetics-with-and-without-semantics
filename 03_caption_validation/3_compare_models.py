# Combines the CLIP similarities of the three models for one generation round.
# A caption is accepted when at least one of its three regenerated images reaches the threshold.
# Output: one row per image (all three scores, best model, accepted yes/no) and a list of the failed images.
import os
import csv
from collections import defaultdict

THRESHOLD = 0.62           # 75% threshold from the 2AFC experiment (04_human_similarity_experiment)
ROUND = 1                  # see 1_generate_images_FLUX.py for the rounds
CAPTIONS_CSV = "results/AVA_subset/captions_AVA.csv"
GENERATED_ROOT = f"results/regenerated/round_{ROUND}"     # <model>/<bin>/comparison_results_clip.csv
MODELS = {"Flux": "flux", "SANA": "sana", "Lumina": "lumina"}
OUT_CSV = f"results/caption_validation/clip_scores_round_{ROUND}.csv"
FAILED_CSV = f"results/caption_validation/failed_after_round_{ROUND}.csv"


def load_scores(path):
    scores = {}
    if not os.path.exists(path):
        return scores
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scores[os.path.splitext(r["Original Image"])[0]] = float(r["CLIP Cosine Similarity"])
    return scores


def load_captions(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    key = "Image ID" if rows and "Image ID" in rows[0] else "Image_Name"
    return {os.path.splitext(r[key].strip())[0]: r.get("Caption", "").strip() for r in rows}


if __name__ == "__main__":
    captions = load_captions(CAPTIONS_CSV)
    rows, failed = [], []
    bins = sorted({b for m in MODELS.values() for b in os.listdir(os.path.join(GENERATED_ROOT, m))})
    for b in bins:
        scores = {name: load_scores(os.path.join(GENERATED_ROOT, folder, b, "comparison_results_clip.csv"))
                  for name, folder in MODELS.items()}
        for image_id in sorted(set().union(*[s.keys() for s in scores.values()])):
            s = {name: scores[name][image_id] for name in MODELS if image_id in scores[name]}
            passed = [name for name, v in s.items() if v >= THRESHOLD]
            best = max(s, key=s.get)
            rows.append({"Image ID": image_id, "Bin": b,
                         **{f"{name} CCS": f"{s[name]:.4f}" if name in s else "" for name in MODELS},
                         "Highest CCS": f"{s[best]:.4f}", "Model": best,
                         "Above threshold": "yes" if passed else "no",
                         "Models pass threshold": ", ".join(passed)})
            if not passed:
                failed.append({"Image ID": image_id, "Bin": b, "Caption": captions.get(image_id, ""),
                               "CLIP Cosine Similarity": f"{s[best]:.4f}"})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(FAILED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Image ID", "Bin", "Caption", "CLIP Cosine Similarity"]); w.writeheader(); w.writerows(failed)

    n = len(rows)
    print(f"{n} images, accepted: {n - len(failed)} ({100 * (n - len(failed)) / n:.2f}%)")
    for name in MODELS:
        k = sum(name in r["Models pass threshold"].split(", ") for r in rows)
        print(f"  {name}: {k} ({100 * k / n:.2f}%) above threshold")
    best_counts = defaultdict(int)
    for r in rows:
        best_counts[r["Model"]] += 1
    print("  highest CCS:", dict(best_counts))
