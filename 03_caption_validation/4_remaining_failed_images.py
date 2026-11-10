# Rounds 2 and 3 regenerate the images that failed in round 1 with the same captions.
# Images still failing after round 3 get a second caption (5_copy_images_for_second_caption_AVA.py).
import pandas as pd

FAILED_ROUND_1 = "results/caption_validation/failed_after_round_1.csv"      # from 3_compare_models.py, ROUND = 1
LATER_ROUNDS = ["results/caption_validation/clip_scores_round_2.csv",
                "results/caption_validation/clip_scores_round_3.csv"]
OUT_CSV = "results/caption_validation/failed_after_round_3.csv"

failed = pd.read_csv(FAILED_ROUND_1, dtype={"Image ID": str})
print(f"failed in round 1: {len(failed)}")
for path in LATER_ROUNDS:
    cur = pd.read_csv(path, dtype={"Image ID": str})
    accepted = set(cur.loc[cur["Above threshold"].str.lower() == "yes", "Image ID"])
    n = len(failed)
    failed = failed[~failed["Image ID"].isin(accepted)]
    print(f"{path}: {n - len(failed)} accepted, {len(failed)} still failing")
failed.to_csv(OUT_CSV, index=False)
