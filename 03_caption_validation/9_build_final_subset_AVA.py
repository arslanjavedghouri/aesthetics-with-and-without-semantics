# Final AVA subset after caption validation, in the format of the data repository:
#   AVA_subset_images.csv  bins and human rating counts of the images in the subset
#   AVA_captions.csv       the caption used for each image and the challenge name (and description, if available)
# An image keeps its first caption if it was accepted in rounds 1-3, its second caption if accepted in round 4;
# otherwise it is replaced (8_select_replacements.py). Images without a round-1 result are not in the subset.
import os
import re
import pandas as pd

MU_FILE        = "results/AVA_ratings_with_MU.xlsx"            # from 01_dataset_preparation/1_truncated_gaussian_mu_AVA.py
CHALLENGES     = "data/AVA/challenges.txt"
FIRST_CAPTIONS = "results/AVA_subset/captions_AVA.csv"          # from 02_captions/caption_AVA.py
SECOND_CAPTIONS = "results/AVA_second_caption/captions_AVA_second.csv"   # from 6_caption_AVA_second.py
SCORES         = "results/caption_validation/clip_scores_round_{}.csv"   # from 3_compare_models.py, rounds 1-4
REPLACEMENTS   = "results/caption_validation/selected_replacements.csv"  # from 8_select_replacements.py
# Optional: table of challenge names and descriptions collected from the public DPChallenge challenge pages
# ('Challenge ID', 'Challenge Name', 'Challenge Description'); AVA itself gives only ID and name. Not included here:
# the descriptions used in the paper are in the data repository (02_captions/AVA_captions.csv). Without this file,
# names come from challenges.txt and no description column is written.
DESCRIPTIONS   = "data/AVA/challenge_descriptions.csv"
OUT_DIR        = "results/final_subset"


def accepted(round_no):
    s = pd.read_csv(SCORES.format(round_no), dtype={"Image ID": str})
    return set(s["Image ID"]), set(s.loc[s["Above threshold"].str.lower() == "yes", "Image ID"])


def captions(path):
    c = pd.read_csv(path)
    c["Image ID"] = c["Image_Name"].astype(str).str.replace(r"\.\w+$", "", regex=True)
    return c.drop_duplicates("Image ID", keep="last").set_index("Image ID")["Caption"]


if __name__ == "__main__":
    processed, ok1 = accepted(1)
    ok_first = ok1 | accepted(2)[1] | accepted(3)[1]
    ok_second = accepted(4)[1] - ok_first
    replaced = processed - ok_first - ok_second
    repl = pd.read_csv(REPLACEMENTS, dtype={"Image ID": str})
    print(f"round 1: {len(processed)} images; first caption kept: {len(ok_first & processed)}, "
          f"second caption: {len(ok_second)}, replaced: {len(replaced)} by {len(repl)}")

    first, second = captions(FIRST_CAPTIONS), captions(SECOND_CAPTIONS)
    rows = ([(i, "first_caption", first[i]) for i in sorted(ok_first & processed)]
            + [(i, "second_caption_after_failed_validation", second[i]) for i in sorted(ok_second)]
            + [(r["Image ID"], "replacement_image", r["Caption"]) for _, r in repl.iterrows()])
    cap = pd.DataFrame(rows, columns=["image_id", "caption_type", "caption"])

    names = {}
    with open(CHALLENGES, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                names[int(parts[0])] = parts[1].replace("_", " ")

    mu = pd.read_excel(MU_FILE)
    mu["image_id"] = mu["Image ID"].astype(int).astype(str)
    mu = mu.set_index("image_id").loc[cap["image_id"]].reset_index()
    rating_cols = [f"Rating {i}" for i in range(1, 11)]
    images = pd.DataFrame({
        "image_id": mu["image_id"].astype(int),
        "bin_10": mu["MU"].astype(int) + 1,
        "truncated_gaussian_mean": mu["MU"].round(4),
        "challenge_id": mu["Challenge ID"].astype(int),
    })
    images.insert(2, "class_5", (images["bin_10"] + 1) // 2)
    images["challenge_name"] = images["challenge_id"].map(names)
    for i, c in enumerate(rating_cols, 1):
        images[f"human_r{i}"] = mu[c].astype(int)
    images["human_n"] = mu[rating_cols].sum(axis=1).astype(int)

    cap["image_id"] = cap["image_id"].astype(int)
    cap = cap.merge(images[["image_id", "challenge_id", "challenge_name"]], on="image_id")
    if os.path.exists(DESCRIPTIONS):
        # challenges whose page could not be retrieved get the same placeholder text that was given to the model
        d = pd.read_csv(DESCRIPTIONS).set_index("Challenge ID")
        missing = ~cap["challenge_id"].isin(d.index)
        cap["challenge_name"] = cap["challenge_id"].map(d["Challenge Name"]).where(~missing, "No Challenge Name available")
        cap["challenge_description"] = cap["challenge_id"].map(d["Challenge Description"]).where(~missing, "No description available")
        images["challenge_name"] = images["challenge_id"].map(d["Challenge Name"]).where(images["challenge_id"].isin(d.index), "No Challenge Name available")
    cap = cap.drop(columns="challenge_id")

    os.makedirs(OUT_DIR, exist_ok=True)
    images.sort_values(["bin_10", "image_id"]).to_csv(os.path.join(OUT_DIR, "AVA_subset_images.csv"), index=False)
    cap.to_csv(os.path.join(OUT_DIR, "AVA_captions.csv"), index=False)
    print(f"{len(images)} images; bins 1-10:", images["bin_10"].value_counts().sort_index().tolist())
