# QIP table for exactly the AVA images of the language-mediated pathway (same images and classes).
# Image list and classes from AVA_subset_images.csv (data repository, or 03_caption_validation/9_build_final_subset_AVA.py);
# QIP features from the table of all AVA images, matched on image ID.
# Output: img_file, Class Name, QIP features.

import os
import numpy as np
import pandas as pd

SUBSET_CSV   = r"data/01_image_sets/AVA_subset_images.csv"
FULL_QIP_CSV = r"results/QIP/AVA_QIP_with_class.csv"   # from 3_class_labels_AVA.py
OUT_CSV      = r"results/QIP/AVA_QIP_matched_to_LLM_set.csv"

QIP_ID_COL, QIP_CLASS_COL = "img_file", "Class Name"


def norm_id(x):
    s = str(x).strip().replace("\\", "/")
    s = os.path.splitext(os.path.basename(s))[0]
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def main():
    llm = pd.read_csv(SUBSET_CSV)
    llm["id"] = llm["image_id"].map(norm_id)
    llm["llm_class"] = llm["class_5"].astype(int)
    assert not llm["id"].duplicated().any(), "duplicate image IDs"
    print(f"AVA subset: {len(llm)} images")

    full = pd.read_csv(FULL_QIP_CSV)
    full["id"] = full[QIP_ID_COL].map(norm_id)
    n_dup = int(full["id"].duplicated().sum())
    print(f"QIP table of all AVA images: {len(full)} rows, {n_dup} duplicate IDs")
    full = full.drop_duplicates("id")

    merged = llm[["id", "llm_class"]].merge(full, on="id", how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"]
    print(f"\nimages found in the QIP table: {len(merged) - len(missing)} / {len(llm)}")
    if len(missing):
        print(f"{len(missing)} images have no QIP features:")
        print(missing["id"].head(20).tolist())
        missing[["id", "llm_class"]].to_csv(OUT_CSV.replace(".csv", "_MISSING.csv"), index=False)
        merged = merged[merged["_merge"] == "both"]

    # check that both tables give the same class
    if QIP_CLASS_COL in merged.columns:
        diff = int((merged[QIP_CLASS_COL].astype(int) != merged["llm_class"]).sum())
        print(f"images whose class differs between the two tables: {diff}")

    out = merged.drop(columns=["_merge", "id"])
    out[QIP_CLASS_COL] = out["llm_class"].astype(int)
    out = out.drop(columns=["llm_class"])
    cols = [QIP_ID_COL, QIP_CLASS_COL] + [c for c in out.columns if c not in (QIP_ID_COL, QIP_CLASS_COL)]
    out = out[cols]
    out.to_csv(OUT_CSV, index=False)

    counts = out[QIP_CLASS_COL].value_counts().sort_index()
    llm_counts = llm["llm_class"].value_counts().sort_index()
    print(f"\nSaved {len(out)} images to: {OUT_CSV}")
    print("class counts (QIP table | AVA subset):")
    for k in range(1, 6):
        print(f"  class {k}: {counts.get(k, 0):5d} | {llm_counts.get(k, 0):5d}")
    ok = len(missing) == 0 and (counts.reindex(range(1, 6)).values == llm_counts.reindex(range(1, 6)).values).all()
    print(f"\nSame images and classes as the language-mediated pathway: {'yes' if ok else 'no'}")


if __name__ == "__main__":
    main()