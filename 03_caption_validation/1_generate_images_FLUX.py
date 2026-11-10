# Regenerates an image from each caption with FLUX.1-schnell (default pipeline settings, no fixed seed).
# Output: results/regenerated/round_<ROUND>/flux/<bin>/<image id>.png
from pathlib import Path
import pandas as pd
import torch
from diffusers import FluxPipeline

# Rounds of caption validation (set ROUND and CAPTIONS_CSV for each run):
#   1: all first captions             results/AVA_subset/captions_AVA.csv                  (02_captions/caption_AVA.py)
#   2, 3: images failing round 1      results/caption_validation/failed_after_round_1.csv  (same captions, generated again)
#   4: second captions                results/AVA_second_caption/captions_AVA_second.csv   (6_caption_AVA_second.py)   
#   5: replacement candidates         results/AVA_replacement_candidates/captions_AVA.csv
ROUND = 1
CAPTIONS_CSV = "results/AVA_subset/captions_AVA.csv"
OUT_ROOT = Path(f"results/regenerated/round_{ROUND}/flux")


def load_captions(path):
    # caption files from 02_captions (Image_Name, Class) or failed lists from 3_compare_models.py (Image ID, Bin)
    df = pd.read_csv(path)
    if "Image ID" not in df.columns:
        df["Image ID"] = df["Image_Name"].astype(str).str.replace(r"\.\w+$", "", regex=True)
    if "Class" not in df.columns:
        df["Class"] = df["Bin"]
    return df


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

df = load_captions(CAPTIONS_CSV)
for _, row in df.iterrows():
    out_dir = OUT_ROOT / str(row["Class"])
    out_dir.mkdir(parents=True, exist_ok=True)
    image = pipe(prompt=str(row["Caption"])).images[0]
    image.save(str(out_dir / f"{row['Image ID']}.png"))
