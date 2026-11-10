# CLIP cosine similarity between each original image and the image regenerated from its caption.
# CLIP image embeddings from OpenCLIP ViT-B-32, loaded through SpLiCE (https://github.com/AI4LIFE-GROUP/SpLiCE).
# Writes results/regenerated/round_<ROUND>/<model>/<bin>/comparison_results_clip.csv
import os
import csv
import torch
import splice
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

ORIGINAL_DIR = "data/AVA/images"
ROUND = 1
GENERATED_ROOT = f"results/regenerated/round_{ROUND}"   # from 1_generate_images_*.py
MODELS = ["flux", "sana", "lumina"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "open_clip:ViT-B-32"
model = splice.load(model_name, "laion", 10000, device, l1_penalty=0.05, return_weights=True, download_root="data")
preprocess = splice.get_preprocess(model_name)


def load_image(path):
    return preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def clip_similarity(orig_path, gen_path):
    with torch.no_grad():
        e1 = model.clip.encode_image(load_image(orig_path)).cpu()
        e2 = model.clip.encode_image(load_image(gen_path)).cpu()
    e1 = torch.nn.functional.normalize(e1, dim=1)
    e2 = torch.nn.functional.normalize(e2, dim=1)
    return float(cosine_similarity(e1.numpy(), e2.numpy())[0][0])


def find_original(stem):
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(ORIGINAL_DIR, stem + ext)
        if os.path.exists(p):
            return p
    return None


for m in MODELS:
    for bin_name in sorted(os.listdir(os.path.join(GENERATED_ROOT, m))):
        gen_dir = os.path.join(GENERATED_ROOT, m, bin_name)
        if not os.path.isdir(gen_dir):
            continue
        out_csv = os.path.join(gen_dir, "comparison_results_clip.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Original Image", "Generated Image", "CLIP Cosine Similarity"])
            w.writeheader()
            for fname in sorted(os.listdir(gen_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                orig = find_original(os.path.splitext(fname)[0])
                if orig is None:
                    print(f"Original image not found for {fname}")
                    continue
                try:
                    w.writerow({"Original Image": os.path.basename(orig), "Generated Image": fname,
                                "CLIP Cosine Similarity": clip_similarity(orig, os.path.join(gen_dir, fname))})
                except Exception as e:
                    print(f"Error for {fname}: {e}")
        print("saved", out_csv)
