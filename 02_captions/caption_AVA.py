# GPT-4o captions of the AVA images (first caption). Also used for the replacement candidates.
# Output: <root>/captions_AVA.csv. If the script is stopped, running it again continues where it stopped.
import os
import csv
import base64
import openai   # old API (openai < 1.0)
from tqdm import tqdm

openai.api_key = os.environ["OPENAI_API_KEY"]

IMAGE_ROOTS = [
    "results/AVA_subset",                     # from 01_dataset_preparation/2_ and 3_
    # "results/AVA_replacement_candidates",   # from 03_caption_validation/7_replacement_candidates_AVA.py
]
OUT_NAME = "captions_AVA.csv"

SYSTEM_PROMPT = "You are an expert image captioning assistant..."
USER_PROMPT = (
    "Provide a concise yet vivid description of the image. List all visible elements—people, objects, objects angles and positions, actions, colors, textures, and background. Describe the overall mood or setting if evident. Do not infer or imagine details which are not present in the image but do not skip any information from the image as well. If the image is blank, simply state its color."
)


def caption_image(image_path):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ],
    )
    usage = response.get("usage", {})
    return (response["choices"][0]["message"]["content"].strip(),
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


def collect_images(root):
    # root/Bin-<bin>/<challenge name>/<image>  or  root/<challenge name>/<image>
    items = []
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                rel = os.path.relpath(dirpath, root).split(os.sep)
                bin_name = next((p for p in rel if p.startswith("Bin-")), "")
                challenge = rel[-1] if rel[-1] != bin_name else ""
                items.append((bin_name, challenge, name, os.path.join(dirpath, name)))
    return items


if __name__ == "__main__":
    for root in IMAGE_ROOTS:
        out_csv = os.path.join(root, OUT_NAME)
        done = set()
        if os.path.exists(out_csv):   # continue an interrupted run
            with open(out_csv, encoding="utf-8") as f:
                done = {r["Image_Name"] for r in csv.DictReader(f)}
        items = [it for it in collect_images(root) if it[2] not in done]
        print(f"{root}: {len(items)} images to caption ({len(done)} already done)")

        new_file = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["Image_Name", "Challenge_Name", "Class", "Caption", "Prompt_Tokens", "Completion_Tokens"])
            for bin_name, challenge, name, path in tqdm(items):
                try:
                    caption, pt, ct = caption_image(path)
                    w.writerow([name, challenge, bin_name, caption, pt, ct])
                except Exception as e:
                    print(f"Error for {name}: {e}")
