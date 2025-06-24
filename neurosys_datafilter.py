import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import re

AGAR_ROOT = Path("E:/Subhasis/combined_dataset/neurosys_agar_filtered")
IMG_DIR = AGAR_ROOT / "images"
META_FILE = AGAR_ROOT / "dataset" / "annotations.json"
TRAIN_LIST = AGAR_ROOT / "training_lists" / "higher_resolution_train.txt"
VAL_LIST = AGAR_ROOT / "training_lists" / "higher_resolution_val.txt"
OUT_ROOT = Path("data")  # <-- Model-compatible folder

import re

def read_id_list(txt_path):
    with open(txt_path, "r") as f:
        content = f.read()

    # Use regex to extract all digit strings (ignores invalid stuff)
    id_matches = re.findall(r'"(\d+)"', content)
    return set(int(x) for x in id_matches)


def copy_and_annotate(images, split):
    split_img_dir = OUT_ROOT / split / "images"
    split_img_dir.mkdir(parents=True, exist_ok=True)
    annotations = []

    for img in tqdm(images, desc=f"Processing {split}"):
        file_name = img["file_name"]
        img_id = img["id"]

        src = IMG_DIR / file_name
        dst = split_img_dir / file_name
        if src.exists():
            shutil.copy(src, dst)

        annotations.append({
            "image_name": file_name,
            "colonies": []  # Empty for now — real labels can be inserted later
        })

    anno_path = OUT_ROOT / split / "annotations.json"
    with open(anno_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"✅ Saved {split}/annotations.json with {len(annotations)} entries.")

def main():
    with open(META_FILE, "r") as f:
        meta = json.load(f)

    images = meta["images"]
    train_ids = read_id_list(TRAIN_LIST)
    val_ids = read_id_list(VAL_LIST)

    train_images = [img for img in images if img["id"] in train_ids]
    val_images = [img for img in images if img["id"] in val_ids]

    copy_and_annotate(train_images, "train")
    copy_and_annotate(val_images, "val")

if __name__ == "__main__":
    main()