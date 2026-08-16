"""
Converts the Kaggle "Car Object Detection" dataset (images + CSV boxes,
same data used in Part 1) into the folder/label format Ultralytics YOLO
expects for fine-tuning.

Input (expected, same as Part 1):
  data/training_images/*.jpg
  data/train_solution_bounding_boxes.csv   (image, xmin, ymin, xmax, ymax)

Output:
  data/yolo_dataset/images/train/*.jpg
  data/yolo_dataset/images/val/*.jpg
  data/yolo_dataset/labels/train/*.txt
  data/yolo_dataset/labels/val/*.txt
  data/yolo_dataset/data.yaml

YOLO label format (one .txt per image, one line per box):
  class_id  x_center  y_center  width  height      (all normalized 0-1)

Since there's only one class here ("vehicle"), class_id is always 0.
"""
import os
import random
import shutil

import pandas as pd
from PIL import Image

CSV_PATH = "data/train_solution_bounding_boxes.csv"
IMG_DIR = "data/training_images"
OUT_DIR = "data/yolo_dataset"
VAL_FRACTION = 0.15
SEED = 42


def convert():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"image", "xmin", "ymin", "xmax", "ymax"}.issubset(df.columns), \
        f"Unexpected CSV columns: {df.columns.tolist()}"

    image_ids = sorted(df["image"].unique().tolist())
    random.Random(SEED).shuffle(image_ids)
    n_val = int(len(image_ids) * VAL_FRACTION)
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

    for image_id in image_ids:
        split = "val" if image_id in val_ids else "train"
        src_img = os.path.join(IMG_DIR, image_id)
        if not os.path.exists(src_img):
            print(f"warning: missing image {src_img}, skipping")
            continue

        with Image.open(src_img) as im:
            w, h = im.size

        dst_img = os.path.join(OUT_DIR, "images", split, image_id)
        shutil.copyfile(src_img, dst_img)

        rows = df[df["image"] == image_id][["xmin", "ymin", "xmax", "ymax"]].values
        label_lines = []
        for xmin, ymin, xmax, ymax in rows:
            cx = (xmin + xmax) / 2 / w
            cy = (ymin + ymax) / 2 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_name = os.path.splitext(image_id)[0] + ".txt"
        with open(os.path.join(OUT_DIR, "labels", split, label_name), "w") as f:
            f.write("\n".join(label_lines))

    yaml_content = f"""path: {os.path.abspath(OUT_DIR)}
train: images/train
val: images/val
names:
  0: vehicle
"""
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"train images: {len(train_ids)}  val images: {len(val_ids)}")
    print(f"Wrote dataset to {OUT_DIR}/ and config to {OUT_DIR}/data.yaml")


if __name__ == "__main__":
    convert()
