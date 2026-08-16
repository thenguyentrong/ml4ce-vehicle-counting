"""
Data loader for the Kaggle "Car Object Detection" dataset.

Expects:
  data/training_images/*.jpg
  data/train_solution_bounding_boxes.csv   with columns: image, xmin, ymin, xmax, ymax

Converts every image's boxes into a (GRID, GRID, 5) target tensor:
  channel 0        = objectness (1 if a box center falls in this cell, else 0)
  channels 1..2    = box center offset within the cell, in [0, 1]
  channels 3..4    = box width, height, normalized by the *image* size (0..1)

Adjust CSV_PATH / IMG_DIR / column names if your download differs.
"""
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_SIZE = 512       # resize every image to this (square) size
STRIDE = 32          # backbone stride -> grid = IMG_SIZE / STRIDE
GRID = IMG_SIZE // STRIDE   # 16x16 for the defaults above

# default data locations -- adjust if you unzip the Kaggle dataset elsewhere
CSV_PATH = "data/train_solution_bounding_boxes.csv"
IMG_DIR = "data/training_images"


class VehicleDataset(Dataset):
    def __init__(self, csv_path, img_dir, image_ids, augment=False):
        """
        csv_path: path to the bounding-box CSV
        img_dir: folder containing the images
        image_ids: list of filenames (image column values) to include in this split
        """
        df = pd.read_csv(csv_path)
        # normalize column names just in case (strip spaces, lowercase check)
        df.columns = [c.strip().lower() for c in df.columns]
        assert {"image", "xmin", "ymin", "xmax", "ymax"}.issubset(df.columns), \
            f"Unexpected CSV columns: {df.columns.tolist()} -- adjust dataset.py"

        self.df = df[df["image"].isin(image_ids)].reset_index(drop=True)
        self.img_dir = img_dir
        self.image_ids = image_ids
        self.augment = augment

        self.to_tensor = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        boxes = self.df[self.df["image"] == image_id][["xmin", "ymin", "xmax", "ymax"]].values

        img_tensor = self.to_tensor(img)  # (3, IMG_SIZE, IMG_SIZE)

        target = torch.zeros((GRID, GRID, 5), dtype=torch.float32)
        cell_size = IMG_SIZE / GRID

        for xmin, ymin, xmax, ymax in boxes:
            # rescale box coords from original image size to IMG_SIZE
            xmin = xmin / orig_w * IMG_SIZE
            xmax = xmax / orig_w * IMG_SIZE
            ymin = ymin / orig_h * IMG_SIZE
            ymax = ymax / orig_h * IMG_SIZE

            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            col = int(cx // cell_size)
            row = int(cy // cell_size)
            col = min(max(col, 0), GRID - 1)
            row = min(max(row, 0), GRID - 1)

            # offset of center within its cell, in [0, 1]
            off_x = (cx - col * cell_size) / cell_size
            off_y = (cy - row * cell_size) / cell_size

            target[row, col, 0] = 1.0
            target[row, col, 1] = off_x
            target[row, col, 2] = off_y
            target[row, col, 3] = w / IMG_SIZE
            target[row, col, 4] = h / IMG_SIZE

        return img_tensor, target


def make_splits(csv_path, seed=42, train_frac=0.8, val_frac=0.15):
    """Returns (train_ids, val_ids, test_ids) -- unique image filenames, shuffled."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    image_ids = sorted(df["image"].unique().tolist())

    random.Random(seed).shuffle(image_ids)
    n = len(image_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    # quick sanity check -- uses the default paths above; adjust CSV_PATH/IMG_DIR
    # at the top of this file if you unzipped the data somewhere else
    train_ids, val_ids, test_ids = make_splits(CSV_PATH)
    print(f"train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")

    ds = VehicleDataset(CSV_PATH, IMG_DIR, train_ids)
    img, target = ds[0]
    print("image tensor shape:", img.shape)
    print("target shape:", target.shape)
    print("num positive cells:", int(target[..., 0].sum().item()))
