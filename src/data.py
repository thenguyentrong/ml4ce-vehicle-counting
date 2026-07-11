"""Dataset acquisition, annotation parsing and the train/val/test split.

Part 1 and Part 2 both consume this module, which guarantees they see *exactly* the same
splits — otherwise a comparison between our hand-built detector and the fine-tuned YOLO
would be meaningless.

Run `python -m src.data` to download the dataset and print a summary of what arrived.

Author: Vinh Nguyen
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config


@dataclass(frozen=True)
class DatasetPaths:
    """Where the pieces of the Kaggle archive ended up after download."""

    root: Path
    images_dir: Path
    csv_path: Path


def download() -> Path:
    """Download the Kaggle 'Car Object Detection' dataset and return its root folder.

    `config` sets KAGGLEHUB_CACHE to `data/kagglehub`, so the archive lands inside the
    project rather than in the user profile. kagglehub is a no-op if already cached.
    """
    import kagglehub  # imported lazily: only needed on first run

    path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    return Path(path)


def resolve_dataset_paths(root: Path | None = None) -> DatasetPaths:
    """Locate the training images and the annotation CSV inside the downloaded archive.

    The archive's internal layout is discovered rather than assumed, so a change on
    Kaggle's side surfaces as a clear error here instead of an empty dataloader later.
    """
    root = root or download()

    # The annotation CSV is the one with bounding box columns; a sample_submission.csv also
    # ships with the dataset and must not be picked up by mistake.
    csv_candidates = [
        p
        for p in root.rglob("*.csv")
        if {"xmin", "ymin", "xmax", "ymax"}.issubset(
            {c.strip().lower() for c in pd.read_csv(p, nrows=0).columns}
        )
    ]
    if not csv_candidates:
        raise FileNotFoundError(
            f"No annotation CSV with xmin/ymin/xmax/ymax columns found under {root}. "
            f"CSVs present: {[p.name for p in root.rglob('*.csv')]}"
        )
    csv_path = csv_candidates[0]

    # The image folder is the one holding the files the CSV refers to. `testing_images/`
    # exists too but is unlabeled, so it is useless for training or scoring.
    first_image = str(pd.read_csv(csv_path)["image"].iloc[0])
    image_dirs = {p.parent for p in root.rglob(first_image)}
    if not image_dirs:
        raise FileNotFoundError(
            f"CSV {csv_path.name} references '{first_image}', which was not found under {root}."
        )
    images_dir = sorted(image_dirs)[0]

    return DatasetPaths(root=root, images_dir=images_dir, csv_path=csv_path)


def load_annotations(paths: DatasetPaths | None = None) -> pd.DataFrame:
    """Return one row per bounding box: image, xmin, ymin, xmax, ymax (pixels, original size).

    Degenerate boxes (zero or negative width/height) are dropped — they cannot be encoded
    into a grid target and would poison the IoU losses with NaNs.
    """
    paths = paths or resolve_dataset_paths()
    df = pd.read_csv(paths.csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    n_before = len(df)
    df = df[(df.xmax > df.xmin) & (df.ymax > df.ymin)].reset_index(drop=True)
    if (dropped := n_before - len(df)) > 0:
        print(f"[data] dropped {dropped} degenerate box(es) with non-positive width/height")

    return df


def list_images(paths: DatasetPaths | None = None) -> list[str]:
    """Every image file in the labeled image folder, including those with no vehicle.

    Images without any annotation row are *kept*: they are valid all-negative samples and
    teach the objectness head what empty road looks like.
    """
    paths = paths or resolve_dataset_paths()
    return sorted(p.name for p in paths.images_dir.glob("*.jpg"))


def frame_number(image: str) -> int:
    """Frame index from a filename like 'vid_4_10020.jpg' -> 10020.

    Every image in this dataset comes from ONE video (vid_4), sampled every 20 frames, so
    the filename number is a timestamp. That fact drives the split below.
    """
    m = re.search(r"_(\d+)\.jpg$", image)
    if not m:
        raise ValueError(f"cannot read a frame number from {image!r}")
    return int(m.group(1))


def make_splits(
    images: list[str] | None = None,
    seed: int = config.SEED,
    mode: str = config.SPLIT_MODE,
) -> dict[str, list[str]]:
    """Split image *filenames* 80/15/5 into train/val/test.

    Two modes, and the choice matters more than any hyperparameter in this project:

    **"temporal" (default, honest).** All 1001 images are frames of a *single* video, sampled
    every 20 frames — roughly 0.67 s apart at 30 fps. Adjacent frames are near-duplicates:
    the same cars, the same lighting, shifted slightly. A *random* split therefore puts a
    frame in train and its almost-identical neighbour in test, and the model gets credit for
    recognising a car it has already memorised. Splitting on the time axis instead — train on
    the earliest 80% of the video, validate on the next 15%, test on the last 5% — means the
    test frames show road the model has genuinely never seen.

    **"random" (leaky, kept for comparison).** The naive split. We report it alongside the
    temporal one precisely to show how large the leakage-induced optimism is.

    In both cases the split is by image, never by box: boxes from one image are correlated.
    """
    images = images or list_images()

    if mode == "temporal":
        ordered = sorted(images, key=frame_number)  # chronological
    elif mode == "random":
        rng = np.random.default_rng(seed)
        ordered = list(images)
        rng.shuffle(ordered)
    else:
        raise ValueError(f"unknown split mode {mode!r} (expected 'temporal' or 'random')")

    n = len(ordered)
    n_train = int(round(config.SPLIT_TRAIN * n))
    n_val = int(round(config.SPLIT_VAL * n))

    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def boxes_for_image(df: pd.DataFrame, image: str) -> np.ndarray:
    """All boxes of one image as an (N, 4) float array [xmin, ymin, xmax, ymax]; (0, 4) if none."""
    rows = df[df.image == image]
    if rows.empty:
        return np.zeros((0, 4), dtype=np.float32)
    return rows[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)


def main() -> None:
    """Download the dataset and report what actually arrived."""
    from PIL import Image

    paths = resolve_dataset_paths()
    df = load_annotations(paths)
    images = list_images(paths)
    splits = make_splits(images)

    with Image.open(paths.images_dir / images[0]) as im:
        size = im.size

    labeled = set(df.image.unique())
    empty = [im for im in images if im not in labeled]

    print(f"root         : {paths.root}")
    print(f"images_dir   : {paths.images_dir}")
    print(f"csv          : {paths.csv_path.name}")
    print(f"columns      : {list(df.columns)}")
    print(f"images       : {len(images)}  (native size {size[0]}x{size[1]})")
    print(f"boxes        : {len(df)}")
    print(f"empty images : {len(empty)}  ({100 * len(empty) / len(images):.1f}%)")
    print(f"boxes/image  : {len(df) / max(1, len(images)):.2f} mean")
    print(f"split        : train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
