"""Exploratory data analysis: dataset statistics + the sanity checks that decide the design.

Two questions this answers before a single model is trained:

1. How many boxes does the prescribed one-box-per-16x16-cell target encoding silently drop?
   That number is a hard ceiling on recall and belongs in the report.
2. Are the 64.5% of images with no annotation genuinely empty road, or are they unlabeled
   vehicles? Unlabeled vehicles used as negatives would teach the objectness head to
   *suppress* cars - the single most damaging thing that could happen to this project.

Run: `python -m src.eda`  -> prints stats, writes montages to runs/eda/.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

import config
from src import data as data_mod
from src.part1.dataset import encode_target

OUT_DIR = config.RUNS_DIR / "eda"


def montage(images_dir, names, df, out_path, draw_boxes: bool, cols: int = 3, scale: float = 1.0):
    """Write a grid of sample images to `out_path`, optionally with GT boxes drawn in red."""
    tiles = []
    for name in names:
        with Image.open(images_dir / name) as im:
            im = im.convert("RGB")
        if draw_boxes:
            d = ImageDraw.Draw(im)
            for xmin, ymin, xmax, ymax in data_mod.boxes_for_image(df, name):
                d.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=3)
        d = ImageDraw.Draw(im)
        d.text((6, 6), name, fill=(255, 255, 0))
        tiles.append(im)

    w, h = tiles[0].size
    w, h = int(w * scale), int(h * scale)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for k, t in enumerate(tiles):
        sheet.paste(t.resize((w, h)), ((k % cols) * w, (k // cols) * h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def main() -> None:
    paths = data_mod.resolve_dataset_paths()
    df = data_mod.load_annotations(paths)
    images = data_mod.list_images(paths)

    with Image.open(paths.images_dir / images[0]) as im:
        img_w, img_h = im.size

    labeled = set(df.image.unique())
    empty = [im for im in images if im not in labeled]

    # ---- box geometry -----------------------------------------------------------------
    w = (df.xmax - df.xmin).to_numpy()
    h = (df.ymax - df.ymin).to_numpy()
    per_img = Counter(df.image)
    counts = np.array([per_img[i] for i in images])

    print("=" * 70)
    print(f"images            : {len(images)}  ({img_w}x{img_h})")
    print(f"boxes             : {len(df)}")
    print(f"images with boxes : {len(labeled)}  ({100*len(labeled)/len(images):.1f}%)")
    print(f"images with none  : {len(empty)}  ({100*len(empty)/len(images):.1f}%)")
    print(f"boxes per image   : mean {counts.mean():.2f}, max {counts.max()}")
    print(f"  distribution    : {dict(sorted(Counter(counts).items()))}")
    print(f"box width  (px)   : median {np.median(w):.0f}, min {w.min():.0f}, max {w.max():.0f}")
    print(f"box height (px)   : median {np.median(h):.0f}, min {h.min():.0f}, max {h.max():.0f}")
    print(f"box area (% img)  : median {100*np.median(w*h)/(img_w*img_h):.2f}%")

    # ---- the recall ceiling of one-box-per-cell ---------------------------------------
    total_boxes, total_collisions = 0, 0
    for name in labeled:
        boxes = data_mod.boxes_for_image(df, name)
        _, _, collisions = encode_target(boxes, img_w, img_h)
        total_boxes += len(boxes)
        total_collisions += collisions

    print("-" * 70)
    print(f"GRID {config.GRID}x{config.GRID} @ stride {config.STRIDE} (input {config.IMG_SIZE}px)")
    print(f"boxes lost to same-cell collisions: {total_collisions} / {total_boxes} "
          f"({100*total_collisions/max(1,total_boxes):.2f}%)")
    print(f"  -> recall ceiling of this design : {100*(1-total_collisions/max(1,total_boxes)):.2f}%")

    # ---- are the 'empty' images really empty? -----------------------------------------
    rng = np.random.default_rng(config.SEED)
    empty_sample = list(rng.choice(empty, size=min(9, len(empty)), replace=False))
    boxed_sample = list(rng.choice(sorted(labeled), size=min(9, len(labeled)), replace=False))

    p1 = montage(paths.images_dir, empty_sample, df, OUT_DIR / "unlabeled_images.png", False)
    p2 = montage(paths.images_dir, boxed_sample, df, OUT_DIR / "labeled_images.png", True)

    print("-" * 70)
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print("INSPECT unlabeled_images.png: if these contain visible cars, the 'empty' images")
    print("are mislabeled and must NOT be used as all-negative training samples.")


if __name__ == "__main__":
    main()
