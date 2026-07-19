"""Convert the Kaggle dataset into the folder/label layout ultralytics expects.

Run `python -m src.part2.yolo_data` to build it and print a summary.

The splits come from `src.data.make_splits`, the *same* function Part 1 uses, so the fine-tuned
YOLO is trained and scored on exactly the images our hand-built detector was. Re-splitting here
would make any Part 1 vs Part 2 comparison meaningless - and, because the dataset is one video
sampled every 20 frames, a fresh random split would also leak near-duplicate frames between
train and test (see `src.data.make_splits`).

Layout written, which is what ultralytics discovers by convention - it takes an `images` path and
finds `labels` by string-replacing that one path component:

    data/yolo/
        data.yaml
        train/images/*.jpg   train/labels/*.txt
        val/images/*.jpg     val/labels/*.txt
        test/images/*.jpg    test/labels/*.txt

Label format is one line per box, `class cx cy w h`, with the four geometry values normalised to
[0, 1] by image size. Class is always 0: the task sheet asks for a single `vehicle` class.

Images with no vehicle get **no label file at all** - that is how ultralytics represents a
background image, and it is not the same as an empty file. They are kept deliberately: 64.5% of
this dataset is genuinely empty road, and those frames are what teach the model not to fire on
asphalt.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from PIL import Image

import config
from src import data


def _write_label(path: Path, boxes, width: int, height: int) -> None:
    """Write one YOLO label file: `0 cx cy w h`, normalised, one line per box."""
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        # Boxes that touch the border can round marginally outside [0, 1]; ultralytics warns
        # and drops those, so clamp rather than lose a legitimate vehicle.
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(bw, 1.0), min(bh, 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(out_dir: Path | None = None, force: bool = False) -> Path:
    """Materialise the YOLO-format dataset and return the path of its data.yaml."""
    out_dir = out_dir or config.YOLO_DATA_DIR
    yaml_path = out_dir / "data.yaml"

    if yaml_path.exists() and not force:
        print(f"[yolo_data] already built: {yaml_path}  (use --force to rebuild)")
        return yaml_path

    if out_dir.exists():
        shutil.rmtree(out_dir)

    paths = data.resolve_dataset_paths()
    df = data.load_annotations(paths)
    splits = data.make_splits(data.list_images(paths))

    counts: dict[str, dict[str, int]] = {}
    for split, images in splits.items():
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n_boxes, n_empty = 0, 0
        for name in images:
            src = paths.images_dir / name
            shutil.copy2(src, img_dir / name)

            boxes = data.boxes_for_image(df, name)
            if len(boxes) == 0:
                n_empty += 1  # no label file: this is a background image
                continue

            with Image.open(src) as im:
                width, height = im.size
            _write_label(lbl_dir / f"{Path(name).stem}.txt", boxes, width, height)
            n_boxes += len(boxes)

        counts[split] = {"images": len(images), "boxes": n_boxes, "empty": n_empty}

    # `path` is absolute so the file works regardless of the directory ultralytics is invoked
    # from; the split entries are relative to it.
    spec = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "vehicle"},
    }
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    for split, c in counts.items():
        print(
            f"[yolo_data] {split:5s} images={c['images']:4d} boxes={c['boxes']:4d} "
            f"empty={c['empty']:4d}"
        )
    print(f"[yolo_data] wrote {yaml_path}")
    return yaml_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Convert the Kaggle dataset to YOLO format")
    p.add_argument("--force", action="store_true", help="rebuild even if it already exists")
    args = p.parse_args()
    build(force=args.force)


if __name__ == "__main__":
    main()
