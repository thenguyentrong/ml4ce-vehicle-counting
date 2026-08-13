"""Part 1 inference on any folder of images — the entry point for the held-out test set.

    python -m src.part1.predict --images path/to/images
    python -m src.part1.predict --images path/to/images --csv ground_truth.csv

`evaluate.py` can only score our own Kaggle split: it goes through `build_loaders()`, which needs
the Kaggle CSV and our train/val/test split. The course tests the model on a separate set after
submission, so this takes the other route — a directory of images, no annotations needed.

With no `--tag` it takes the run with the best recorded test F1 and uses the threshold and NMS IoU
that run tuned on validation. Preprocessing is the same as `VehicleGridDataset`, and boxes come
back in **original-image pixels**, which is the frame any external ground truth is in.

Writes to `runs/predict/<name>/`: `predictions.csv`, `predictions.json`, annotated JPEGs, and with
`--csv` also `metrics.json` (precision / recall / F1 / AP50 at IoU 0.5).

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from src.part1.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.part1.evaluate import pr_curve, prf_at_threshold
from src.part1.infer import decode_predictions
from src.part1.model import VehicleDetector
from src.part1.visualize import GREEN, RED, draw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_run(tag: str | None) -> tuple[Path, dict]:
    """Return (run_dir, metrics) for `tag`, or for the best-scoring run if `tag` is None.

    "Best" comes from each run's `metrics.json`, so the ranking is measured test F1 and not a
    hard-coded name that goes stale.
    """
    if tag is not None:
        run_dir = config.RUNS_DIR / tag
        if not (run_dir / "best.pt").exists():
            raise SystemExit(f"no checkpoint at {run_dir / 'best.pt'}")
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        return run_dir, metrics

    scored = []
    for ckpt in sorted(config.RUNS_DIR.glob("*/best.pt")):
        metrics_path = ckpt.parent / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        scored.append((metrics.get("test", {}).get("f1", 0.0), ckpt.parent, metrics))

    if not scored:
        raise SystemExit(
            f"no evaluated checkpoint found under {config.RUNS_DIR}. Train one with "
            f"`python -m src.part1.train`, or pass --tag explicitly."
        )

    f1, run_dir, metrics = max(scored, key=lambda s: s[0])
    print(f"[predict] no --tag given; using {run_dir.name} (test F1 {f1:.3f})")
    return run_dir, metrics


def load_model(run_dir: Path, dev: torch.device) -> tuple[VehicleDetector, dict]:
    """Rebuild the architecture the checkpoint was trained with and load its weights.

    Backbone, stride, input size and assignment come from the checkpoint, never from config: a
    `--assign multi` model decoded with the "center" activation gives displaced boxes, no error.
    """
    ckpt = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
    saved = ckpt.get("config", {})

    model = VehicleDetector(
        backbone=saved.get("backbone", config.BACKBONE),
        freeze=not saved.get("unfreeze", False),
        pretrained=False,
        stride=saved.get("stride", config.STRIDE),
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, saved


def list_image_files(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise SystemExit(f"--images {images_dir} is not a directory")
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"no images ({', '.join(sorted(IMAGE_SUFFIXES))}) in {images_dir}")
    return files


def preprocess(path: Path, img_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Image file -> (CHW tensor, original size). Identical to what the dataset does in training."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        original_size = im.size  # (w, h)
        arr = np.asarray(im.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32) / 255.0

    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), original_size


def to_original_pixels(boxes: torch.Tensor, size: tuple[int, int], img_size: int) -> torch.Tensor:
    """Rescale boxes from network-input pixels back to the original image's pixels."""
    if boxes.numel() == 0:
        return boxes
    w, h = size
    scale = torch.tensor([w / img_size, h / img_size] * 2, dtype=boxes.dtype)
    return boxes * scale


def ground_truth_boxes(csv_path: Path) -> dict[str, np.ndarray]:
    """Parse an annotation CSV into {image filename: (N, 4) boxes in original pixels}.

    Kaggle layout (`image,xmin,ymin,xmax,ymax`, any column order or case) - the format this
    dataset ships in, and the likeliest shape of a held-out test CSV.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"image", "xmin", "ymin", "xmax", "ymax"}
    if not required.issubset(df.columns):
        raise SystemExit(
            f"{csv_path.name} must have columns {sorted(required)}; found {list(df.columns)}"
        )

    df = df[(df.xmax > df.xmin) & (df.ymax > df.ymin)]
    return {
        str(name): group[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
        for name, group in df.groupby("image")
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Part 1 detector on a folder of images")
    p.add_argument("--images", type=Path, required=True, help="folder of images to run on")
    p.add_argument("--tag", default=None, help="run under runs/ to load; default: best test F1")
    p.add_argument("--csv", type=Path, default=None,
                   help="optional ground truth (image,xmin,ymin,xmax,ymax) -> scores the run")
    p.add_argument("--out", type=Path, default=None, help="output dir; default runs/predict/<name>")
    p.add_argument("--score", type=float, default=None, help="override the score threshold")
    p.add_argument("--nms", type=float, default=None, help="override the NMS IoU")
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--annotate", type=int, default=12, help="how many annotated images to write")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, metrics = resolve_run(args.tag)
    model, saved = load_model(run_dir, dev)

    img_size = saved.get("img_size", config.IMG_SIZE)
    assign = saved.get("assign", config.ASSIGN)
    # Validation-tuned settings from evaluate.py; --score/--nms override them deliberately.
    score_thresh = args.score if args.score is not None else metrics.get("threshold", config.SCORE_THRESH)
    nms_iou = args.nms if args.nms is not None else metrics.get("nms_iou", config.NMS_IOU)

    files = list_image_files(args.images)
    out_dir = args.out or (config.RUNS_DIR / "predict" / args.images.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[predict] {len(files)} images from {args.images}")
    print(f"[predict] weights {run_dir/'best.pt'}  (assign={assign}, input={img_size})")
    print(f"[predict] score >= {score_thresh:.2f}, NMS IoU {nms_iou:.2f}, device {dev.type}")

    truth = ground_truth_boxes(args.csv) if args.csv else None
    rows: list[dict] = []
    scored: list[dict] = []  # for metrics, only when ground truth is supplied
    n_annotated = 0

    with torch.no_grad():
        for start in range(0, len(files), args.batch):
            chunk = files[start : start + args.batch]
            tensors, sizes = zip(*(preprocess(f, img_size) for f in chunk))
            # Threshold 0 here, thresholded below - same as evaluate.py. AP50 integrates the
            # whole PR curve, so it needs the low-confidence tail; cutting it first understates
            # AP (0.846 instead of 0.871 on the project test split).
            preds = decode_predictions(
                model(torch.stack(tensors).to(dev)),
                score_thresh=0.0,
                nms_iou=nms_iou,
                assign=assign,
                img_size=img_size,
            )

            for path, pred, size in zip(chunk, preds, sizes):
                all_boxes = to_original_pixels(pred["boxes"], size, img_size)
                all_scores = pred["scores"]

                keep = all_scores >= score_thresh
                boxes, scores = all_boxes[keep], all_scores[keep]
                rows.extend(
                    {
                        "image": path.name,
                        "xmin": round(float(b[0]), 2), "ymin": round(float(b[1]), 2),
                        "xmax": round(float(b[2]), 2), "ymax": round(float(b[3]), 2),
                        "score": round(float(s), 4),
                    }
                    for b, s in zip(boxes, scores)
                )

                gt = None
                if truth is not None:
                    gt = torch.from_numpy(truth.get(path.name, np.zeros((0, 4), dtype=np.float32)))
                    scored.append({"boxes": all_boxes, "scores": all_scores, "gt": gt})

                if n_annotated < args.annotate:
                    annotate(path, boxes, scores, gt, out_dir / "annotated")
                    n_annotated += 1

            print(f"[predict] {min(start + args.batch, len(files))}/{len(files)}")

    frame = pd.DataFrame(rows, columns=["image", "xmin", "ymin", "xmax", "ymax", "score"])
    frame.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "predictions.json").write_text(
        json.dumps(
            {
                "run": run_dir.name,
                "images_dir": str(args.images),
                "n_images": len(files),
                "n_boxes": len(rows),
                "score_thresh": float(score_thresh),
                "nms_iou": float(nms_iou),
                "assign": assign,
                "img_size": img_size,
                "coordinates": "original image pixels, [xmin, ymin, xmax, ymax]",
                "predictions": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[predict] {len(rows)} boxes over {len(files)} images "
          f"({len(rows) / len(files):.2f} per image)")
    print(f"[predict] wrote {out_dir/'predictions.csv'} and {out_dir/'predictions.json'}")
    if n_annotated:
        print(f"[predict] wrote {n_annotated} annotated images to {out_dir/'annotated'}")

    if scored:
        report(scored, out_dir, score_thresh, nms_iou, run_dir)


def annotate(path: Path, boxes, scores, gt, out_dir: Path) -> None:
    """Write a copy of the image with predictions drawn (and ground truth, when available).

    Green is ground truth, red is a prediction — deliberately the same colour language as
    `visualize.py`, so the two sets of figures can be read side by side.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as im:
        im = im.convert("RGB")

    if gt is not None and len(gt):
        im = draw(im, gt, GREEN, width=2)
    im = draw(im, boxes, RED, [f"{float(s):.2f}" for s in scores], width=2)
    im.save(out_dir / f"{path.stem}.jpg", quality=90)


def report(scored: list[dict], out_dir: Path, score_thresh: float, nms_iou: float,
           run_dir: Path) -> None:
    """Precision / recall / F1 / AP50 against supplied ground truth, at IoU >= 0.5.

    The threshold is not re-tuned here - it comes from the validation split. Re-picking it
    against the new labels would be tuning on the test set.
    """
    n_gt = sum(len(r["gt"]) for r in scored)
    if n_gt == 0:
        print("[predict] ground truth CSV matched no image in this folder - no metrics computed")
        return

    prf = prf_at_threshold(scored, score_thresh)
    _, _, ap = pr_curve(scored)

    print("\n" + "=" * 62)
    print(f"SCORED against ground truth  ({len(scored)} images, {n_gt} boxes, IoU >= {config.IOU_MATCH})")
    print("=" * 62)
    print(f"  precision : {prf['precision']:.3f}")
    print(f"  recall    : {prf['recall']:.3f}")
    print(f"  F1        : {prf['f1']:.3f}")
    print(f"  AP50      : {ap:.3f}")
    print(f"  TP {prf['tp']}   FP {prf['fp']}   FN {prf['fn']}")

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run": run_dir.name,
                "threshold": float(score_thresh),
                "nms_iou": float(nms_iou),
                "n_images": len(scored),
                "n_gt_boxes": n_gt,
                "metrics": prf,
                "ap50": ap,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"\n[predict] wrote {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
