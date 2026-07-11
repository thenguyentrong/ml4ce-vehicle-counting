"""Side-by-side ground truth vs prediction figures for the test split (task sheet step 3c).

    uv run python -m src.part1.visualize --tag temporal

Left panel: ground-truth boxes (green). Right panel: predictions (blue = matched a GT box at
IoU >= 0.5, red = false positive), each with its confidence. Writes runs/<tag>/predictions.png.

Looking at these is not decoration - a precision of 0.38 could mean "boxes are in the wrong
place" or "boxes are right but the labels are missing", and only the picture tells you which.

Author: Vinh Nguyen
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw

import config
from src import data as data_mod
from src.part1.dataset import IMAGENET_MEAN, IMAGENET_STD, build_loaders
from src.part1.evaluate import match
from src.part1.infer import decode_predictions
from src.part1.model import VehicleDetector

GREEN, BLUE, RED, YELLOW = (60, 220, 60), (60, 150, 255), (255, 60, 60), (255, 220, 0)


def to_pil(img_t: torch.Tensor) -> Image.Image:
    """Undo the ImageNet normalization and turn a CHW tensor back into a viewable image."""
    arr = img_t.permute(1, 2, 0).cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def draw(im: Image.Image, boxes, color, labels=None, width=3) -> Image.Image:
    im = im.copy()
    d = ImageDraw.Draw(im)
    for k, box in enumerate(boxes):
        d.rectangle([float(v) for v in box], outline=color, width=width)
        if labels is not None:
            d.text((float(box[0]) + 2, max(0, float(box[1]) - 12)), labels[k], fill=color)
    return im


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize predictions against ground truth")
    p.add_argument("--tag", default="temporal")
    p.add_argument("--n", type=int, default=6, help="how many test images to show")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = config.RUNS_DIR / args.tag
    ckpt = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
    saved = ckpt.get("config", {})

    model = VehicleDetector(
        backbone=saved.get("backbone", config.BACKBONE),
        freeze=not saved.get("unfreeze", False),
        pretrained=False,
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Use the same threshold evaluate.py tuned on validation, so the picture matches the table.
    metrics_path = run_dir / "metrics.json"
    thresh = (
        json.loads(metrics_path.read_text())["threshold"]
        if metrics_path.exists()
        else config.SCORE_THRESH
    )

    loaders = build_loaders(augment=False, split_mode=saved.get("split_mode", config.SPLIT_MODE))

    rows = []
    with torch.no_grad():
        for imgs, _, _, gts in loaders["test"]:
            preds = decode_predictions(model(imgs.to(dev)), score_thresh=thresh)
            for img_t, pred, gt in zip(imgs, preds, gts):
                if len(gt) == 0:
                    continue  # show frames that actually contain vehicles
                _, _, _, is_tp = match(pred["boxes"], pred["scores"], gt)

                # Predictions come back confidence-sorted from match(); re-sort to align.
                order = torch.argsort(pred["scores"], descending=True)
                boxes, scores = pred["boxes"][order], pred["scores"][order]

                base = to_pil(img_t)
                left = draw(base, gt, GREEN)
                right = draw(
                    base,
                    boxes,
                    BLUE,  # per-box colour is applied below
                    labels=[f"{s:.2f}" for s in scores.tolist()],
                )
                # Re-draw so false positives stand out in red.
                right = draw(base, [b for b, t in zip(boxes, is_tp) if t], BLUE,
                             [f"{s:.2f}" for s, t in zip(scores.tolist(), is_tp) if t])
                right = draw(right, [b for b, t in zip(boxes, is_tp) if not t], RED,
                             [f"{s:.2f}" for s, t in zip(scores.tolist(), is_tp) if not t])

                rows.append((left, right))
                if len(rows) >= args.n:
                    break
            if len(rows) >= args.n:
                break

    if not rows:
        raise SystemExit("no test images with vehicles found")

    w, h = rows[0][0].size
    pad, header = 8, 26
    sheet = Image.new("RGB", (2 * w + pad, len(rows) * (h + pad) + header), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    d.text((w // 2 - 40, 6), "GROUND TRUTH", fill=GREEN)
    d.text((w + pad + w // 2 - 90, 6), "PREDICTED  (blue = correct, red = false positive)", fill=YELLOW)

    for k, (left, right) in enumerate(rows):
        y = header + k * (h + pad)
        sheet.paste(left, (0, y))
        sheet.paste(right, (w + pad, y))

    out = run_dir / "predictions.png"
    sheet.save(out)
    print(f"wrote {out}  (score threshold {thresh:.2f}, {len(rows)} test images)")


if __name__ == "__main__":
    main()
