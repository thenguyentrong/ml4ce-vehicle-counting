"""Confusion matrices and error diagnostics for the Part 1 detector.

    uv run python -m src.part1.analysis --tag mobilenet_multi_unfreeze

Produces, in runs/<tag>/:
    confusion_matrix.png    detection-level AND cell-level objectness confusion matrices
    recall_by_size.png      recall bucketed by ground-truth box area  <- diagnoses the recall cliff
    analysis.json           the same numbers, for the report

Why two confusion matrices? Object detection has no natural "true negative": the model does not
classify a fixed set of items, it *proposes* boxes. So:

  - **Detection level** is the one people mean by "confusion matrix for a detector" (this is what
    Ultralytics draws): predicted-vehicle vs actual-vehicle, with a `background` row and column
    absorbing false positives and misses. The background/background cell is undefined - there is
    no such thing as "correctly predicted nothing here" when "here" is every possible box.
  - **Cell level** is a genuine 2x2 with a real TN count, and it is arguably the more honest one
    for *this* architecture: our head literally is a binary classifier run over 256 grid cells, so
    "is there a vehicle centered in this cell?" has a well-defined negative class.

Author: Vinh Nguyen
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import config
from src.part1.dataset import build_loaders
from src.part1.evaluate import match
from src.part1.infer import decode_predictions
from src.part1.model import VehicleDetector


def detection_confusion(results: list[dict], score_thresh: float) -> dict:
    """TP / FP / FN at the detection level, at IoU >= 0.5."""
    tp = fp = fn = 0
    for r in results:
        keep = r["scores"] >= score_thresh
        t, f, m, _ = match(r["boxes"][keep], r["scores"][keep], r["gt"])
        tp, fp, fn = tp + t, fp + f, fn + m
    return {"tp": tp, "fp": fp, "fn": fn}


@torch.no_grad()
def cell_confusion(model, loader, dev, score_thresh: float, assign: str) -> dict:
    """True 2x2 confusion over every grid cell: does this cell contain a vehicle center?

    This is the classifier the objectness head actually is, so unlike the detection-level
    matrix it has a well-defined TN - and it exposes the class imbalance in raw numbers.
    """
    model.eval()
    tp = fp = fn = tn = 0

    for imgs, obj_t, _, _ in loader:
        logits = model(imgs.to(dev))[:, 0]  # (B, G, G) objectness
        pred = (torch.sigmoid(logits).cpu() > score_thresh).numpy().astype(bool)
        true = (obj_t.numpy() > 0.5).astype(bool)

        tp += int((pred & true).sum())
        fp += int((pred & ~true).sum())
        fn += int((~pred & true).sum())
        tn += int((~pred & ~true).sum())

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def recall_by_size(results: list[dict], score_thresh: float) -> list[dict]:
    """Recall bucketed by ground-truth box area — the diagnostic for the recall cliff.

    If the misses are concentrated in the smallest bucket, the fix is resolution (a finer
    grid / larger input), not a better loss or more epochs. If they are spread evenly, the
    fix is something else entirely. Guessing here wastes days; measuring takes a minute.
    """
    edges = [0, 1000, 2500, 5000, 10000, np.inf]  # box area in network-input px^2
    names = ["<1k (tiny)", "1k-2.5k", "2.5k-5k", "5k-10k", ">10k (large)"]
    found = [0] * len(names)
    total = [0] * len(names)

    for r in results:
        keep = r["scores"] >= score_thresh
        boxes, scores, gt = r["boxes"][keep], r["scores"][keep], r["gt"]
        if len(gt) == 0:
            continue

        # Which GT boxes were matched? Re-run the greedy matcher and track the GT side.
        from src.part1.infer import box_iou

        matched = np.zeros(len(gt), dtype=bool)
        if len(boxes):
            order = torch.argsort(scores, descending=True)
            ious = box_iou(boxes[order], gt)
            for k in range(len(order)):
                row = ious[k].clone()
                row[torch.from_numpy(matched)] = -1.0
                best = int(row.argmax())
                if float(row[best]) >= config.IOU_MATCH:
                    matched[best] = True

        areas = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).numpy()
        for a, m in zip(areas, matched):
            b = int(np.searchsorted(edges, a, side="right")) - 1
            b = min(b, len(names) - 1)
            total[b] += 1
            found[b] += int(m)

    return [
        {"bucket": n, "n_gt": t, "found": f, "recall": (f / t if t else float("nan"))}
        for n, t, f in zip(names, total, found)
    ]


def plot_confusions(det: dict, cell: dict, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- detection level: background/background is undefined ---------------------------
    m = np.array([[det["tp"], det["fp"]], [det["fn"], np.nan]])
    ax = axes[0]
    ax.imshow(np.nan_to_num(m), cmap="Blues")
    labels = [["TP", "FP"], ["FN", "n/a"]]
    for i in range(2):
        for j in range(2):
            txt = "n/a" if np.isnan(m[i, j]) else f"{labels[i][j]}\n{int(m[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], ["vehicle", "background"])
    ax.set_yticks([0, 1], ["vehicle", "background"])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    p = det["tp"] / max(1, det["tp"] + det["fp"])
    r = det["tp"] / max(1, det["tp"] + det["fn"])
    ax.set_title(f"Detection level (IoU>=0.5)\nprecision {p:.3f}  recall {r:.3f}")

    # --- cell level: a real 2x2, TN included -------------------------------------------
    m2 = np.array([[cell["tp"], cell["fp"]], [cell["fn"], cell["tn"]]])
    ax = axes[1]
    ax.imshow(np.log1p(m2), cmap="Blues")  # log scale: TN dwarfs everything else
    labels2 = [["TP", "FP"], ["FN", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels2[i][j]}\n{int(m2[i, j]):,}", ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], ["vehicle", "no vehicle"])
    ax.set_yticks([0, 1], ["vehicle", "no vehicle"])
    ax.set_xlabel("Actual (grid cell)")
    ax.set_ylabel("Predicted (grid cell)")
    p2 = cell["tp"] / max(1, cell["tp"] + cell["fp"])
    r2 = cell["tp"] / max(1, cell["tp"] + cell["fn"])
    ax.set_title(
        f"Grid-cell objectness (log colour)\nprecision {p2:.3f}  recall {r2:.3f}"
        f"\n{cell['tn']:,} true negatives = the imbalance we fight"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_recall_by_size(buckets: list[dict], out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [b["bucket"] for b in buckets]
    recalls = [0 if np.isnan(b["recall"]) else b["recall"] for b in buckets]
    counts = [b["n_gt"] for b in buckets]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, recalls, color="#4c78a8")
    for bar, b in zip(bars, buckets):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{b['found']}/{b['n_gt']}",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall @ IoU 0.5")
    ax.set_xlabel("Ground-truth box area (network-input px²)")
    ax.set_title("Recall by object size — where the misses actually are")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="Confusion matrices and error diagnostics")
    p.add_argument("--tag", default="mobilenet_multi_unfreeze")
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = config.RUNS_DIR / args.tag
    ckpt = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
    saved = ckpt.get("config", {})
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assign = saved.get("assign", config.ASSIGN)
    thresh, nms_iou = metrics["threshold"], metrics["nms_iou"]

    stride = saved.get("stride", config.STRIDE)
    img_size = saved.get("img_size", config.IMG_SIZE)
    model = VehicleDetector(
        backbone=saved.get("backbone", config.BACKBONE),
        freeze=not saved.get("unfreeze", False),
        pretrained=False,
        stride=stride,
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loaders = build_loaders(
        augment=False,
        split_mode=saved.get("split_mode", config.SPLIT_MODE),
        assign=assign,
        stride=stride,
        img_size=img_size,
    )
    loader = loaders[args.split]

    # Detections, decoded exactly as evaluate.py does.
    results = []
    with torch.no_grad():
        for imgs, _, _, gts in loader:
            preds = decode_predictions(
                model(imgs.to(dev)), score_thresh=0.0, nms_iou=nms_iou, assign=assign, img_size=img_size
            )
            for pred, gt in zip(preds, gts):
                results.append({"boxes": pred["boxes"], "scores": pred["scores"], "gt": gt})

    det = detection_confusion(results, thresh)
    cell = cell_confusion(model, loader, dev, thresh, assign)
    buckets = recall_by_size(results, thresh)

    plot_confusions(det, cell, run_dir / "confusion_matrix.png")
    plot_recall_by_size(buckets, run_dir / "recall_by_size.png")

    print(f"=== {args.tag}  ({args.split} split, score>={thresh:.2f}, nms={nms_iou:.2f}) ===\n")
    print(f"DETECTION LEVEL   TP {det['tp']}   FP {det['fp']}   FN {det['fn']}")
    print(f"GRID-CELL LEVEL   TP {cell['tp']}   FP {cell['fp']}   FN {cell['fn']}   TN {cell['tn']:,}")
    print(f"  -> {cell['tn']:,} true negatives vs {cell['tp']} true positives: the imbalance in raw numbers\n")
    print("RECALL BY BOX SIZE (this is what the PR-curve cliff is made of):")
    for b in buckets:
        r = "  n/a" if np.isnan(b["recall"]) else f"{b['recall']:.3f}"
        print(f"  {b['bucket']:<14} {b['found']:>3}/{b['n_gt']:<3}  recall {r}")

    (run_dir / "analysis.json").write_text(
        json.dumps({"detection": det, "cell": cell, "recall_by_size": buckets}, indent=2, default=float)
    )
    print(f"\nwrote {run_dir/'confusion_matrix.png'}, {run_dir/'recall_by_size.png'}")


if __name__ == "__main__":
    main()
