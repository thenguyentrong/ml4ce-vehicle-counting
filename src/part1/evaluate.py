"""Evaluation: precision and recall at IoU >= 0.5, plus a PR curve and AP50.

    uv run python -m src.part1.evaluate --tag baseline

Matching rule (task sheet step 3c): a prediction is correct if it overlaps a ground-truth box
with IoU >= 0.5. Detections are matched greedily in order of confidence, and **each ground-truth
box can only be claimed once** - without that rule, ten copies of one correct box would score as
ten true positives and precision would be meaningless.

The score threshold is chosen on the **validation** split and only then applied to **test**.
Picking it on test would be tuning on the test set, which inflates the reported numbers.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import config
from src.part1.dataset import build_loaders
from src.part1.infer import box_iou, decode_predictions
from src.part1.model import VehicleDetector


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    dev,
    nms_iou: float = config.NMS_IOU,
    assign: str = config.ASSIGN,
    img_size: int | None = None,
) -> list[dict]:
    """Run the model over a split; keep raw scores so any score threshold can be applied later.

    `nms_iou` must be fixed here (NMS happens before scores are thresholded downstream), so
    sweeping it means re-running this - cheap enough on 150 images.

    `assign` and `img_size` must be what the model was TRAINED with. `assign` selects the
    offset activation and `img_size` sets the pixel scale of the decoded boxes; get either
    wrong and the boxes come out subtly misplaced, with no error raised.
    """
    model.eval()
    results = []
    for imgs, _, _, gt_boxes in loader:
        preds = decode_predictions(
            model(imgs.to(dev)), score_thresh=0.0, nms_iou=nms_iou, assign=assign, img_size=img_size
        )
        for pred, gt in zip(preds, gt_boxes):
            results.append({"boxes": pred["boxes"], "scores": pred["scores"], "gt": gt})
    return results


def match(pred_boxes, pred_scores, gt_boxes, iou_thresh=config.IOU_MATCH):
    """Greedy confidence-ordered matching -> (n_tp, n_fp, n_fn, matched_flags).

    Each GT box may be claimed by at most one prediction; further overlapping predictions
    are false positives (this is what NMS is supposed to have removed).
    """
    n_gt = len(gt_boxes)
    if len(pred_boxes) == 0:
        return 0, 0, n_gt, np.zeros(0, dtype=bool)
    if n_gt == 0:
        return 0, len(pred_boxes), 0, np.zeros(len(pred_boxes), dtype=bool)

    order = torch.argsort(pred_scores, descending=True)
    ious = box_iou(pred_boxes[order], gt_boxes)

    gt_taken = np.zeros(n_gt, dtype=bool)
    is_tp = np.zeros(len(pred_boxes), dtype=bool)

    for k in range(len(order)):
        row = ious[k].clone()
        row[torch.from_numpy(gt_taken)] = -1.0  # already-claimed GT boxes are off the table
        best = int(row.argmax())
        if float(row[best]) >= iou_thresh:
            gt_taken[best] = True
            is_tp[k] = True

    n_tp = int(is_tp.sum())
    return n_tp, len(pred_boxes) - n_tp, n_gt - n_tp, is_tp


def prf_at_threshold(results: list[dict], thresh: float) -> dict[str, float]:
    """Aggregate precision / recall / F1 over a whole split at one score threshold."""
    tp = fp = fn = 0
    for r in results:
        keep = r["scores"] >= thresh
        t, f, m, _ = match(r["boxes"][keep], r["scores"][keep], r["gt"])
        tp, fp, fn = tp + t, fp + f, fn + m

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def pr_curve(results: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    """Precision/recall at every confidence level, plus AP50 (area under the PR curve).

    AP is computed with the standard all-point interpolation: precision is made monotonically
    decreasing before integrating, so a jitter in the ranking cannot inflate the area.
    """
    scored = []
    n_gt_total = 0
    for r in results:
        n_gt_total += len(r["gt"])
        _, _, _, is_tp = match(r["boxes"], r["scores"], r["gt"])
        order = torch.argsort(r["scores"], descending=True)
        for s, t in zip(r["scores"][order].tolist(), is_tp):
            scored.append((s, bool(t)))

    if not scored or n_gt_total == 0:
        return np.array([0.0]), np.array([0.0]), 0.0

    scored.sort(key=lambda x: -x[0])
    tps = np.cumsum([t for _, t in scored])
    fps = np.cumsum([not t for _, t in scored])

    recalls = tps / n_gt_total
    precisions = tps / np.maximum(tps + fps, 1e-9)

    # Monotonic envelope, then integrate.
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return precisions, recalls, ap


def plot_pr(precisions, recalls, ap, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recalls, precisions, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall @ IoU 0.5  (AP50 = {ap:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the Part 1 detector")
    p.add_argument("--tag", default="baseline")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = config.RUNS_DIR / args.tag
    ckpt = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)

    saved = ckpt.get("config", {})
    stride = saved.get("stride", config.STRIDE)
    img_size = saved.get("img_size", config.IMG_SIZE)
    model = VehicleDetector(
        backbone=saved.get("backbone", config.BACKBONE),
        freeze=not saved.get("unfreeze", False),
        pretrained=False,
        stride=stride,
    ).to(dev)
    model.load_state_dict(ckpt["model"])

    # Evaluate on the same split the model was trained under, otherwise a temporally-trained
    # model would be scored on frames a randomly-split run had used for training. Likewise the
    # assignment scheme selects the offset activation and must match training.
    split_mode = saved.get("split_mode", config.SPLIT_MODE)
    assign = saved.get("assign", config.ASSIGN)
    print(f"split mode: {split_mode}   assign: {assign}")
    loaders = build_loaders(
        augment=False, split_mode=split_mode, assign=assign, stride=stride, img_size=img_size
    )

    # --- tune BOTH the score threshold and the NMS IoU on VALIDATION --------------------
    # NMS matters more than it looks: one vehicle often excites two neighbouring cells, and
    # the two boxes it produces can overlap too little for a lax NMS to merge them. The
    # duplicate is then a false positive *and* the fragment misses the GT box at IoU 0.5,
    # so a single car costs a FP and a FN at once. Sweeping the NMS threshold fixes that.
    score_grid = np.arange(0.05, 0.96, 0.05)
    nms_grid = [0.1, 0.2, 0.3, 0.4, 0.5]

    best = None
    print("tuning on VAL (score threshold x NMS IoU):")
    for nms_iou in nms_grid:
        val_res = collect_predictions(
            model, loaders["val"], dev, nms_iou=nms_iou, assign=assign, img_size=img_size
        )
        t, m = max(((t, prf_at_threshold(val_res, t)) for t in score_grid), key=lambda x: x[1]["f1"])
        print(f"  nms={nms_iou:.1f}  score={t:.2f}  P {m['precision']:.3f}  "
              f"R {m['recall']:.3f}  F1 {m['f1']:.3f}")
        if best is None or m["f1"] > best[2]["f1"]:
            best = (t, nms_iou, m)

    best_t, best_nms, best_val = best
    print(f"\nbest on VAL: score={best_t:.2f}  nms={best_nms:.1f}  F1 {best_val['f1']:.3f}")

    # --- report on TEST at those settings -----------------------------------------------
    test_res = collect_predictions(
        model, loaders["test"], dev, nms_iou=best_nms, assign=assign, img_size=img_size
    )
    test = prf_at_threshold(test_res, best_t)
    precisions, recalls, ap = pr_curve(test_res)

    print("\n" + "=" * 62)
    print(f"TEST SET  ({len(test_res)} images, IoU >= {config.IOU_MATCH})")
    print("=" * 62)
    print(f"  precision : {test['precision']:.3f}")
    print(f"  recall    : {test['recall']:.3f}")
    print(f"  F1        : {test['f1']:.3f}")
    print(f"  AP50      : {ap:.3f}")
    print(f"  TP {test['tp']}   FP {test['fp']}   FN {test['fn']}")

    plot_pr(precisions, recalls, ap, run_dir / "pr_curve.png")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "threshold": best_t,
                "nms_iou": best_nms,
                "split_mode": split_mode,
                "val": best_val,
                "test": test,
                "ap50": ap,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nwrote {run_dir/'pr_curve.png'} and {run_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
