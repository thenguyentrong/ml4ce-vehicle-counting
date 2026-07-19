"""Fine-tune YOLO11-nano on the Part 1 dataset.

Run `python -m src.part2.finetune` to train, `--eval-only <weights>` to score existing weights.

This is the point of Part 2 where we deliberately stop building the detector ourselves: the task
sheet asks us to take a small pretrained model from an established framework, adapt it with a few
epochs, and spend our own effort on the tracking instead. `yolo11n` is ~2.6M parameters.

**What "fine-tune" means here.** The stock weights are trained on COCO's 80 classes; we retrain
the head for our single `vehicle` class on 801 images. The backbone is *not* frozen - unlike
Part 1, where freezing was prescribed to keep the exercise about the head, here the whole network
adapts, which is what ultralytics does by default and what "fine-tuning" normally means.

**An honest caveat we intend to measure rather than assume.** The Part 1 dataset is *dashcam*
footage: rear views of vehicles from road level. The Part 2 video is a *static street camera*,
which sees vehicles head-on and from a different height. Fine-tuning on 355 vehicle-bearing
dashcam frames may well specialise the model *away* from the footage we then run it on, in which
case the stock COCO weights would be the better detector. `run_count.py --weights stock` runs the
un-fine-tuned model so the two can be compared on the video itself - the task sheet asks us to
compare methods and explain which works best and why, and this is one of the comparisons that
actually has a surprising answer.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

import config
from src.part2 import yolo_data


def train(
    tag: str = "finetune",
    epochs: int = config.YOLO_EPOCHS,
    imgsz: int = config.YOLO_IMGSZ,
    batch: int = config.YOLO_BATCH,
    model_name: str = config.YOLO_MODEL,
) -> Path:
    """Fine-tune and return the path of the best checkpoint."""
    data_yaml = yolo_data.build()

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(config.RUNS_DIR / "yolo"),
        name=tag,
        exist_ok=True,
        seed=config.SEED,
        verbose=True,
    )

    best = config.RUNS_DIR / "yolo" / tag / "weights" / "best.pt"
    print(f"[finetune] best weights: {best}")
    return best


def evaluate(weights: str | Path, split: str = "test") -> dict[str, float]:
    """Score a checkpoint on one split and print the headline metrics.

    Reported at IoU >= 0.5 (`map50`) so the numbers sit on the same scale as Part 1's AP50.
    """
    data_yaml = yolo_data.build()
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=split, verbose=False)

    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    f1 = (
        2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
        if result["precision"] + result["recall"] > 0
        else 0.0
    )
    result["f1"] = f1

    print(
        f"[finetune] {split}: P {result['precision']:.3f}  R {result['recall']:.3f}  "
        f"F1 {f1:.3f}  AP50 {result['map50']:.3f}  AP50-95 {result['map50_95']:.3f}"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune YOLO11-nano on the Part 1 dataset")
    p.add_argument("--tag", default="finetune", help="run name -> runs/yolo/<tag>/")
    p.add_argument("--epochs", type=int, default=config.YOLO_EPOCHS)
    p.add_argument("--imgsz", type=int, default=config.YOLO_IMGSZ)
    p.add_argument("--batch", type=int, default=config.YOLO_BATCH)
    p.add_argument("--model", default=config.YOLO_MODEL)
    p.add_argument("--eval-only", metavar="WEIGHTS", help="skip training, just score these weights")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = p.parse_args()

    weights = args.eval_only or train(
        tag=args.tag,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
    )
    evaluate(weights, split=args.split)


if __name__ == "__main__":
    main()
