"""Training loop for the Part 1 detection head.

    uv run python -m src.part1.train                  # train with the config.py defaults
    uv run python -m src.part1.train --overfit        # sanity check: overfit ONE batch
    uv run python -m src.part1.train --tag ciou --box-loss ciou --epochs 40   # an ablation

`--overfit` is the check that must pass before any real training is trusted: a correct
pipeline can drive the loss on a single batch to ~0. If it cannot, the bug is in the
encoding, the decoding or the loss - and no amount of epochs on the full dataset will fix it.

Every run writes runs/<tag>/: best.pt, history.json, curves.png.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from src.part1.dataset import build_loaders, collate
from src.part1.losses import DetectionLoss
from src.part1.model import VehicleDetector


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None) -> dict[str, float]:
    """One pass over `loader`. Training if `optimizer` is given, else evaluation.

    Returns the mean of each loss component. The components are tracked separately because
    the total is dominated by the objectness term: a total that falls while `box` stays flat
    means the model is only learning to say "no vehicle", which the total alone would hide.
    """
    train = optimizer is not None
    model.train(train)

    sums, n_batches = {"total": 0.0, "obj": 0.0, "box": 0.0}, 0
    dev = device()

    with torch.set_grad_enabled(train):
        for imgs, obj_t, box_t, _ in loader:
            imgs, obj_t, box_t = imgs.to(dev), obj_t.to(dev), box_t.to(dev)

            pred = model(imgs)
            loss, parts = criterion(pred, obj_t, box_t)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            for k in sums:
                sums[k] += parts[k]
            n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


def overfit_one_batch(steps: int = 300) -> float:
    """Sanity check: drive the loss on a single batch to ~0.

    This is the cheapest possible proof that image loading, target encoding, the model, the
    decoder and the loss all agree with each other. A pipeline with a coordinate bug plateaus
    here instead of converging.
    """
    dev = device()
    loaders = build_loaders(augment=False)

    # Deliberately pick a batch that *contains vehicles*: 64.5% of this dataset is empty
    # road, and a batch of pure background would drive the loss to zero without proving that
    # box regression works at all.
    batch = None
    for candidate in loaders["train"]:
        if candidate[1].sum() >= 4:  # at least 4 positive cells in the batch
            batch = candidate
            break
    assert batch is not None, "no batch with >= 4 vehicles found"

    imgs, obj_t, box_t, _ = (x.to(dev) if torch.is_tensor(x) else x for x in batch)
    print(f"overfitting one batch: {imgs.shape[0]} images, {int(obj_t.sum())} positive cells")

    model = VehicleDetector().to(dev)
    criterion = DetectionLoss().to(dev)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    first = None
    for step in range(steps):
        model.train()
        pred = model(imgs)
        loss, parts = criterion(pred, obj_t, box_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        first = first or parts["total"]
        if step % 50 == 0 or step == steps - 1:
            print(
                f"  step {step:4d}  total {parts['total']:.5f}  "
                f"obj {parts['obj']:.5f}  box {parts['box']:.5f}"
            )

    print(f"\nloss {first:.4f} -> {parts['total']:.4f}  ({100*(1-parts['total']/first):.1f}% down)")
    return parts["total"]


def train(args) -> Path:
    """Full training run. Returns the directory holding the best checkpoint."""
    dev = device()
    out_dir = config.RUNS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_loaders(
        augment=not args.no_augment,
        split_mode=args.split_mode,
        assign=args.assign,
        stride=args.stride,
        img_size=args.img_size,
    )
    model = VehicleDetector(
        backbone=args.backbone, freeze=not args.unfreeze, stride=args.stride
    ).to(dev)
    criterion = DetectionLoss(
        box_loss=args.box_loss,
        imbalance=args.imbalance,
        lambda_obj=args.lambda_obj,
        lambda_box=args.lambda_box,
        assign=args.assign,
    ).to(dev)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"device={dev}  backbone={args.backbone}  trainable params={model.trainable_parameters():,}")
    print(f"train={len(loaders['train'].dataset)}  val={len(loaders['val'].dataset)}  "
          f"test={len(loaders['test'].dataset)}")

    history, best_val, t0 = [], float("inf"), time.time()

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, loaders["train"], criterion, optimizer)
        va = run_epoch(model, loaders["val"], criterion)
        scheduler.step()

        history.append({"epoch": epoch, "train": tr, "val": va})
        marker = ""
        if va["total"] < best_val:
            best_val = va["total"]
            torch.save(
                {"model": model.state_dict(), "config": vars(args), "epoch": epoch, "val": va},
                out_dir / "best.pt",
            )
            marker = "  <- best"

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train {tr['total']:.4f} (obj {tr['obj']:.4f} box {tr['box']:.4f})  "
            f"val {va['total']:.4f} (obj {va['obj']:.4f} box {va['box']:.4f}){marker}"
        )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_curves(history, out_dir / "curves.png")

    print(f"\ndone in {time.time()-t0:.0f}s  best val loss {best_val:.4f}  -> {out_dir}")
    return out_dir


def plot_curves(history: list[dict], out_path: Path) -> None:
    """Train/val curves for the total loss and its two components."""
    import matplotlib

    matplotlib.use("Agg")  # no display on a headless run
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes, ["total", "obj", "box"], ["Total", "Objectness", "Box"]):
        ax.plot(epochs, [h["train"][key] for h in history], label="train")
        ax.plot(epochs, [h["val"][key] for h in history], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"{title} loss")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Train the Part 1 detection head")
    p.add_argument("--overfit", action="store_true", help="sanity check on a single batch")
    p.add_argument("--tag", default="baseline", help="name of the run -> runs/<tag>/")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--backbone", default=config.BACKBONE, choices=["resnet18", "mobilenet_v3_large"])
    p.add_argument("--box-loss", default=config.BOX_LOSS, choices=["l1", "ciou"])
    p.add_argument("--imbalance", default="pos_weight", choices=["pos_weight", "plain", "focal"])
    p.add_argument("--lambda-obj", type=float, default=config.LAMBDA_OBJ)
    p.add_argument("--lambda-box", type=float, default=config.LAMBDA_BOX)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--unfreeze", action="store_true", help="also train the backbone")
    p.add_argument(
        "--split-mode",
        default=config.SPLIT_MODE,
        choices=["temporal", "random"],
        help="temporal = honest (frames are from one video); random = leaky, for comparison",
    )
    p.add_argument(
        "--assign",
        default=config.ASSIGN,
        choices=["center", "multi"],
        help="center = task sheet (1 positive cell); multi = center + 2 neighbours (3x supervision)",
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=config.IMG_SIZE,
        help="network input size. Larger = small distant vehicles get more pixels, which the "
             "error analysis says is where every missed detection lives.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=config.STRIDE,
        choices=[16, 32],
        help="32 = 16x16 grid (task sheet); 16 = 32x32 grid (better on small distant vehicles)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.overfit:
        overfit_one_batch()
    else:
        train(args)
