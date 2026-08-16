"""
Training loop for the detection head.

Loss = BCE(objectness) + lambda_box * L1(box params, only on positive cells)
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import VehicleDataset, make_splits, IMG_SIZE, GRID
from src.model import VehicleDetector

CSV_PATH = "data/train_solution_bounding_boxes.csv"
IMG_DIR = "data/training_images"
CKPT_DIR = "checkpoints"
LAMBDA_BOX = 5.0     # weight of box regression loss vs. objectness loss
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-3
POS_WEIGHT = 20.0    # how much more heavily to weight positive (vehicle) cells in BCE loss

def detection_loss(pred, target):
    """
    pred, target: (B, GRID, GRID, 5)
    channel 0 = objectness, channels 1-4 = box params (offset_x, offset_y, w, h)
    """
    obj_pred = pred[..., 0]
    obj_target = target[..., 0]

    pos_weight = torch.tensor(POS_WEIGHT, device=pred.device)
    bce = nn.functional.binary_cross_entropy_with_logits(obj_pred, obj_target, pos_weight=pos_weight)

    pos_mask = obj_target > 0.5
    if pos_mask.sum() > 0:
        box_pred = pred[..., 1:][pos_mask]
        box_target = target[..., 1:][pos_mask]
        l1 = nn.functional.l1_loss(box_pred, box_target)
    else:
        l1 = torch.tensor(0.0, device=pred.device)

    total = bce + LAMBDA_BOX * l1
    return total, bce.item(), l1.item() if torch.is_tensor(l1) else l1


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total_loss, total_bce, total_l1 = 0.0, 0.0, 0.0

    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)

        with torch.set_grad_enabled(train):
            preds = model(imgs)
            loss, bce, l1 = detection_loss(preds, targets)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_bce += bce
        total_l1 += l1

    n = len(loader)
    return total_loss / n, total_bce / n, total_l1 / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_ids, val_ids, test_ids = make_splits(CSV_PATH)
    train_ds = VehicleDataset(CSV_PATH, IMG_DIR, train_ids)
    val_ds = VehicleDataset(CSV_PATH, IMG_DIR, val_ids)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = VehicleDetector().to(device)
    # only the head's parameters require grad, since the backbone is frozen
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_bce, train_l1 = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_bce, val_l1 = run_epoch(model, val_loader, optimizer, device, train=False)

        print(f"epoch {epoch:02d}  "
              f"train_loss={train_loss:.4f} (bce={train_bce:.4f}, l1={train_l1:.4f})  "
              f"val_loss={val_loss:.4f} (bce={val_bce:.4f}, l1={val_l1:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best.pt"))
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
