"""Part 1 data loader: images + CSV boxes -> (image tensor, grid target).

This is the heart of the "understand how a detector works" exercise. A detector's target
encoding is where most silent bugs live, so the encode step has an exact inverse
(`decode_target`) and `tests/test_encoding.py` asserts that encode -> decode round-trips to
IoU 1.0 against the original boxes.

Target layout, per image, on a GRID x GRID grid (16x16 for a 512x512 input at stride 32):

    obj  [GRID, GRID]      1.0 in the cell containing a box center, 0.0 everywhere else
    box  [4, GRID, GRID]   (off_x, off_y, w, h) - only meaningful where obj == 1
                             off_x, off_y : center offset *within* the cell, in [0, 1)
                             w, h         : box size as a fraction of the image, in (0, 1]

All four values live in [0, 1], so the model can produce them with a plain sigmoid and no
anchors are needed - a legitimate simplification for a single class of similarly sized objects.

Author: Vinh Nguyen
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from src import data as data_mod

# ImageNet statistics: the backbone is pretrained on ImageNet and expects this normalization.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def assigned_cells(gx: float, gy: float, grid: int, assign: str) -> list[tuple[int, int]]:
    """Which cells are responsible for a box whose center is at grid coordinate (gx, gy)?

    "center": just the cell the center falls in - literally what the task sheet prescribes.

    "multi":  that cell plus its two nearest neighbours (the ones the center leans towards).
              A center at gx=4.2 leans left, so cell 3 also predicts it; at gx=4.8 it leans
              right, so cell 5 does. Those neighbouring cells must then express a center that
              lies *outside themselves* (offset 1.2 and -0.2 respectively), which is exactly
              why the offset range widens to [-0.5, 1.5] for this mode.

              This triples the positive supervision - the binding constraint here, with only
              453 training boxes - and turns the neighbours from fragment-emitters into
              agreeing votes that NMS merges.
    """
    i, j = min(int(gx), grid - 1), min(int(gy), grid - 1)
    cells = [(i, j)]

    if assign == "multi":
        fx, fy = gx - i, gy - j  # where inside the cell the center sits, in [0, 1)
        if fx < 0.5 and i > 0:
            cells.append((i - 1, j))
        elif fx >= 0.5 and i < grid - 1:
            cells.append((i + 1, j))
        if fy < 0.5 and j > 0:
            cells.append((i, j - 1))
        elif fy >= 0.5 and j < grid - 1:
            cells.append((i, j + 1))

    return cells


def encode_target(
    boxes: np.ndarray,
    img_w: int,
    img_h: int,
    grid: int = config.GRID,
    img_size: int = config.IMG_SIZE,
    assign: str = config.ASSIGN,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Turn pixel boxes from the *original* image into a grid target.

    Args:
        boxes: (N, 4) array of [xmin, ymin, xmax, ymax] in original-image pixels.
        img_w, img_h: size of the original image (boxes are rescaled to `img_size`).
        grid: grid resolution (16).
        img_size: network input size (512).

    Returns:
        obj:       (grid, grid) float tensor, 1.0 where a box center falls.
        box:       (4, grid, grid) float tensor of (off_x, off_y, w, h), all in [0, 1].
        collisions: how many boxes were *lost* because an earlier box already claimed their
                    cell. The prescribed one-box-per-cell design cannot represent them; we
                    count them so the recall ceiling can be reported honestly.

    A cell is positive if a box *center* falls inside it - exactly as the task sheet states.
    """
    obj = torch.zeros((grid, grid), dtype=torch.float32)
    box = torch.zeros((4, grid, grid), dtype=torch.float32)
    collisions = 0

    # Rescale boxes from the original image to the img_size x img_size network input.
    sx, sy = img_size / img_w, img_size / img_h
    cell = img_size / grid  # 32 px

    for raw in boxes:
        # float() throughout: numpy scalars cannot be assigned into a torch tensor.
        xmin, xmax = float(raw[0]) * sx, float(raw[2]) * sx
        ymin, ymax = float(raw[1]) * sy, float(raw[3]) * sy

        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        w, h = xmax - xmin, ymax - ymin

        gx, gy = cx / cell, cy / cell  # center in grid coordinates

        # A center exactly on the right/bottom edge would index out of the grid -> clamp
        # happens inside assigned_cells().
        owners = assigned_cells(gx, gy, grid, assign)
        counted_collision = False

        for i, j in owners:
            # Cell already claimed by an earlier box: one cell can only carry one box, so
            # this one is dropped. Count it once per box, not once per assigned cell.
            if obj[j, i] == 1.0 and not counted_collision:
                collisions += 1
                counted_collision = True

            obj[j, i] = 1.0
            box[0, j, i] = gx - i  # center offset relative to THIS cell
            box[1, j, i] = gy - j  # in [0,1) for the owning cell, [-0.5,1.5] for a neighbour
            box[2, j, i] = w / img_size  # size as a fraction of the image, in (0, 1]
            box[3, j, i] = h / img_size

    return obj, box, collisions


def decode_target(
    obj: torch.Tensor,
    box: torch.Tensor,
    grid: int = config.GRID,
    img_size: int = config.IMG_SIZE,
) -> torch.Tensor:
    """Exact inverse of `encode_target`: grid target -> (N, 4) boxes in `img_size` pixels.

    Used by the round-trip test and, with predicted tensors, by inference.
    """
    cell = img_size / grid
    js, is_ = torch.nonzero(obj > 0.5, as_tuple=True)  # rows, columns

    off_x, off_y = box[0, js, is_], box[1, js, is_]
    w, h = box[2, js, is_] * img_size, box[3, js, is_] * img_size

    cx = (is_.float() + off_x) * cell
    cy = (js.float() + off_y) * cell

    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


class VehicleGridDataset(Dataset):
    """Images + grid targets for one split.

    Each item is (image [3, 512, 512], obj [16, 16], box [4, 16, 16], boxes_px [N, 4]).
    `boxes_px` are the ground-truth boxes in network-input pixels, kept for evaluation:
    scoring against the *encoded* target instead would hide exactly the boxes that the grid
    dropped, flattering our own recall.
    """

    def __init__(
        self,
        images: list[str],
        df,
        images_dir,
        augment: bool = False,
        assign: str = config.ASSIGN,
    ):
        self.images = images
        self.df = df
        self.images_dir = images_dir
        self.augment = augment
        self.assign = assign
        self.img_size = config.IMG_SIZE

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        name = self.images[idx]

        with Image.open(self.images_dir / name) as im:
            im = im.convert("RGB")
            img_w, img_h = im.size
            im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0

        boxes = data_mod.boxes_for_image(self.df, name)  # original-image pixels

        if self.augment:
            arr, boxes = self._augment(arr, boxes, img_w, img_h)

        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # HWC -> CHW

        obj, box, _ = encode_target(boxes, img_w, img_h, assign=self.assign)

        # Ground-truth boxes in network pixels, for evaluation.
        scale = np.array([self.img_size / img_w, self.img_size / img_h] * 2, dtype=np.float32)
        boxes_px = torch.from_numpy(boxes * scale) if len(boxes) else torch.zeros((0, 4))

        return img, obj, box, boxes_px

    def _augment(self, arr: np.ndarray, boxes: np.ndarray, img_w: int, img_h: int):
        """Horizontal flip and colour jitter.

        The flip must mirror the boxes too - forgetting that trains the model on wrong
        targets while the loss still looks healthy, which is a classic silent failure.
        """
        if config.AUG_HFLIP and np.random.rand() < 0.5:
            arr = arr[:, ::-1].copy()
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = img_w - boxes[:, [2, 0]]  # mirror, keeping xmin < xmax

        if config.AUG_COLOR_JITTER:
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            arr = np.clip((arr - 0.5) * contrast + 0.5 * brightness, 0.0, 1.0)

        return arr, boxes


def collate(batch):
    """Stack the fixed-size tensors; keep the variable-length GT box lists as a list."""
    imgs, objs, boxes, boxes_px = zip(*batch)
    return (
        torch.stack(imgs),
        torch.stack(objs),
        torch.stack(boxes),
        list(boxes_px),  # ragged: each image has a different number of boxes
    )


def build_loaders(
    augment: bool = True,
    split_mode: str = config.SPLIT_MODE,
    assign: str = config.ASSIGN,
) -> dict[str, DataLoader]:
    """DataLoaders for train/val/test, all sharing the split from `src.data`."""
    paths = data_mod.resolve_dataset_paths()
    df = data_mod.load_annotations(paths)
    splits = data_mod.make_splits(data_mod.list_images(paths), mode=split_mode)

    loaders = {}
    for split, names in splits.items():
        ds = VehicleGridDataset(
            names, df, paths.images_dir, augment=(augment and split == "train"), assign=assign
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=config.BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=config.NUM_WORKERS,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=config.NUM_WORKERS > 0,
        )
    return loaders
