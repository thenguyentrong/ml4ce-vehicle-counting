"""Inference: raw grid logits -> a list of boxes, via thresholding and NMS.

This is step 3b of the task sheet: "at inference time, threshold the objectness scores and
apply non-maximum suppression to the remaining boxes".

Why NMS is needed at all when each cell predicts at most one box: neighbouring cells around a
vehicle also fire (the feature map is smooth, and a car straddling a cell boundary excites
both cells), producing several heavily overlapping boxes for one car. NMS keeps the most
confident and discards the rest.

Author: Vinh Nguyen
"""

from __future__ import annotations

import torch
from torchvision.ops import nms

import config
from src.part1.losses import activate_box, decode_boxes


@torch.no_grad()
def decode_predictions(
    pred: torch.Tensor,
    score_thresh: float = config.SCORE_THRESH,
    nms_iou: float = config.NMS_IOU,
    img_size: int | None = None,
    assign: str = config.ASSIGN,
) -> list[dict[str, torch.Tensor]]:
    """(B, 5, G, G) raw logits -> one dict per image with `boxes` (N,4) and `scores` (N,).

    Boxes are in network-input pixels (0..img_size), the same frame as the ground truth
    returned by the dataset, so evaluation compares like with like.

    `assign` must match what the model was TRAINED with - it selects the offset activation.
    """
    img_size = img_size or config.IMG_SIZE
    scores_grid = torch.sigmoid(pred[:, 0])  # (B, G, G)
    boxes_grid = decode_boxes(activate_box(pred[:, 1:], assign), img_size)  # (B, 4, G, G)

    out = []
    for b in range(pred.shape[0]):
        scores = scores_grid[b].reshape(-1)  # (G*G,)
        boxes = boxes_grid[b].permute(1, 2, 0).reshape(-1, 4)  # (G*G, 4)

        keep = scores >= score_thresh
        boxes, scores = boxes[keep], scores[keep]

        if boxes.numel():
            boxes = boxes.clamp(0, img_size)  # a box may extend past the image edge
            keep = nms(boxes, scores, nms_iou)
            boxes, scores = boxes[keep], scores[keep]

        out.append({"boxes": boxes.cpu(), "scores": scores.cpu()})
    return out


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between (N, 4) and (M, 4) boxes -> (N, M). Empty-safe."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]))

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)
