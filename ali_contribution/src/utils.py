"""
Small helper functions shared across the project:
- box format conversions
- IoU computation
- non-maximum suppression (NMS)
"""
import torch


def xyxy_to_cxcywh(boxes):
    """boxes: (N, 4) tensor in [xmin, ymin, xmax, ymax] -> [cx, cy, w, h]"""
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + w / 2
    cy = ymin + h / 2
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes):
    """boxes: (N, 4) tensor in [cx, cy, w, h] -> [xmin, ymin, xmax, ymax]"""
    cx, cy, w, h = boxes.unbind(-1)
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def box_iou(boxes1, boxes2):
    """
    boxes1: (N, 4), boxes2: (M, 4), both in [xmin, ymin, xmax, ymax].
    Returns an (N, M) IoU matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def nms(boxes, scores, iou_threshold=0.5):
    """
    Greedy NMS. boxes: (N, 4) xyxy, scores: (N,).
    Returns indices of boxes to keep, sorted by score descending.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long)
