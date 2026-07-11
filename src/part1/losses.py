"""Loss for the Part 1 detector: objectness (BCE) + box regression (L1 or CIoU).

    total = LAMBDA_OBJ * objectness_loss + LAMBDA_BOX * box_loss

Two things make this loss non-obvious, and both are worth a slide:

**Class imbalance.** Of the 256 cells in the grid, typically 0-3 are positive. A model that
predicts "no object" everywhere already achieves ~99% accuracy, and plain BCE is happy to sit
there. Two ways out, both implemented so they can be compared:
  - `pos_weight`: multiply the loss of positive cells by a constant (config.POS_WEIGHT)
  - focal loss: down-weight the *easy* negatives instead of up-weighting the positives

**The box loss must only look at positive cells.** In a negative cell the four box channels are
meaningless - there is no ground-truth box to regress towards. Averaging them in would drag
every box prediction towards zero. Hence the mask.

Author: Vinh Nguyen
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def activate_box(raw: torch.Tensor, assign: str = config.ASSIGN) -> torch.Tensor:
    """Raw head logits (B, 4, G, G) -> box parameters in the same space the targets live in.

    The offset activation MUST match the assignment scheme, and this is the one place it is
    defined so the loss, the decoder and inference cannot drift apart:

      "center": offsets = sigmoid(t)          -> [0, 1]     (center stays inside its own cell)
      "multi":  offsets = sigmoid(t)*2 - 0.5  -> [-0.5, 1.5] (a neighbour cell must be able to
                                                              place the center outside itself)

    Sizes are always sigmoid -> a fraction of the image in (0, 1].

    Get this wrong and nothing crashes: the model simply cannot represent the target, and the
    box loss plateaus at a suspiciously non-zero value.
    """
    lo, hi = config.OFFSET_RANGE[assign]
    offsets = torch.sigmoid(raw[:, :2]) * (hi - lo) + lo
    sizes = torch.sigmoid(raw[:, 2:])
    return torch.cat([offsets, sizes], dim=1)


def decode_boxes(box_params: torch.Tensor, img_size: int = config.IMG_SIZE) -> torch.Tensor:
    """(B, 4, G, G) of sigmoid-space (off_x, off_y, w, h) -> (B, 4, G, G) of (x1, y1, x2, y2) px.

    Differentiable, so it can be used inside the CIoU loss. Mirrors
    `dataset.decode_target` exactly - if these two ever disagree, the model trains against
    one coordinate convention and is evaluated against another.
    """
    b, _, gy, gx = box_params.shape
    cell = img_size / gx

    # Cell indices, broadcast over the batch.
    ii = torch.arange(gx, device=box_params.device).view(1, 1, gx)
    jj = torch.arange(gy, device=box_params.device).view(1, gy, 1)

    cx = (ii + box_params[:, 0]) * cell
    cy = (jj + box_params[:, 1]) * cell
    w = box_params[:, 2] * img_size
    h = box_params[:, 3] * img_size

    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete-IoU loss between two (N, 4) sets of (x1, y1, x2, y2) boxes.

    CIoU = IoU - center_distance_penalty - aspect_ratio_penalty. Unlike L1 it optimizes the
    quantity we actually evaluate on (IoU >= 0.5), and it is scale-invariant: a 10 px error
    on a 20 px box matters more than on a 200 px box, which plain L1 cannot express.
    """
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    pw, ph = (px2 - px1).clamp(min=0), (py2 - py1).clamp(min=0)
    tw, th = (tx2 - tx1).clamp(min=0), (ty2 - ty1).clamp(min=0)

    inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0) * (
        torch.min(py2, ty2) - torch.max(py1, ty1)
    ).clamp(min=0)
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    # Smallest box enclosing both, for the distance and aspect terms.
    cw = (torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(min=0)
    ch = (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(min=0)
    c2 = cw**2 + ch**2 + eps

    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2  # squared center distance

    v = (4 / (torch.pi**2)) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return (1 - iou + rho2 / c2 + alpha * v).mean()


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, alpha=0.25, gamma=2.0) -> torch.Tensor:
    """Focal loss: BCE re-weighted by (1 - p_t)^gamma so easy negatives stop dominating."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


class DetectionLoss(nn.Module):
    """Objectness + box loss, with the box term restricted to positive cells.

    Args:
        box_loss:   "l1" or "ciou"                      (ablation)
        imbalance:  "pos_weight" | "plain" | "focal"    (ablation)
    """

    def __init__(
        self,
        box_loss: str = config.BOX_LOSS,
        imbalance: str = "pos_weight",
        lambda_obj: float = config.LAMBDA_OBJ,
        lambda_box: float = config.LAMBDA_BOX,
        pos_weight: float = config.POS_WEIGHT,
        assign: str = config.ASSIGN,
    ):
        super().__init__()
        self.box_loss = box_loss
        self.imbalance = imbalance
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box
        self.assign = assign
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(
        self, pred: torch.Tensor, obj_t: torch.Tensor, box_t: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Args:
            pred:  (B, 5, G, G) raw logits from the model
            obj_t: (B, G, G)    1.0 where a box center falls
            box_t: (B, 4, G, G) target (off_x, off_y, w, h), valid only where obj_t == 1

        Returns (total_loss, {component: value}) - the components are logged per epoch, since
        a total that falls while the box term is flat means the model is just learning to say
        "no object", which the total alone would hide.
        """
        obj_logits = pred[:, 0]
        box_params = activate_box(pred[:, 1:], self.assign)  # same space as the targets

        # ---- objectness, over every cell ----------------------------------------------
        if self.imbalance == "focal":
            loss_obj = focal_bce(obj_logits, obj_t)
        elif self.imbalance == "plain":
            loss_obj = F.binary_cross_entropy_with_logits(obj_logits, obj_t)
        else:  # "pos_weight"
            loss_obj = F.binary_cross_entropy_with_logits(
                obj_logits, obj_t, pos_weight=self.pos_weight
            )

        # ---- box regression, positive cells only --------------------------------------
        mask = obj_t > 0.5
        n_pos = int(mask.sum())

        if n_pos == 0:
            # A batch of pure empty road (64.5% of this dataset is empty!) has no box to
            # regress. Return a real zero tensor, not the float 0.0, so .backward() works.
            loss_box = box_params.sum() * 0.0
        elif self.box_loss == "ciou":
            pred_xyxy = decode_boxes(box_params).permute(0, 2, 3, 1)[mask]  # (n_pos, 4)
            tgt_xyxy = decode_boxes(box_t).permute(0, 2, 3, 1)[mask]
            loss_box = ciou_loss(pred_xyxy, tgt_xyxy)
        else:  # "l1", in the sigmoid space the model directly predicts
            loss_box = F.l1_loss(
                box_params.permute(0, 2, 3, 1)[mask], box_t.permute(0, 2, 3, 1)[mask]
            )

        total = self.lambda_obj * loss_obj + self.lambda_box * loss_box

        # .detach() before float(): the returned dict is for logging only, and keeping it
        # attached to the graph would silently retain the whole batch's activations.
        return total, {
            "total": float(total.detach()),
            "obj": float(loss_obj.detach()),
            "box": float(loss_box.detach()),
            "n_pos": n_pos,
        }
