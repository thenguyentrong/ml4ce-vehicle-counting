"""Does the grid target encoding actually preserve the boxes?

The target encoding is where a detector fails *silently*: a wrong coordinate convention or an
off-by-one in the cell index still trains, still shows a falling loss, and simply never
detects anything well. The only way to know is to encode ground-truth boxes and decode them
straight back - a correct encoder round-trips to IoU 1.0.

Run: `uv run pytest tests/test_encoding.py -v`

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import config
from src.part1.dataset import decode_target, encode_target
from src.part1.infer import box_iou
from src.part1.losses import decode_boxes


def test_roundtrip_recovers_boxes_exactly():
    """encode -> decode must return the original boxes (IoU = 1), up to float precision."""
    img_w, img_h = 676, 380  # the dataset's native size
    boxes = np.array(
        [
            [100.0, 150.0, 200.0, 192.0],  # a typical car: 100 x 42 px
            [400.0, 200.0, 500.0, 250.0],
            [10.0, 10.0, 60.0, 40.0],  # near the top-left corner
            [600.0, 330.0, 670.0, 375.0],  # near the bottom-right corner
        ],
        dtype=np.float32,
    )

    obj, box, collisions = encode_target(boxes, img_w, img_h, assign="center")
    assert collisions == 0, "these four boxes are far apart; none should collide"
    assert obj.sum() == len(boxes), "one positive cell per box"

    recovered = decode_target(obj, box)

    # The original boxes, expressed in network-input pixels - the frame decode works in.
    scale = np.array([config.IMG_SIZE / img_w, config.IMG_SIZE / img_h] * 2, dtype=np.float32)
    expected = torch.from_numpy(boxes * scale)

    iou = box_iou(recovered, expected)
    best = iou.max(dim=1).values  # decode returns boxes in grid order, not input order
    assert torch.allclose(best, torch.ones_like(best), atol=1e-4), f"round-trip IoU: {best}"


def test_center_lands_in_the_claimed_cell():
    """The positive cell must be the one containing the box *center* - the task sheet's rule."""
    boxes = np.array([[320.0, 160.0, 352.0, 192.0]], dtype=np.float32)  # center (336, 176)
    obj, _, _ = encode_target(
        boxes, 512, 512, grid=16, img_size=512, assign="center"
    )  # already at network scale

    j, i = torch.nonzero(obj)[0].tolist()  # row, column
    assert (i, j) == (336 // 32, 176 // 32) == (10, 5)


def test_offsets_stay_in_unit_range():
    """off_x, off_y in [0,1) and w, h in (0,1] - otherwise a sigmoid head cannot represent them."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        x1, y1 = rng.uniform(0, 600), rng.uniform(0, 330)
        boxes = np.array([[x1, y1, x1 + rng.uniform(20, 70), y1 + rng.uniform(18, 45)]], np.float32)
        _, box, _ = encode_target(boxes, 676, 380, assign="center")

        vals = box[box != 0]
        assert (vals >= 0).all() and (vals <= 1).all(), f"target outside [0,1]: {vals}"


def test_collision_is_detected_and_reported():
    """Two centers in one cell: the encoder must report the loss instead of hiding it."""
    boxes = np.array(
        [[100.0, 100.0, 130.0, 120.0], [102.0, 101.0, 132.0, 121.0]],  # nearly identical centers
        dtype=np.float32,
    )
    obj, _, collisions = encode_target(boxes, 676, 380, assign="center")
    assert collisions == 1, "the second box shares a cell and must be counted as lost"
    assert obj.sum() == 1, "one cell can only hold one box"


def test_decode_boxes_matches_decode_target():
    """The differentiable decoder used in the loss and the plain one must agree exactly.

    If they drift apart, the model optimizes one coordinate convention and is evaluated
    against another - and every metric silently understates the model.
    """
    boxes = np.array([[100.0, 150.0, 200.0, 192.0], [400.0, 200.0, 500.0, 250.0]], np.float32)
    obj, box, _ = encode_target(boxes, 676, 380, assign="center")

    from_target = decode_target(obj, box)
    from_loss = decode_boxes(box.unsqueeze(0))[0].permute(1, 2, 0)[obj > 0.5]

    iou = box_iou(from_target, from_loss).max(dim=1).values
    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-4)


def test_multi_assign_marks_three_cells_per_box():
    """'multi' must light up the center cell plus the two neighbours the center leans toward."""
    # Center at (336, 176) -> grid (10.5, 5.5): leans right and down -> cells (10,5),(11,5),(10,6).
    boxes = np.array([[320.0, 160.0, 352.0, 192.0]], dtype=np.float32)
    obj, _, _ = encode_target(boxes, 512, 512, grid=16, img_size=512, assign="multi")

    cells = {(int(i), int(j)) for j, i in torch.nonzero(obj).tolist()}
    assert cells == {(10, 5), (11, 5), (10, 6)}, cells


def test_multi_assign_roundtrips_through_its_own_activation():
    """The whole point of 'multi': every assigned cell must decode back to the SAME box.

    A neighbouring cell has to express a center lying outside itself (offset > 1 or < 0),
    which only works if the offset activation is widened to [-0.5, 1.5]. If encoder and
    activation disagree, this test fails - and nothing else would have caught it.
    """
    boxes = np.array([[100.0, 150.0, 200.0, 192.0]], dtype=np.float32)
    obj, box, _ = encode_target(boxes, 676, 380, assign="multi")

    assert obj.sum() == 3, "expected the center cell plus two neighbours"

    # Offsets must fall inside the widened range the 'multi' activation can actually produce.
    lo, hi = config.OFFSET_RANGE["multi"]
    offs = box[:2][box[:2] != 0]
    assert (offs >= lo).all() and (offs <= hi).all(), f"offsets outside [{lo},{hi}]: {offs}"

    # All three cells must decode to the same box as the original.
    recovered = decode_target(obj, box)
    scale = np.array([config.IMG_SIZE / 676, config.IMG_SIZE / 380] * 2, dtype=np.float32)
    expected = torch.from_numpy(boxes * scale)

    iou = box_iou(recovered, expected).squeeze(1)
    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-4), f"per-cell IoU: {iou}"


def test_activation_can_reach_the_multi_targets():
    """Sanity: the widened activation's range must cover what the encoder actually emits.

    A plain sigmoid ([0,1]) cannot produce the offset 1.2 a left-neighbour cell needs, so a
    'multi' model trained with the 'center' activation would silently never fit its targets.
    """
    from src.part1.losses import activate_box

    raw = torch.linspace(-8, 8, 64).view(1, 1, 8, 8).repeat(1, 4, 1, 1)

    center = activate_box(raw, "center")[:, :2]
    assert float(center.min()) >= 0.0 and float(center.max()) <= 1.0

    multi = activate_box(raw, "multi")[:, :2]
    assert float(multi.min()) < 0.0 and float(multi.max()) > 1.0, "range was not widened"


@pytest.mark.parametrize("n", [0, 1, 5])
def test_empty_and_multi_box_images(n):
    """Zero boxes (64.5% of this dataset) must encode cleanly to an all-negative target."""
    boxes = np.array([[50 + 80 * k, 50, 110 + 80 * k, 90] for k in range(n)], dtype=np.float32)
    boxes = boxes.reshape(-1, 4) if n else np.zeros((0, 4), dtype=np.float32)

    obj, box, collisions = encode_target(boxes, 676, 380, assign="center")
    assert obj.shape == (config.GRID, config.GRID)
    assert box.shape == (4, config.GRID, config.GRID)
    assert obj.sum() == n - collisions
