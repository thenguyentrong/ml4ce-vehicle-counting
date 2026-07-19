"""Does the tracker keep identities, and does the counter count each vehicle exactly once?

Tracking and counting fail *silently* in the same way the grid encoding does: a broken tracker
still produces a plausible-looking video with boxes and IDs on it, and the only symptom is that
the final number is wrong - which you cannot notice without a ground truth. So the association
and crossing logic are tested on synthetic boxes whose correct answer is known by construction,
independently of any detector.

Run: `uv run pytest tests/test_tracking.py -v`

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.part2.counter import LineCounter, segment_crossing
from src.part2.suggest_line import alignment, dominant_flow_axis, suggest
from src.part2.tracker import IoUTracker, iou_matrix, match_greedy, match_hungarian


def box(cx, cy, w=100.0, h=80.0):
    """Box centred on (cx, cy) as (x1, y1, x2, y2)."""
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# --------------------------------------------------------------------------------------
# IoU
# --------------------------------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    boxes = np.array([box(100, 100)], dtype=np.float32)
    assert iou_matrix(boxes, boxes)[0, 0] == 1.0


def test_iou_disjoint_boxes_is_zero():
    a = np.array([box(100, 100)], dtype=np.float32)
    b = np.array([box(1000, 1000)], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_matches_hand_computed_value():
    """The worked example from the module docstring: two 100x80 boxes offset by (20, 10).

    intersection 80 * 70 = 5600; union 8000 + 8000 - 5600 = 10400; IoU = 0.5385.
    """
    a = np.array([[100.0, 100.0, 200.0, 180.0]], dtype=np.float32)
    b = np.array([[120.0, 110.0, 220.0, 190.0]], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == np.float32(5600 / 10400)


def test_iou_matrix_shape_with_empty_input():
    """An empty frame must give an empty matrix, not raise - it happens on real footage."""
    assert iou_matrix(np.zeros((0, 4)), np.array([box(0, 0)])).shape == (0, 1)
    assert iou_matrix(np.array([box(0, 0)]), np.zeros((0, 4))).shape == (1, 0)


# --------------------------------------------------------------------------------------
# Matching strategies
# --------------------------------------------------------------------------------------


def test_hungarian_beats_greedy_on_the_occlusion_case():
    """The case from `match_hungarian`'s docstring, where greedy provably picks wrong.

                D1     D2
        T1     0.60   0.50
        T2     0.55   0.00

    Greedy takes the single best pair (T1-D1, 0.60) and strands T2 with a sub-threshold 0.00.
    Hungarian compares totals - 0.60 vs 1.05 - and keeps both tracks alive.
    """
    iou = np.array([[0.60, 0.50], [0.55, 0.00]], dtype=np.float32)

    greedy = dict(match_greedy(iou, thresh=0.3))
    assert greedy == {0: 0}, "greedy matches only T1, T2 is lost"

    hungarian = dict(match_hungarian(iou, thresh=0.3))
    assert hungarian == {0: 1, 1: 0}, "hungarian keeps both tracks by taking the better total"


def test_hungarian_still_applies_the_threshold():
    """linear_sum_assignment returns a COMPLETE assignment; sub-threshold pairs must be dropped.

    Without this filter the tracker silently glues unrelated vehicles together.
    """
    iou = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    assert match_hungarian(iou, thresh=0.3) == []


def test_both_matchers_agree_when_the_assignment_is_unambiguous():
    iou = np.array([[0.9, 0.0], [0.0, 0.8]], dtype=np.float32)
    assert sorted(match_greedy(iou, 0.3)) == sorted(match_hungarian(iou, 0.3)) == [(0, 0), (1, 1)]


# --------------------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------------------


def test_temporal_thresholds_scale_with_frame_rate():
    """The same config must mean the same *duration* on any video, not the same frame count.

    The course tests the model on a separate video we never see. If `max_age` were a fixed frame
    count, a 60 fps clip would halve the time a track survives an occlusion and fragment it, with
    no error raised anywhere.
    """
    slow = IoUTracker(fps=30.0)
    fast = IoUTracker(fps=60.0)
    assert fast.max_age == 2 * slow.max_age
    assert fast.min_hits == 2 * slow.min_hits
    assert slow.max_age / 30.0 == pytest.approx(config.TRACK_MAX_AGE_SECONDS, abs=0.02)


def test_thresholds_never_round_down_to_zero():
    """A very low frame rate must still give a usable tracker, not max_age=0."""
    tracker = IoUTracker(fps=1.0)
    assert tracker.max_age >= 1
    assert tracker.min_hits >= 1


def test_track_keeps_its_id_while_the_vehicle_moves():
    """A car moving 15 px per frame overlaps itself heavily, so its ID must never change."""
    tracker = IoUTracker(match="hungarian")
    for step in range(10):
        tracks = tracker.update(np.array([box(100 + 15 * step, 100)], dtype=np.float32))
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 10


def test_unmatched_detection_gets_a_new_id():
    tracker = IoUTracker()
    tracker.update(np.array([box(100, 100)], dtype=np.float32))
    tracks = tracker.update(np.array([box(100, 100), box(900, 900)], dtype=np.float32))
    assert sorted(t.track_id for t in tracks) == [1, 2]


def test_track_is_terminated_after_max_age_misses():
    """A track must survive a brief gap but eventually be dropped, or tracks accumulate forever."""
    tracker = IoUTracker(max_age=3)
    tracker.update(np.array([box(100, 100)], dtype=np.float32))

    for _ in range(3):
        tracks = tracker.update(np.zeros((0, 4), dtype=np.float32))
        assert len(tracks) == 1, "must survive a short occlusion"

    assert tracker.update(np.zeros((0, 4), dtype=np.float32)) == []


def test_brief_occlusion_does_not_split_a_track():
    """Disappear for 2 frames, come back: same vehicle, so the same ID - not a new one."""
    tracker = IoUTracker(max_age=5)
    tracker.update(np.array([box(100, 100)], dtype=np.float32))
    tracker.update(np.zeros((0, 4), dtype=np.float32))
    tracker.update(np.zeros((0, 4), dtype=np.float32))
    tracks = tracker.update(np.array([box(105, 100)], dtype=np.float32))
    assert [t.track_id for t in tracks] == [1]


# --------------------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------------------


def test_segment_crossing_detects_direction():
    line = ((0.0, 100.0), (200.0, 100.0))
    assert segment_crossing(line, (50.0, 50.0), (50.0, 150.0)) == 1  # downward
    assert segment_crossing(line, (50.0, 150.0), (50.0, 50.0)) == -1  # upward
    assert segment_crossing(line, (50.0, 10.0), (50.0, 50.0)) == 0  # never reaches it


def test_segment_crossing_ignores_the_lines_infinite_extension():
    """A vehicle crossing far outside the drawn segment must not be counted.

    This is what allows the line to span one carriageway without also catching a side road.
    """
    line = ((0.0, 100.0), (200.0, 100.0))
    assert segment_crossing(line, (900.0, 50.0), (900.0, 150.0)) == 0


def test_vehicle_is_counted_exactly_once_even_while_it_keeps_moving():
    """The core guarantee: one vehicle, one count - no matter how many frames it stays visible."""
    tracker = IoUTracker()
    counter = LineCounter(((0.0, 300.0), (1000.0, 300.0)))

    # Drive one box from y=100 down to y=500, crossing y=300 once.
    for frame_idx, y in enumerate(range(100, 520, 20)):
        tracks = tracker.update(np.array([box(500, y)], dtype=np.float32))
        counter.update(tracks, frame_idx)

    assert counter.summary()["total"] == 1
    assert counter.counts[1] == 1


def test_opposite_directions_are_tallied_separately():
    tracker = IoUTracker()
    counter = LineCounter(((0.0, 300.0), (1000.0, 300.0)))

    for frame_idx, step in enumerate(range(0, 420, 20)):
        down = box(300, 100 + step)  # travelling towards the camera
        up = box(700, 500 - step)  # travelling away
        tracks = tracker.update(np.array([down, up], dtype=np.float32))
        counter.update(tracks, frame_idx)

    assert counter.counts[1] == 1
    assert counter.counts[-1] == 1
    assert counter.summary()["total"] == 2


def test_unconfirmed_track_is_not_counted():
    """A one-frame detector false positive must not become a phantom vehicle in the count."""
    tracker = IoUTracker(min_hits=3)  # stated here, not inherited from config
    counter = LineCounter(((0.0, 300.0), (1000.0, 300.0)))

    # Two frames only - one short of min_hits - while crossing the line.
    for frame_idx, y in enumerate((280, 320)):
        tracks = tracker.update(np.array([box(500, y)], dtype=np.float32))
        counter.update(tracks, frame_idx)

    assert counter.summary()["total"] == 0


# --------------------------------------------------------------------------------------
# Placing the counting line on an unseen video
# --------------------------------------------------------------------------------------


def test_flow_axis_survives_two_opposing_streams():
    """A two-way road must yield ONE axis, not cancel to zero.

    Averaging the displacement vectors of a balanced two-way road gives ~(0,0), which carries no
    information. The orientation tensor makes v and -v reinforce instead.
    """
    moving = {
        1: [(0.0, 500.0), (1800.0, 560.0)],  # left to right
        2: [(1800.0, 620.0), (0.0, 560.0)],  # right to left
    }
    axis = dominant_flow_axis(moving)
    assert abs(axis[0]) > 0.95, "flow is lateral, so the axis must be near-horizontal"


def test_lateral_traffic_gets_a_vertical_counting_line():
    """The bug this whole feature exists to prevent.

    Traffic running left-right across the frame must be counted with a VERTICAL line, so that
    direction is read from the dominant component of motion. A horizontal line would be crossed
    too - by the small vertical drift - but its direction call would rest on that residual.
    """
    moving = {
        i: [(100.0, 500.0 + 20 * i), (1800.0, 560.0 + 20 * i)] for i in range(6)
    }
    moving.update({
        10 + i: [(1800.0, 700.0 + 20 * i), (100.0, 640.0 + 20 * i)] for i in range(6)
    })

    best = suggest(moving, width=1920, height=1080)[0]
    assert best["orientation"] == "vertical"
    assert best["alignment"] > 0.9


def test_alignment_prefers_the_line_the_flow_crosses_head_on():
    lateral = np.array([1.0, 0.0])
    assert alignment(lateral, "vertical") == pytest.approx(1.0)
    assert alignment(lateral, "horizontal") == pytest.approx(0.0, abs=1e-9)


def test_stationary_vehicle_on_the_line_is_never_counted():
    """Parked cars sit in frame for the whole clip; they must contribute nothing to the count."""
    tracker = IoUTracker()
    counter = LineCounter(((0.0, 300.0), (1000.0, 300.0)))

    for frame_idx in range(30):
        tracks = tracker.update(np.array([box(500, 150)], dtype=np.float32))
        counter.update(tracks, frame_idx)

    assert counter.summary()["total"] == 0
