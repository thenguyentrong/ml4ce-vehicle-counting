"""Part 2 tracker: associate per-frame detections into persistent tracks using IoU.

A detector is memoryless - it re-answers "where are the vehicles?" from scratch on every frame
and has no idea that the car in frame 41 is the car from frame 40. Counting therefore cannot be
done on detections; a car queuing for three seconds would be counted ~90 times. The tracker is
what turns detections into *objects with identity*, and the count is over identities.

The association rule the task sheet prescribes:

    match detections of the current frame to the active tracks of the previous frame using the
    IoU between their boxes; assign a new ID to every unmatched detection; terminate tracks that
    have not been matched for a few consecutive frames.

Why IoU is a sensible similarity here: between two consecutive frames a real vehicle barely
moves, so its box overlaps its own previous box heavily, while a *different* vehicle nearby
overlaps much less. IoU is also scale invariant, which matters on this footage - a car near the
horizon is ~20 px wide and one in the foreground ~300 px, and a single threshold has to work for
both. A center-distance criterion would need a per-depth threshold; IoU does not.

Its failure mode is worth stating because it bounds what this design can do: if an object moves
further than its own size between frames, IoU collapses to 0 and the track breaks even though
nothing went wrong. That is a frame-rate/speed limit, not a bug - at 30 fps the vehicles here
move ~10-20 px against ~100 px boxes, so we are far from it. Fixing it would need motion
prediction (a Kalman filter, as in SORT); the task sheet does not ask for one and we do not add
one.

Two matching strategies are implemented so they can be compared on real data (the presentation
is explicitly graded on comparing methods); `--match` selects between them.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

import config

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


# --------------------------------------------------------------------------------------
# IoU
# --------------------------------------------------------------------------------------


def iou_matrix(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """IoU of every track box against every detection box.

    Args:
        tracks: (M, 4) array of track boxes, each (x1, y1, x2, y2).
        detections: (N, 4) array of detection boxes.

    Returns:
        (M, N) array where entry [i, j] is IoU(track i, detection j), in [0, 1].
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    t = np.asarray(tracks, dtype=np.float32)[:, None, :]  # (M, 1, 4)
    d = np.asarray(detections, dtype=np.float32)[None, :, :]  # (1, N, 4)

    # Intersection rectangle: the larger of the two left edges to the smaller of the right
    # edges. clip(min=0) is what makes non-overlapping pairs come out as 0 rather than as a
    # negative "area".
    inter_w = np.clip(np.minimum(t[..., 2], d[..., 2]) - np.maximum(t[..., 0], d[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(t[..., 3], d[..., 3]) - np.maximum(t[..., 1], d[..., 1]), 0, None)
    inter = inter_w * inter_h

    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    union = area_t + area_d - inter

    # A degenerate zero-area box would divide by zero; guard rather than emit a silent nan
    # that would then poison the whole assignment.
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# Matching strategies
# --------------------------------------------------------------------------------------


def match_greedy(iou: np.ndarray, thresh: float) -> list[tuple[int, int]]:
    """Repeatedly take the single highest-IoU pair still available.

    Fast and obvious, but only *locally* optimal: it commits to the best individual pair without
    considering what that costs the remaining tracks. See `match_hungarian` for the case where
    this demonstrably picks the wrong assignment.
    """
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs

    used_t: set[int] = set()
    used_d: set[int] = set()

    # Candidate pairs sorted by IoU, best first. argsort is ascending, hence the reverse.
    order = np.dstack(np.unravel_index(np.argsort(iou, axis=None)[::-1], iou.shape))[0]
    for i, j in order:
        i, j = int(i), int(j)
        if iou[i, j] < thresh:
            break  # sorted, so everything after this is below threshold too
        if i in used_t or j in used_d:
            continue
        pairs.append((i, j))
        used_t.add(i)
        used_d.add(j)
    return pairs


def match_hungarian(iou: np.ndarray, thresh: float) -> list[tuple[int, int]]:
    """Choose the assignment maximising *total* IoU (Hungarian / Kuhn-Munkres algorithm).

    Where this beats greedy - two tracks, two detections, T2 partly occluded so its box is
    clipped and its IoU is dragged down:

                D1     D2
        T1     0.60   0.50
        T2     0.55   0.00

    Greedy takes 0.60 (T1-D1) first, leaving T2 only D2 at 0.00, which is below threshold and is
    rejected. T2 dies and D2 is declared a new vehicle - a lost track *and* a phantom ID, and
    every phantom ID is a +1 error in the final count. The Hungarian algorithm compares the
    complete assignments, 0.60 + 0.00 against 0.50 + 0.55 = 1.05, and picks the second. Both
    tracks survive.

    Two details that are easy to get wrong:
      * `linear_sum_assignment` *minimises* cost, so it is fed 1 - IoU, not IoU.
      * It returns a *complete* assignment, including pairs it only made because it had to match
        something. The threshold must therefore be applied AFTER solving, not before - skipping
        that filter silently glues unrelated vehicles together.
    """
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs

    rows, cols = linear_sum_assignment(1.0 - iou)
    for i, j in zip(rows, cols):
        if iou[i, j] >= thresh:
            pairs.append((int(i), int(j)))
    return pairs


MATCHERS = {"greedy": match_greedy, "hungarian": match_hungarian}


# --------------------------------------------------------------------------------------
# Tracks
# --------------------------------------------------------------------------------------


@dataclass
class Track:
    """One tracked vehicle: a persistent identity plus its most recent box."""

    track_id: int
    box: Box
    hits: int = 1  # frames in which this track was matched to a detection
    age: int = 0  # frames since the track was created
    time_since_update: int = 0  # consecutive frames with no match -> triggers termination
    counted: bool = False  # set once the track has crossed the counting line
    trail: list[tuple[float, float]] = field(default_factory=list)  # center history
    min_hits: int = 3  # set by the tracker from the video's frame rate; see IoUTracker

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def confirmed(self) -> bool:
        """Whether this track has been seen often enough to be trusted.

        A single-frame false positive from the detector would otherwise create a track, cross
        the line and add a phantom vehicle to the count. Requiring `min_hits` matches before a
        track may be counted is what makes the count robust to detector flicker.

        `min_hits` is carried on the track rather than read from `config` at call time, so a
        track's behaviour is fixed by the tracker that made it. Reading a mutable global here
        would mean a test could not state the threshold it is asserting against - the same
        mistake the Part 1 tests made (see NOTES.md).
        """
        return self.hits >= self.min_hits

    def update(self, box: Box) -> None:
        self.box = box
        self.hits += 1
        self.time_since_update = 0
        self.trail.append(self.center)

    def mark_missed(self) -> None:
        self.time_since_update += 1


class IoUTracker:
    """The tracker itself: detections in, identified tracks out, one frame at a time.

    Deliberately memoryless about motion - a track's predicted position for the next frame is
    simply where it was last seen. The task sheet asks for IoU association and nothing more, and
    adding a motion model would make it impossible to attribute the results to the prescribed
    method.
    """

    def __init__(
        self,
        iou_thresh: float = config.TRACK_IOU_THRESH,
        max_age: int | None = None,
        min_hits: int | None = None,
        match: str = config.TRACK_MATCH,
        fps: float = 30.0,
    ):
        """`max_age` and `min_hits` default to `config`'s seconds-based values scaled by `fps`.

        Pass them explicitly (in frames) to override - the unit tests do, so that each test
        states the threshold it asserts against instead of inheriting a global.
        """
        if match not in MATCHERS:
            raise ValueError(f"unknown matching strategy {match!r}; choose from {list(MATCHERS)}")
        default_age, default_hits = config.track_frames(fps)
        self.iou_thresh = iou_thresh
        self.max_age = default_age if max_age is None else max_age
        self.min_hits = default_hits if min_hits is None else min_hits
        self.match_fn = MATCHERS[match]
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: np.ndarray) -> list[Track]:
        """Advance the tracker by one frame.

        Args:
            detections: (N, 4) array of detection boxes for the current frame.

        Returns:
            The list of tracks that are alive after this frame.
        """
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 4)

        for t in self.tracks:
            t.age += 1

        track_boxes = np.array([t.box for t in self.tracks], dtype=np.float32).reshape(-1, 4)
        iou = iou_matrix(track_boxes, detections)
        pairs = self.match_fn(iou, self.iou_thresh)

        matched_t = {i for i, _ in pairs}
        matched_d = {j for _, j in pairs}

        for i, j in pairs:
            self.tracks[i].update(tuple(detections[j]))

        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.mark_missed()

        # Every detection that matched nothing is, by definition, a vehicle we have not seen
        # before -> it gets a fresh identity.
        for j in range(len(detections)):
            if j not in matched_d:
                self.tracks.append(self._new_track(tuple(detections[j])))

        # Terminate stale tracks. max_age > 1 is what lets a track survive a brief occlusion
        # (a car passing behind a van) instead of being torn down and re-created with a new ID,
        # which would double-count it.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.tracks

    def _new_track(self, box: Box) -> Track:
        track = Track(track_id=self._next_id, box=box, min_hits=self.min_hits)
        track.trail.append(track.center)
        self._next_id += 1
        return track
