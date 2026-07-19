"""Part 2 counting: a virtual line, crossed once per track, tallied per direction.

The task sheet: "define a virtual counting line and count a track exactly once, at the moment
its box center crosses the line."

Three design points, all of which exist to stop the same class of error - counting one vehicle
more than once:

1. **Crossing, not presence.** We do not count tracks that are *on* one side of the line; we
   count the *event* of the center moving from one side to the other. That is tested as a proper
   segment-segment intersection between the line and the short segment the center travelled
   since the previous frame. Testing "which side is it on now" would re-count a vehicle on every
   subsequent frame.

2. **Once per identity.** A `counted` flag on the track. A vehicle stopped exactly on the line in
   queued traffic will jitter across it repeatedly as the box wobbles; without the flag every
   wobble is another vehicle.

3. **Confirmed tracks only.** A track has to survive `TRACK_MIN_HITS` frames before it may be
   counted, so a one-frame detector false positive cannot become a phantom vehicle.

Direction is taken from the *sign* of the crossing rather than from "up/down", so the line does
not have to be horizontal - it works for a diagonal line across an angled road. The two signs are
mapped to human-readable names by `config.DIRECTION_LABELS`.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from src.part2.tracker import Track

Point = tuple[float, float]
Line = tuple[Point, Point]


@dataclass
class Crossing:
    """One counted event: track `track_id` crossed the line on `frame` going `direction`."""

    track_id: int
    frame: int
    direction: int  # +1 or -1; see config.DIRECTION_LABELS
    point: Point


def _side(line: Line, point: Point) -> float:
    """Signed area of the triangle (line[0], line[1], point).

    The sign says which side of the (infinite) line the point lies on; zero means exactly on it.
    This is the 2-D cross product, and it is the whole reason the counter is orientation
    agnostic - no special casing for horizontal, vertical or diagonal lines.
    """
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)


def segment_crossing(line: Line, start: Point, end: Point) -> int:
    """Does the movement `start -> end` cross the line *segment*, and in which direction?

    Returns +1 or -1 for a crossing, 0 for none.

    Both tests are needed. The first (the endpoints of the movement lie on opposite sides of the
    line) alone would also fire for a vehicle crossing the line's *infinite extension* far off to
    the side - e.g. traffic on a side road, when the counting line only spans one carriageway.
    The second test (the endpoints of the line lie on opposite sides of the movement) restricts
    it to the drawn segment.

    Note the `>= 0` comparisons. A center landing *exactly* on the line gives `_side == 0`, and
    treating that as a third state ("neither side") loses the crossing entirely: the vehicle
    steps onto the line in one frame and off it in the next, and neither step ever registers two
    different sides. Folding zero into the positive side makes "which side" a genuine boolean, so
    every crossing is caught exactly once. This is not hypothetical - box centers are computed
    from integer pixel coordinates and land on a round y value regularly, and the unit tests
    caught it on the very first synthetic vehicle.
    """
    d_start = _side(line, start)
    d_end = _side(line, end)
    if (d_start >= 0) == (d_end >= 0):
        return 0

    movement: Line = (start, end)
    d_a = _side(movement, line[0])
    d_b = _side(movement, line[1])
    if (d_a >= 0) == (d_b >= 0):
        return 0

    return 1 if d_end >= 0 else -1


def line_to_pixels(line_norm: Line, width: int, height: int) -> Line:
    """Convert a line given in normalised [0, 1] coordinates to pixels.

    The line lives in `config.COUNT_LINE` in normalised form so that it is resolution
    independent: the same configuration is valid whether we run on the 1080p or the 4K version
    of a clip, and re-encoding the video at another size cannot silently move the line.
    """
    (x1, y1), (x2, y2) = line_norm
    return ((x1 * width, y1 * height), (x2 * width, y2 * height))


class LineCounter:
    """Tallies confirmed tracks crossing a virtual line, separately per direction."""

    def __init__(self, line: Line, labels: dict[int, str] | None = None):
        self.line = line
        self.labels = labels or config.DIRECTION_LABELS
        self.counts: dict[int, int] = {1: 0, -1: 0}
        self.crossings: list[Crossing] = []

    def update(self, tracks: list[Track], frame_idx: int) -> list[Crossing]:
        """Check every live track for a line crossing this frame. Returns the new crossings."""
        new: list[Crossing] = []
        for track in tracks:
            # `trail` only grows on a matched update, so consecutive entries are the centers of
            # the two most recent frames in which this vehicle was actually detected.
            if track.counted or not track.confirmed or len(track.trail) < 2:
                continue

            direction = segment_crossing(self.line, track.trail[-2], track.trail[-1])
            if direction == 0:
                continue

            track.counted = True
            self.counts[direction] += 1
            crossing = Crossing(track.track_id, frame_idx, direction, track.trail[-1])
            self.crossings.append(crossing)
            new.append(crossing)
        return new

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> dict[str, int]:
        """Counts keyed by human-readable direction name, plus the total."""
        out = {self.labels[d]: n for d, n in self.counts.items()}
        out["total"] = self.total
        return out
