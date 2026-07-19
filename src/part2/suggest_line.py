"""Propose a counting line for a video we have never seen.

    python -m src.part2.suggest_line --video path/to/unseen.mp4 --weights runs/yolo/finetune/weights/best.pt

The counting line is the one part of this pipeline that cannot be video independent: the task is
*defined* as "count what crosses this line", and where the line belongs depends on where the road
is. Everything else in `config.py` is either dimensionless or expressed in seconds, but the line
has to be placed per video.

That is a robustness problem, because the course tests the submitted model on a separate video.
Placing the line by eye does not survive that - we tried it on our own clip and got it wrong: we
drew a horizontal line across what looked like the direction of travel, and the traffic's dominant
flow turned out to be along the other axis entirely. Only 0 of 34 moving vehicles crossed it in one
direction.

So the line is placed by *measurement* instead. This module runs the detector and tracker over the
video, collects the path of every moving track, then sweeps candidate horizontal and vertical lines
and scores each one by:

  * **coverage**  - how many moving tracks cross it (more is better: a line nobody crosses counts
    nothing), and
  * **balance**   - how evenly the crossings split between the two directions (the task asks for
    counts *per direction*, so a line that only ever sees one-way traffic is close to useless).

It prints the ranked candidates and the `COUNT_LINE` tuple to paste into `config.py`, rather than
editing anything itself - the choice of line is a modelling decision and should be a visible,
deliberate line of configuration, not a hidden side effect.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import config
from src.part2.counter import segment_crossing
from src.part2.tracker import IoUTracker

# A track has to actually travel before its path says anything about traffic flow; a parked car's
# "path" is a jittering dot and would bias every candidate line it happens to sit on.
MIN_TRAVEL_PX = 80
MIN_PATH_POINTS = 5


def collect_paths(video: Path, weights: str, seconds: float, conf: float) -> tuple[dict, int, int]:
    """Track the video and return {track_id: [centers]} plus the frame size."""
    from ultralytics import YOLO

    model = YOLO(config.YOLO_MODEL if weights == "stock" else weights)
    stock = weights == "stock"

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(seconds * fps))

    tracker = IoUTracker(match=config.TRACK_MATCH, fps=fps)
    paths: dict[int, list[tuple[float, float]]] = {}
    try:
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            result = model(frame, conf=conf, verbose=False)[0]
            boxes = [
                [float(v) for v in b.xyxy[0]]
                for b in result.boxes
                if not stock or int(b.cls) in config.COCO_VEHICLE_CLASSES
            ]
            dets = np.array(boxes, dtype=np.float32).reshape(-1, 4)
            for track in tracker.update(dets):
                if track.confirmed:
                    paths.setdefault(track.track_id, []).append(track.center)
    finally:
        cap.release()

    return paths, width, height


def moving_only(paths: dict) -> dict:
    """Discard tracks that never went anywhere - parked cars, and detector flicker."""
    out = {}
    for tid, pts in paths.items():
        if len(pts) < MIN_PATH_POINTS:
            continue
        travel = np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        if travel >= MIN_TRAVEL_PX:
            out[tid] = pts
    return out


def dominant_flow_axis(moving: dict) -> np.ndarray:
    """The axis along which traffic predominantly travels, as a unit vector.

    Computed as the principal eigenvector of the orientation tensor sum(v v^T) over the unit net
    displacement v of every moving track. The outer product is what makes this an *axis* rather
    than a direction: v and -v contribute identically, so the two opposing streams of a two-way
    road reinforce the same axis instead of cancelling to zero, which is exactly what a plain
    mean of the displacement vectors would do.
    """
    tensor = np.zeros((2, 2))
    for pts in moving.values():
        v = np.array([pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]], dtype=float)
        norm = float(np.hypot(*v))
        if norm < 1e-6:
            continue
        v /= norm
        tensor += np.outer(v, v)

    _, eigenvectors = np.linalg.eigh(tensor)
    return eigenvectors[:, -1]  # eigenvector of the largest eigenvalue


def alignment(flow_axis: np.ndarray, orientation: str) -> float:
    """How well a line of this orientation is *crossed* by the flow, in [0, 1].

    A counting line reads direction from the component of motion along its **normal** - the
    component that carries a vehicle from one side to the other. If the flow runs almost parallel
    to the line's normal (alignment near 1) direction is read from the dominant part of the
    motion and is robust. If the flow is nearly parallel to the *line* (alignment near 0),
    vehicles skim across it and direction is decided by the small residual component, where a few
    pixels of box jitter can flip the answer.

    This is not hypothetical: our first counting line on this project was horizontal, drawn by eye
    across what looked like the direction of travel. The traffic turned out to run along a
    diagonal corridor with roughly 6:1 lateral motion, so the direction of every count was being
    decided by the minority 1/6th of each vehicle's displacement.
    """
    normal = np.array([0.0, 1.0]) if orientation == "horizontal" else np.array([1.0, 0.0])
    return float(abs(np.dot(flow_axis, normal)))


def score_line(moving: dict, line) -> tuple[int, int]:
    """(crossings one way, crossings the other) for this candidate line."""
    pos = neg = 0
    for pts in moving.values():
        for a, b in zip(pts, pts[1:]):
            d = segment_crossing(line, a, b)
            if d == 1:
                pos += 1
                break
            if d == -1:
                neg += 1
                break
    return pos, neg


def suggest(moving: dict, width: int, height: int, top: int = 5,
            coverage_tolerance: float = 0.85) -> list[dict]:
    """Sweep horizontal and vertical candidates; return them ranked best first.

    Ranking is deliberately two-stage rather than a weighted sum of three quantities, because
    weights invented to make one video come out right are exactly what does not transfer to the
    unseen video the course tests on:

      1. **Coverage is a gate, not a score.** A line nobody crosses counts nothing, so only
         candidates within `coverage_tolerance` of the best coverage stay in the running.
      2. **Among those, alignment decides**, then balance. Two lines that catch nearly the same
         vehicles are not equally good: the one the traffic crosses head-on reads direction from
         the dominant component of motion, the one traffic skims reads it from the residual.

    The failure this ordering prevents is the one we actually hit: a horizontal line scored the
    single best coverage (42 crossings) on our clip and was chosen, while a vertical line catching
    41 read direction from six times more signal.
    """
    candidates = []
    for frac in np.arange(0.30, 0.90, 0.05):
        candidates.append(("horizontal", frac, ((0.0, frac), (1.0, frac))))
    for frac in np.arange(0.20, 0.85, 0.05):
        candidates.append(("vertical", frac, ((frac, 0.25), (frac, 1.0))))

    flow_axis = dominant_flow_axis(moving)

    scored = []
    for orientation, frac, norm in candidates:
        pixels = (
            (norm[0][0] * width, norm[0][1] * height),
            (norm[1][0] * width, norm[1][1] * height),
        )
        pos, neg = score_line(moving, pixels)
        total = pos + neg
        balance = min(pos, neg) / max(pos, neg) if max(pos, neg) else 0.0
        scored.append(
            {
                "orientation": orientation,
                "frac": round(float(frac), 2),
                # float() is not cosmetic: np.arange yields np.float64, and the repr of those is
                # `np.float64(0.65)`, which is what would get printed for pasting into config.py
                # and would not evaluate there without numpy imported.
                "line": ((round(float(norm[0][0]), 2), round(float(norm[0][1]), 2)),
                         (round(float(norm[1][0]), 2), round(float(norm[1][1]), 2))),
                "pos": pos,
                "neg": neg,
                "total": total,
                "balance": round(balance, 2),
                "alignment": round(alignment(flow_axis, orientation), 2),
            }
        )

    best_coverage = max(s["total"] for s in scored) if scored else 0
    gate = coverage_tolerance * best_coverage
    eligible = [s for s in scored if s["total"] >= gate] or scored
    eligible.sort(key=lambda s: (s["alignment"], s["balance"], s["total"]), reverse=True)

    # Everything else, still ranked by coverage, so the printed table shows what was rejected.
    rest = sorted(
        (s for s in scored if s not in eligible),
        key=lambda s: (s["total"], s["balance"]),
        reverse=True,
    )
    return (eligible + rest)[:top]


def main() -> None:
    p = argparse.ArgumentParser(description="Propose a counting line for an unseen video")
    p.add_argument("--video", type=Path, default=config.VIDEO_PATH)
    p.add_argument("--weights", default="stock",
                   help="fine-tuned weights path, or 'stock' for un-fine-tuned yolo11n")
    p.add_argument("--seconds", type=float, default=config.VIDEO_SECONDS)
    p.add_argument("--conf", type=float, default=config.YOLO_CONF)
    args = p.parse_args()

    if not args.video.exists():
        raise SystemExit(f"{args.video} not found")

    paths, width, height = collect_paths(args.video, args.weights, args.seconds, args.conf)
    moving = moving_only(paths)
    print(f"[suggest_line] {args.video.name}: {len(paths)} tracks, {len(moving)} of them moving\n")
    if not moving:
        raise SystemExit("no moving vehicles found - check the weights, --conf, or the video")

    axis = dominant_flow_axis(moving)
    print(f"[suggest_line] dominant flow axis: ({axis[0]:+.2f}, {axis[1]:+.2f})"
          f"  - traffic runs mostly {'horizontally' if abs(axis[0]) > abs(axis[1]) else 'vertically'}"
          f" across the frame\n")

    ranked = suggest(moving, width, height)
    print(f"{'orientation':12s}{'pos':>6s}{'neg':>6s}{'total':>7s}{'balance':>9s}{'align':>7s}   line")
    for s in ranked:
        print(f"{s['orientation']:12s}{s['pos']:>6d}{s['neg']:>6d}{s['total']:>7d}"
              f"{s['balance']:>9.2f}{s['alignment']:>7.2f}   {s['line']}")

    best = ranked[0]
    print(f"\n[suggest_line] {best['total']} of {len(moving)} moving vehicles cross the best line "
          f"({best['pos']}/{best['neg']} per direction), alignment {best['alignment']:.2f}.")
    if best["alignment"] < 0.5:
        print("[suggest_line] NOTE: even the best candidate is skimmed rather than crossed head-on.")
        print("[suggest_line] Direction is being read from the minority component of motion; treat")
        print("[suggest_line] the per-direction split as less reliable than the total.")
    print("[suggest_line] paste into config.py:\n")
    print(f"    COUNT_LINE = {best['line']}")
    if best["balance"] < 0.15:
        print("\n[suggest_line] WARNING: the two directions are very unbalanced. Either this video")
        print("[suggest_line] has essentially one-way traffic, or the flows disperse and no single")
        print("[suggest_line] line captures both - check the video before trusting per-direction counts.")


if __name__ == "__main__":
    main()
