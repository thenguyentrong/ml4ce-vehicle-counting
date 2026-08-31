"""Part 2 end to end: detect every frame, track, count line crossings, render the output video.

    python -m src.part2.run_count --weights runs/yolo/finetune/weights/best.pt
    python -m src.part2.run_count --weights stock --match greedy --tag stock_greedy

This is the script that produces the deliverable the task sheet asks for: "an output video that
shows the detected boxes, the track IDs, the counting line and the running count", plus the
per-direction totals to compare against the manual count.

Structure, one frame at a time:

    frame -> YOLO detections -> IoUTracker.update() -> LineCounter.update() -> annotated frame

The three stages are deliberately separate modules. The detector is replaceable (stock vs
fine-tuned) without touching the tracker; the tracker's matching strategy is replaceable without
touching the counter; and `tracker.py` and `counter.py` contain no ultralytics or OpenCV
dependency at all, which is what makes them unit-testable on synthetic boxes in `tests/`.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import config
from src.part2.counter import LineCounter, line_to_pixels
from src.part2.tracker import IoUTracker, Track

# Drawn in BGR, because that is the byte order OpenCV uses.
COLOR_LINE = (0, 0, 255)
COLOR_BOX = (0, 200, 0)
COLOR_COUNTED = (0, 160, 255)  # orange; (255, 160, 0) was RGB and rendered blue
COLOR_TEXT = (255, 255, 255)


def detect(model: YOLO, frame: np.ndarray, conf: float, stock: bool) -> np.ndarray:
    """Run the detector on one frame and return an (N, 4) array of vehicle boxes.

    The stock COCO model predicts 80 classes, so its output is filtered down to the four that
    are vehicles and merged into one. The fine-tuned model already predicts a single `vehicle`
    class, so everything it returns is kept.
    """
    result = model(frame, conf=conf, verbose=False)[0]
    boxes = []
    for box in result.boxes:
        if stock and int(box.cls) not in config.COCO_VEHICLE_CLASSES:
            continue
        boxes.append([float(v) for v in box.xyxy[0]])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def draw(
    frame: np.ndarray,
    tracks: list[Track],
    counter: LineCounter,
    frame_idx: int,
) -> np.ndarray:
    """Overlay the counting line, the tracked boxes with their IDs, and the running count."""
    (x1, y1), (x2, y2) = counter.line
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_LINE, 3)

    for track in tracks:
        # An unconfirmed track is not drawn: it is not yet trusted enough to be counted, and
        # drawing it would suggest the counter is about to act on it.
        if not track.confirmed:
            continue
        bx1, by1, bx2, by2 = (int(v) for v in track.box)
        color = COLOR_COUNTED if track.counted else COLOR_BOX
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(
            frame, f"#{track.track_id}", (bx1, by1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    summary = counter.summary()
    lines = [f"frame {frame_idx}"] + [
        f"{name}: {n}" for name, n in summary.items() if name != "total"
    ] + [f"TOTAL: {summary['total']}"]

    # Dark panel behind the text so it stays readable over both bright road and dark vehicles.
    cv2.rectangle(frame, (10, 10), (330, 20 + 32 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(
            frame, text, (20, 42 + 32 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2,
        )
    return frame


def run(
    weights: str,
    tag: str,
    match: str = config.TRACK_MATCH,
    conf: float = config.YOLO_CONF,
    seconds: float | None = None,
    video_path: Path | None = None,
    line: tuple | None = None,
    save_video: bool = True,
) -> dict:
    """Detect, track and count over the video; write the annotated video and a JSON summary."""
    video_path = video_path or config.VIDEO_PATH
    if not video_path.exists():
        raise FileNotFoundError(
            f"{video_path} not found - run `python -m src.part2.video` first"
        )

    stock = weights == "stock"
    model = YOLO(config.YOLO_MODEL if stock else weights)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = config.frames_to_process(video_path, fps, total_frames, seconds)
    if n_frames < total_frames:
        print(f"[run_count] reading the first {n_frames / fps:.0f} s ({n_frames} of "
              f"{total_frames} frames); vehicles crossing later are NOT counted")

    # The tracker's temporal thresholds come from config in SECONDS and are scaled by this
    # video's own frame rate, so the same settings behave identically on unseen footage shot at
    # a different frame rate.
    tracker = IoUTracker(match=match, fps=fps)
    counter = LineCounter(line_to_pixels(line or config.COUNT_LINE, width, height))

    out_dir = config.RUNS_DIR / "part2" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = (
        cv2.VideoWriter(str(out_dir / "counted.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (width, height))
        if save_video
        else None
    )

    n_detections = 0
    try:
        for frame_idx in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break

            detections = detect(model, frame, conf, stock)
            n_detections += len(detections)
            tracks = tracker.update(detections)
            counter.update(tracks, frame_idx)

            if writer is not None:
                writer.write(draw(frame, tracks, counter, frame_idx))

            if frame_idx % 200 == 0:
                print(f"[run_count] frame {frame_idx}/{n_frames}  count={counter.summary()}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    summary = {
        "tag": tag,
        "weights": weights,
        "match": match,
        "conf": conf,
        "video": video_path.name,
        "fps": round(fps, 2),
        "line": line or config.COUNT_LINE,
        "max_age_frames": tracker.max_age,
        "min_hits_frames": tracker.min_hits,
        "frames": n_frames,
        "frames_available": total_frames,
        "detections_total": n_detections,
        "detections_per_frame": round(n_detections / max(1, n_frames), 2),
        "tracks_created": tracker._next_id - 1,
        "counts": counter.summary(),
        "crossings": [
            {"id": c.track_id, "frame": c.frame, "direction": counter.labels[c.direction]}
            for c in counter.crossings
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[run_count] {tag}: {summary['counts']}")
    print(f"[run_count] tracks created {summary['tracks_created']} for "
          f"{summary['counts']['total']} counted vehicles")
    print(f"[run_count] wrote {out_dir}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Detect, track and count vehicles in the video")
    p.add_argument("--weights", default=None,
                   help="path to fine-tuned weights, or 'stock' for un-fine-tuned yolo11n; "
                        "default: the fine-tuned weights if present, else 'stock'")
    p.add_argument("--tag", default=None, help="run name -> runs/part2/<tag>/")
    p.add_argument("--match", default=config.TRACK_MATCH, choices=["hungarian", "greedy"])
    p.add_argument("--conf", type=float, default=config.YOLO_CONF)
    p.add_argument("--seconds", type=float, default=None,
                   help=f"process only the first N seconds; defaults to the whole video, except "
                        f"for our own clip, which is trimmed to {config.VIDEO_SECONDS} s to match "
                        f"the manual count")
    p.add_argument("--video", type=Path, default=None, help="run on a different video")
    p.add_argument(
        "--line", default=None, metavar="X1,Y1,X2,Y2",
        help="counting line in normalised coords, e.g. 0,0.65,1,0.65. Overrides config.COUNT_LINE; "
             "use `python -m src.part2.suggest_line` to place it on an unseen video.",
    )
    p.add_argument("--no-video", action="store_true", help="compute counts without rendering")
    args = p.parse_args()

    line = None
    if args.line:
        try:
            x1, y1, x2, y2 = (float(v) for v in args.line.split(","))
        except ValueError:
            raise SystemExit(f"--line must be four comma-separated numbers, got {args.line!r}")
        line = ((x1, y1), (x2, y2))

    weights = args.weights or config.default_weights()
    if args.weights is None:
        print(f"[run_count] no --weights given; using {weights}")

    tag = args.tag or f"{'stock' if weights == 'stock' else 'finetune'}_{args.match}"
    run(
        weights=weights,
        tag=tag,
        match=args.match,
        conf=args.conf,
        seconds=args.seconds,
        video_path=args.video,
        line=line,
        save_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
