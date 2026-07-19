"""Render the reference clip used to produce the manual ground-truth count.

    python -m src.part2.manual_count

The task sheet asks us to "count the vehicles in the video manually once" and compare per
direction. The trap is that "the number of vehicles in the video" is not well defined, and if the
human and the machine answer *different questions* the comparison measures nothing:

  * Do parked cars at the kerb count? They are vehicles, and they are in the video.
  * Does a vehicle that appears at the top of the frame and turns off before reaching the line?
  * Does a vehicle already past the line in frame 0?

The automatic counter answers exactly one question - *how many tracks had their box centre cross
this line, and in which direction* - so the manual count has to answer the same one. This module
therefore writes the video with the counting line burned in and nothing else, so it can be
watched frame by frame and tallied against precisely that definition. The three cases above all
resolve to "no": no crossing, no count.

Record the tally in `docs/manual_count.md`, then `src.part2.evaluate` reads it back.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import argparse

import cv2

import config
from src.part2.counter import line_to_pixels


def render(seconds: float = config.VIDEO_SECONDS) -> None:
    """Write runs/part2/manual/reference.mp4: the clip plus the counting line and a frame index."""
    if not config.VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"{config.VIDEO_PATH} not found - run `python -m src.part2.video` first"
        )

    cap = cv2.VideoCapture(str(config.VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(seconds * fps))

    (x1, y1), (x2, y2) = line_to_pixels(config.COUNT_LINE, width, height)

    out_dir = config.RUNS_DIR / "part2" / "manual"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_dir / "reference.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    try:
        for frame_idx in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            # The frame index makes the tally auditable: a disputed vehicle can be found again.
            cv2.putText(
                frame, f"frame {frame_idx}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    print(f"[manual_count] wrote {out_dir / 'reference.mp4'} ({n_frames} frames)")
    print("[manual_count] count every vehicle whose CENTRE crosses the red line, per direction,")
    print("[manual_count] then record the totals in docs/manual_count.md")


def main() -> None:
    p = argparse.ArgumentParser(description="Render the clip used for the manual count")
    p.add_argument("--seconds", type=float, default=config.VIDEO_SECONDS)
    args = p.parse_args()
    render(seconds=args.seconds)


if __name__ == "__main__":
    main()
