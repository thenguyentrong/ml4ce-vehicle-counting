"""Acquire the Part 2 traffic video.

Run `python -m src.part2.video` to fetch it and print what arrived.

**Why this file exists.** Part 2 needs a static traffic video and the course did not supply one
with the topic zip, so we sourced it ourselves. Like the Kaggle images it is *not* committed
(`data/` is gitignored) - the repository stays small and carries a script that reproduces the
data instead of the data itself.

**Provenance and licence.**
    "Traffic Flow in an Intersection", by German Korb, on Pexels
    https://www.pexels.com/video/traffic-flow-in-an-intersection-4791734/
    Pexels Licence: free to use and modify, attribution not required, no redistribution on
    other stock-media platforms. Using it as evaluation data in a coursework repository is
    within the licence. We credit the author anyway.
    1920x1080, 29.97 fps, 64.1 s.

**Why this clip and not another.** Four properties had to hold at once, and each eliminated most
candidates:

  * *Static camera.* A virtual counting line is meaningless if the camera moves, and most
    "traffic" stock footage is dashcam or drone.
  * *Two directions through one cross-section.* The task sheet asks for counts **per
    direction**, which needs more than "both directions are visible somewhere in frame": there
    has to exist a single line that both flows actually cross.
  * *Moderate, free-flowing density.* The ground truth is counted **by hand**, once. Congested
    multi-lane footage is both unreliable to hand-count (vehicles occlude each other) and the
    worst case for IoU association. Measured with stock yolo11n, the rejected congested
    candidates ran 16-19 detections/frame against ~9 here.
  * *Daylight.* Not because dusk broke the detector - we measured that and it did not - but
    because headlight glare makes the *manual* count unreliable, and the manual count is the
    only ground truth we have.

The second property is the one that is easy to get wrong by eye, and it eliminated the clip we
first chose. A T-junction scene looked ideal on stills - static, daylight, two-way, ~9
detections/frame - but tracking it showed the traffic *disperses*: vehicles turn off in
different directions, and the best possible line was crossed by only 15 of 34 moving tracks,
14 of them in the same direction. This clip's best line is crossed by 42 of 93, split 31/11.
Both numbers were measured, not estimated; see docs/experiments.md.

Only the first `config.VIDEO_SECONDS` seconds are used. The trim is applied when reading frames
rather than by re-encoding the file, so the pipeline needs no ffmpeg installed.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2

import config


def download(dest: Path | None = None, force: bool = False) -> Path:
    """Download the traffic video to `config.VIDEO_PATH`. No-op if it is already there."""
    dest = dest or config.VIDEO_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"[video] already present: {dest}")
        return dest

    print(f"[video] downloading {config.VIDEO_URL}")
    # Pexels' CDN rejects the default urllib user agent with 403.
    request = urllib.request.Request(config.VIDEO_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(dest, "wb") as fh:
        fh.write(response.read())

    print(f"[video] wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def probe(path: Path | None = None) -> dict[str, float]:
    """Resolution, frame rate and length of the video, read back from the file itself."""
    path = path or config.VIDEO_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.part2.video` first")

    cap = cv2.VideoCapture(str(path))
    try:
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()

    info["seconds"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    return info


def main() -> None:
    path = download()
    info = probe(path)
    used = min(config.VIDEO_SECONDS, info["seconds"])
    print(f"resolution : {info['width']}x{info['height']}")
    print(f"frame rate : {info['fps']:.2f} fps")
    print(f"length     : {info['seconds']:.1f} s ({info['frames']} frames)")
    print(f"used       : first {used:.0f} s ({int(used * info['fps'])} frames)")


if __name__ == "__main__":
    main()
