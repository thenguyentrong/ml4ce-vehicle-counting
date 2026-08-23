"""Pack everything the course wants into one zip for the gigamove upload.

    python make_submission.py                 # -> submission/ML4CE_Topic2_<names>.zip
    python make_submission.py --lean          # without the mp4s
    python make_submission.py --list          # show what goes in, write nothing

The easiest things to forget are the ones the graders cannot rebuild: the **trained weights**
(git ignores `*.pt`) and the **output video** the task sheet asks for. Both are in. The Kaggle
images are not — `python -m src.data` downloads them again.

The 49 MB ResNet18 checkpoints of the ablation runs stay out; their `metrics.json` and curves are
what docs/experiments.md cites. The two checkpoints that do ship are the ones the inference
entry points load by default.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import config

BUNDLE_NAME = "ML4CE_Topic2_Nguyen_Awada_Azemi"

# Compressing an mp4 or a checkpoint gains ~nothing and costs minutes, so those are stored.
STORE_SUFFIXES = {".mp4", ".pt", ".jpg", ".jpeg", ".png"}

# (source, description) - directories are added recursively, minus EXCLUDE_PARTS.
CODE = [
    ("config.py", "all paths and hyperparameters"),
    ("README.md", "how to run everything, and the results"),
    ("NOTES.md", "lab notebook: what we tried, what worked, what did not"),
    ("pyproject.toml", "dependencies"),
    ("uv.lock", "exact dependency versions"),
    ("make_submission.py", "this script"),
    ("src", "all source code"),
    ("tests", "unit tests: tracker, counter, grid encoding, manual-count parsing"),
    ("ali_contribution", "Ali Awada's own implementation of both parts, and his report"),
    ("docs", "task spec, course info, experiments, manual count, unseen-video procedure"),
]

# Weights and result artifacts. Kept explicit: a glob over runs/ would sweep in 500 MB of
# ablation checkpoints.
ARTIFACTS = [
    ("runs/mobilenet_multi_unfreeze", "Part 1 best detector: weights, metrics, figures"),
    ("runs/yolo/finetune/weights/best.pt", "Part 2 fine-tuned YOLO11n weights"),
    ("runs/yolo/finetune/results.png", "Part 2 fine-tuning curves"),
    ("runs/yolo/finetune/confusion_matrix.png", "Part 2 confusion matrix"),
    ("runs/yolo/finetune/BoxPR_curve.png", "Part 2 precision-recall curve"),
    ("runs/yolo/finetune/args.yaml", "Part 2 exact training arguments"),
    # Shipped so `--weights stock` reproduces the off-the-shelf comparison without ultralytics
    # silently downloading a possibly newer checkpoint from the internet.
    (config.YOLO_MODEL, "off-the-shelf COCO yolo11n, for the --weights stock comparison"),
]

# Everything else in runs/ that is small enough to ship wholesale: the per-run metrics and
# curves the experiments table cites, without the checkpoints.
ABLATION_PATTERNS = ["runs/*/metrics.json", "runs/*/history.json", "runs/*/curves.png",
                     "runs/*/pr_curve.png", "runs/part2/*/summary.json"]

MEDIA = [
    ("runs/part2/finetune_hungarian/counted.mp4", "the counted output video (task deliverable)"),
    ("data/traffic.mp4", "the source clip, so Part 2 runs without re-downloading"),
]

PRESENTATION_DIR = "presentation"

EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def collect(root: Path, lean: bool) -> tuple[list[tuple[Path, str]], list[str]]:
    """Return [(absolute path, path inside the zip)] plus a list of missing-item warnings."""
    files: list[tuple[Path, str]] = []
    warnings: list[str] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not path.is_file() or any(part in EXCLUDE_PARTS for part in path.parts):
            return
        arc = path.relative_to(root).as_posix()
        if arc not in seen:
            seen.add(arc)
            files.append((path, arc))

    groups = CODE + ARTIFACTS + ([] if lean else MEDIA)
    for rel, _ in groups:
        target = root / rel
        if target.is_dir():
            before = len(files)
            for path in sorted(target.rglob("*")):
                add(path)
            if len(files) == before:
                warnings.append(f"empty, nothing to ship: {rel}")
        elif target.is_file():
            add(target)
        else:
            warnings.append(f"missing: {rel}")

    for pattern in ABLATION_PATTERNS:
        for path in sorted(root.glob(pattern)):
            add(path)

    presentation = root / PRESENTATION_DIR
    if presentation.is_dir() and any(presentation.iterdir()):
        for path in sorted(presentation.rglob("*")):
            add(path)
    else:
        warnings.append(
            f"NO PRESENTATION: put the slides in {PRESENTATION_DIR}/ - the course requires code "
            f"AND presentation in the same submission"
        )

    return files, warnings


def human(n_bytes: int) -> str:
    mb = n_bytes / 1024**2
    return f"{mb:7.1f} MB" if mb >= 1 else f"{n_bytes / 1024:7.1f} kB"


def main() -> None:
    p = argparse.ArgumentParser(description="Build the submission zip")
    p.add_argument("--lean", action="store_true", help="leave out the mp4s (~160 MB)")
    p.add_argument("--list", action="store_true", help="show the contents, write nothing")
    p.add_argument("--out", type=Path, default=None, help="output zip path")
    args = p.parse_args()

    root = config.PROJECT_ROOT
    files, warnings = collect(root, args.lean)
    total = sum(path.stat().st_size for path, _ in files)

    # Grouped by top-level folder, because a 400-line file list is not a check anyone performs.
    groups: dict[str, list[int]] = {}
    for path, arc in files:
        key = arc.split("/")[0] if "/" in arc else "(root files)"
        groups.setdefault(key, []).append(path.stat().st_size)

    print(f"{BUNDLE_NAME}: {len(files)} files, {human(total)}\n")
    for key in sorted(groups):
        sizes = groups[key]
        print(f"  {key:<28} {len(sizes):>4} file(s)  {human(sum(sizes))}")

    for warning in warnings:
        print(f"\n  !! {warning}")

    if args.list:
        print("\n(--list: nothing written)")
        return

    out_path = args.out or (root / "submission" / f"{BUNDLE_NAME}.zip")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path, arc in files:
            compress = (
                zipfile.ZIP_STORED if path.suffix.lower() in STORE_SUFFIXES else zipfile.ZIP_DEFLATED
            )
            zf.write(path, f"{BUNDLE_NAME}/{arc}", compress_type=compress)

    print(f"\nwrote {out_path}  ({human(out_path.stat().st_size)})")
    print("upload to https://gigamove.rwth-aachen.de/en and mail the link to "
          "ml4cegia@lists.rwth-aachen.de")


if __name__ == "__main__":
    main()
