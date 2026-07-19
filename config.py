"""Central configuration: every path and hyperparameter used by the project lives here.

Nothing else in the codebase hard-codes a path, so the project stays runnable on another
machine (and on the graders' held-out test set) by editing this file alone.

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

# kagglehub caches downloads here instead of in the user profile, so the whole project
# (code + data + weights) stays inside one self-contained folder.
KAGGLEHUB_CACHE = DATA_DIR / "kagglehub"
os.environ.setdefault("KAGGLEHUB_CACHE", str(KAGGLEHUB_CACHE))

KAGGLE_DATASET = "sshikamaru/car-object-detection"

# Filled in by src/data.py after the download; see resolve_dataset_paths().
DATASET_SUBDIR = "data"  # the Kaggle archive nests everything under a "data/" folder

# Part 2: the traffic video. The course did not supply one, so we sourced it ourselves. It is
# not committed (data/ is gitignored, exactly like the Kaggle images) but is reproducible with
# `python -m src.part2.video`; provenance and licence are documented in that module.
VIDEO_PATH = DATA_DIR / "traffic.mp4"
VIDEO_URL = "https://videos.pexels.com/video-files/4791734/4791734-hd_1920_1080_30fps.mp4"
VIDEO_SECONDS = 60  # source clip is 64 s; 60 s keeps the manual ground-truth count tractable

# --------------------------------------------------------------------------------------
# Part 1 - detection head on a frozen backbone
# --------------------------------------------------------------------------------------

# Exactly as the task sheet prescribes: "with an input size of 512x512 pixels and a stride-32
# feature map you obtain a 16x16 grid". We tried larger inputs (640, 768) and a finer stride-16
# grid; none of them found a single additional vehicle. See NOTES.md.
IMG_SIZE = 512  # network input is IMG_SIZE x IMG_SIZE
STRIDE = 32  # backbone output stride
GRID = IMG_SIZE // STRIDE  # -> 16 x 16 grid, as prescribed by the task sheet

# The defaults in this file are the configuration the TASK SHEET prescribes, so a plain
# `python -m src.part1.train` reproduces the specified detector. Every deviation we tested is
# an explicit CLI flag and is reported in docs/experiments.md - the two that actually helped
# are `--assign multi` and `--unfreeze` (see below).
BACKBONE = "resnet18"  # "resnet18" | "mobilenet_v3_large"  (ablation)

# Task sheet: "You don't need to train the backbone, only train your detection head."
# We keep that as the default. Note `--unfreeze` (fine-tuning the backbone) was the single
# biggest win we found, +0.42 test F1 - reported as a deviation, not smuggled into the default.
FREEZE_BACKBONE = True  # train the head only  (ablation: --unfreeze)

# How ground-truth boxes are assigned to grid cells  (ablation):
#   "center" - exactly the task sheet: ONLY the cell containing the box center is positive.
#              With 453 training boxes that is 453 positive signals against 205k negative
#              cells, and the neighbouring cells - which fire anyway - are never taught what
#              box to predict, so they emit undersized fragments.
#   "multi"  - the center cell PLUS its 2 nearest neighbours are positive, all regressing the
#              same box (this is what YOLOv5 does). 3x the positive supervision, and the
#              neighbours now agree with the center instead of fragmenting, so NMS merges
#              them cleanly. Requires the wider offset range below.
ASSIGN = "center"  # the task sheet's rule; "multi" is our tested deviation (--assign multi)

# Offset activation range. "center" assignment only ever needs offsets inside the cell -> a
# plain sigmoid, [0, 1]. "multi" needs a neighbouring cell to place a center up to half a cell
# outside itself -> sigmoid(t)*2 - 0.5, giving [-0.5, 1.5]. Getting this wrong is silent: the
# model simply cannot reach the target and box loss plateaus.
OFFSET_RANGE = {"center": (0.0, 1.0), "multi": (-0.5, 1.5)}

# Split: 80% train / 15% val / 5% test, split BY IMAGE so no image leaks across splits.
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0.80, 0.15, 0.05
SEED = 42

# "temporal" | "random". All 1001 images are frames of ONE video sampled every 20 frames, so
# neighbouring frames are near-duplicates. A random split lands a frame in train and its
# 0.67 s-later twin in test, which inflates every metric. "temporal" splits on the time axis
# instead and is the honest setting; "random" is kept so the leakage can be quantified.
SPLIT_MODE = "temporal"

# Training
EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-3  # for the frozen-backbone default; use 1e-4 together with --unfreeze
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Loss weighting: total = LAMBDA_OBJ * objectness + LAMBDA_BOX * box
# Only ~1-3 of the 256 cells are positive, so objectness is heavily imbalanced; POS_WEIGHT
# up-weights the positive term inside BCEWithLogitsLoss.
LAMBDA_OBJ = 1.0
LAMBDA_BOX = 5.0
POS_WEIGHT = 20.0
BOX_LOSS = "l1"  # "l1" | "ciou"  (ablation)

# Augmentation
AUG_HFLIP = True
AUG_COLOR_JITTER = True

# Inference
SCORE_THRESH = 0.5  # tuned on the validation split, never on test
NMS_IOU = 0.5
IOU_MATCH = 0.5  # an IoU >= 0.5 with a GT box counts as a correct detection

# --------------------------------------------------------------------------------------
# Part 2 - YOLO fine-tuning, tracking, counting
# --------------------------------------------------------------------------------------

YOLO_MODEL = "yolo11n.pt"
YOLO_EPOCHS = 40
YOLO_IMGSZ = 640
YOLO_BATCH = 16
YOLO_DATA_DIR = DATA_DIR / "yolo"  # dataset converted to YOLO format
YOLO_CONF = 0.25

# COCO class ids that count as a vehicle. The stock yolo11n weights are trained on COCO's 80
# classes; the task asks for a single `vehicle` class, so these four are merged into one and
# every other class (person, traffic light, ...) is discarded. The fine-tuned model predicts
# one class directly and ignores this list.
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Tracker
#
# The two temporal parameters are given in SECONDS, not frames, and converted with the video's
# own frame rate by `track_frames()` below. This matters because the course tests the submitted
# model on a separate video we will never see: a threshold of "10 frames" means 0.33 s at 30 fps
# but 0.17 s at 60 fps, so the same config would silently make the tracker twice as impatient,
# tear down tracks through short occlusions and fragment them. Nothing would raise an error - the
# count would just be wrong. Expressed in seconds, the behaviour is the same on any frame rate.
TRACK_IOU_THRESH = 0.3  # below this, a detection cannot be matched to a track
TRACK_MAX_AGE_SECONDS = 0.33  # drop a track unseen for this long  (10 frames @ 29.97 fps)
TRACK_MIN_HITS_SECONDS = 0.10  # a track must be seen this long before it may be counted (3 fr)
TRACK_MATCH = "hungarian"  # "hungarian" | "greedy"  (compared in docs/experiments.md)


def track_frames(fps: float) -> tuple[int, int]:
    """(max_age, min_hits) in frames for a video of `fps`. Both are at least 1 frame."""
    return (
        max(1, round(TRACK_MAX_AGE_SECONDS * fps)),
        max(1, round(TRACK_MIN_HITS_SECONDS * fps)),
    )

# Counting line, in *normalized* image coordinates (x, y in [0, 1]) so it is resolution
# independent - the same setting is valid on the 1080p and the 4K encode of the clip, and
# re-encoding cannot silently move the line.
#
# Placed by measurement, not by eye — `python -m src.part2.suggest_line` sweeps candidate lines
# over the tracked paths and scores coverage / direction balance / flow alignment. On this clip
# the horizontal line at y = 0.65 is crossed by 42 of 93 moving tracks (31/11 per direction).
# suggest_line's alignment ranking now prefers the vertical x = 0.70 line (41 crossings, but it
# reads direction from the dominant motion component); we keep the horizontal one because the
# manual ground truth was counted against it, and both lines agree on the split (31/11 vs 30/11).
# See docs/experiments.md.
COUNT_LINE = ((0.0, 0.65), (1.0, 0.65))

# Which sign of the crossing means what. With the line drawn left-to-right, a vehicle moving
# DOWN the frame (towards the camera) crosses to the positive side; one moving up, away.
DIRECTION_LABELS = {1: "toward camera", -1: "away from camera"}
