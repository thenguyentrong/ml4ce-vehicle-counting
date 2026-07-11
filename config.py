"""Central configuration: every path and hyperparameter used by the project lives here.

Nothing else in the codebase hard-codes a path, so the project stays runnable on another
machine (and on the graders' held-out test set) by editing this file alone.

Author: Vinh Nguyen
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

# Part 2: the traffic video. Not yet provided by the course - drop the file here and the
# Part 2 scripts pick it up (or pass --video explicitly).
VIDEO_PATH = DATA_DIR / "traffic.mp4"

# --------------------------------------------------------------------------------------
# Part 1 - detection head on a frozen backbone
# --------------------------------------------------------------------------------------

IMG_SIZE = 512  # network input is IMG_SIZE x IMG_SIZE
STRIDE = 32  # backbone output stride (ResNet18 through layer4)
GRID = IMG_SIZE // STRIDE  # -> 16 x 16 grid, as prescribed by the task sheet

BACKBONE = "resnet18"  # "resnet18" | "mobilenet_v3_large"  (ablation)
FREEZE_BACKBONE = True  # train the head only  (ablation)

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
LR = 1e-3
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

# Tracker
TRACK_IOU_THRESH = 0.3  # below this, a detection cannot be matched to a track
TRACK_MAX_AGE = 10  # kill a track after this many consecutive unmatched frames
TRACK_MIN_HITS = 3  # a track must be confirmed this often before it may be counted

# Counting line, in *normalized* image coordinates (x, y in [0, 1]) so it is resolution
# independent. Default: a horizontal line across the middle of the frame. Re-tune once the
# real video is available.
COUNT_LINE = ((0.0, 0.5), (1.0, 0.5))
