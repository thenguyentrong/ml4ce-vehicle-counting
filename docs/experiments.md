# Experiments & results

Every number that goes on a slide comes from this file. Tables are filled in as runs complete —
**empty cells mean "not run yet", never "we forgot"**.

## Dataset facts (EDA)

Measured by `uv run python -m src.eda`.

| Quantity | Value |
|---|---|
| Images | **1001** (all frames of **one** video, `vid_4`, sampled every 20 frames) |
| Native image size | 676 × 380 |
| Total boxes | **559** |
| Images containing a vehicle | 355 (35.5%) |
| Images with no vehicle | **646 (64.5%)** — genuinely empty road; kept as negatives |
| Boxes per image | mean 0.56, max 7 |
| Box width (px) | median **100** (min 19, max 299) |
| Box height (px) | median **42** (min 17, max 137) |
| Median box area | 1.59% of the image |
| **Boxes lost to same-cell collisions (16×16 grid)** | **2 / 559 = 0.36%** |

The last row is the hard recall ceiling of the one-box-per-cell design the task sheet prescribes:
**99.64%**. It is a non-issue on this dataset (the vehicles are far apart) but would dominate on
dense city traffic — which is exactly why real detectors use anchors and multiple FPN levels.

## The split matters more than any hyperparameter

All 1001 images are frames of a **single video**, 20 frames (~0.67 s) apart. A *random* split puts
a frame in train and its near-identical neighbour in test, so the model is scored on cars it has
already memorised. Splitting along the **time axis** instead (earliest 80% train → next 15% val →
last 5% test) is the honest setting. Identical model and hyperparameters, only the split changed:

| Split | Val precision | Val recall | **Val F1** | Test F1 | Test AP50 |
|---|---|---|---|---|---|
| Random (leaky) | 0.731 | 0.835 | **0.779** | 0.491 | 0.374 |
| **Temporal (honest)** | 0.500 | 0.618 | **0.553** | 0.410 | 0.236 |

**Leakage inflates val F1 by +0.23 (0.55 → 0.78).** Every number below uses the temporal split.
Reporting the random-split number would have overstated the detector by ~40% relative.

## Part 1 — NMS threshold (tuned on val, temporal split)

One vehicle often excites **two neighbouring cells**. The two boxes it produces overlap each other
too little for a lax NMS to merge — so the duplicate becomes a false positive *and* each fragment
misses the ground-truth box at IoU 0.5, costing a false negative too. One car, two errors.

| NMS IoU | Score thr. | Val precision | Val recall | Val F1 |
|---|---|---|---|---|
| **0.1** | 0.70 | **0.679** | 0.559 | **0.613** |
| 0.2 | 0.65 | 0.623 | 0.559 | 0.589 |
| 0.3 | 0.60 | 0.582 | 0.574 | 0.578 |
| 0.4 | 0.65 | 0.558 | 0.632 | 0.593 |
| 0.5 (default) | 0.70 | 0.500 | 0.618 | 0.553 |

Aggressive NMS (0.1) raises precision 0.50 → 0.68 at almost no cost in recall: **val F1 0.553 →
0.613**. It is merging exactly the duplicate fragments predicted from adjacent cells.

## Part 1 — baseline result

Frozen ResNet18, L1 box loss, pos_weight=20, augmentation on, λ_obj:λ_box = 1:5, 40 epochs
(166 s on an RTX 3090). Score threshold **0.70** and NMS IoU **0.1**, both tuned on validation.

| Split | Precision | Recall | F1 | AP50 |
|---|---|---|---|---|
| Validation (150 images) | 0.679 | 0.559 | 0.613 | — |
| **Test (50 images)** | **0.455** | **0.395** | **0.423** | 0.213 |

> ⚠️ **The test split is too small to trust on its own.** 5% of 1001 images = 50 images holding only
> **38 boxes**. A single bad frame moves F1 by several points, which is why test (0.42) and val
> (0.61) diverge so much. The 80/15/5 ratio comes from the task sheet, so we keep it — but we report
> **validation** as the primary number and treat test as a spot check. Worth raising with the
> supervisor.

## Part 1 — ablations (to run)

| # | Backbone | Box loss | Imbalance | Aug | λ_obj:λ_box | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ResNet18 | L1 | pos_weight=20 | on | 1:5 | 0.679 | 0.559 | **0.613** | baseline (val) |
| 2 | ResNet18 | CIoU | pos_weight=20 | on | 1:5 | | | | |
| 3 | ResNet18 | L1 | plain BCE | on | 1:5 | | | | |
| 4 | ResNet18 | L1 | focal | on | 1:5 | | | | |
| 5 | ResNet18 | L1 | pos_weight=20 | **off** | 1:5 | | | | |
| 6 | MobileNetV3 | L1 | pos_weight=20 | on | 1:5 | | | | |
| 7 | ResNet18 (layer4 unfrozen) | L1 | pos_weight=20 | on | 1:5 | | | | |

**Which works best and why:** _to be written once the ablations are in._

## Part 2 — detector comparison

| Detector | Trained on | Precision | Recall | Notes |
|---|---|---|---|---|
| YOLO11n fine-tuned | Kaggle car-object-detection | | | |
| YOLO11n off-the-shelf | COCO (car/truck/bus) | | | zero-shot on the video |
| Part 1 head (ours) | Kaggle car-object-detection | | | for reference — expect it to lose |

## Part 2 — tracking & counting

| Tracker variant | Counted ↑ | Counted ↓ | Manual ↑ | Manual ↓ | Abs. error | ID switches |
|---|---|---|---|---|---|---|
| IoU + Hungarian | | | | | | |
| IoU + Hungarian + constant-velocity prediction | | | | | | |

**Failure cases:** _to be written — where do ID switches and missed vehicles occur, and how do they
move the final count?_
