# Lab notebook

Running log of what we tried, what worked, and **what did not**. The presentation is explicitly
graded on this ("explain what you tried and what did and did not work"), and none of it is
reconstructable from memory in September — so it gets written down the day it happens.

Newest entry at the bottom. Quantitative results go in `docs/experiments.md`; this file records
*decisions and reasoning*.

---

## 2026-07-12 — Project setup

**Environment.** The machine's system Python is 3.14, which is too new to have reliable
torch/ultralytics wheels. Created a dedicated **Python 3.12** virtualenv with `uv` instead.
torch/torchvision are pulled from the CUDA 12.6 index (`download.pytorch.org/whl/cu126`) — the
default PyPI wheels are CPU-only, which would silently make everything ~50x slower without ever
raising an error. GPU is an RTX 3090 (24 GB), so both parts should train in minutes.

**Reading the task sheet.** The DOCX and the course PDF disagree, and it matters:

| | Course PDF (topic slide) | DOCX (topic folder) |
|---|---|---|
| Dataset | Roboflow `vehicles-coco`, 19 000 images | Kaggle "Car Object Detection", ~1000 images |
| Classes | car / truck / bus / motorcycle | **one class is sufficient** |
| Counting | per class | per direction |

We follow the **DOCX**, since it is the document handed out with the topic and is far more specific
(it prescribes the backbone/grid/head design). Recorded in `docs/task_spec.md`. Worth raising with
the supervisor to confirm, since the PDF says models are tested on a held-out set after submission.

**Design decision — one box per cell.** The task sheet prescribes a 16×16 grid where "every grid
cell is responsible for the objects whose center falls into it". This design *cannot represent two
objects whose centers land in the same cell* — the second one is silently dropped from the target.
This is a real ceiling on recall, so rather than hand-wave it we measure the collision rate on the
actual dataset during EDA (see `docs/experiments.md`). It is also the reason real detectors use
anchors / FPN levels, which is a good point for the presentation.

**OneDrive vs. uv.** `uv sync` failed with `os error 396` — it populates a venv by *hardlinking*
from its cache, and OneDrive's cloud filesystem does not support hardlinks. Fixed by pinning
`link-mode = "copy"` in `pyproject.toml`, so the fix travels with the repo instead of living in
someone's shell profile.

---

## 2026-07-12 — EDA: two things about this dataset that change the plan

Dataset downloaded (112 MB): **1001 images at 676×380**, **559 boxes**, from dashcam footage of a
suburban road. Full numbers in `docs/experiments.md`.

**1. Two thirds of the images have no annotation — and that is legitimate.** 646 of 1001 images
(64.5%) carry no box at all. That looked alarming (unlabeled vehicles used as negatives would train
the objectness head to actively *suppress* cars), so we rendered a random sample and looked at them:
they are genuinely empty road — the dashcam driving past trees and asphalt. They are **kept as
all-negative samples**; they are exactly what teaches the head what "no vehicle" looks like. Only
355 images actually contain a vehicle, so the *effective* dataset is far smaller than "1000 images"
suggests — worth saying out loud in the presentation.

**2. The annotations are incomplete, and this will cap our precision.** Spot-checking labeled images
shows frames where one car is boxed while other clearly visible cars — queued, parked, or partially
occluded — are not. A detector that correctly finds those cars gets scored as producing a **false
positive**. So the precision we measure is a *lower bound* on true precision, and a chunk of our
"errors" will be the model being right and the label being wrong. We will show exactly such a case
in the failure-analysis slide rather than pretending the number is clean.

**3. The one-box-per-cell ceiling is a non-issue here.** Measured on the real data: only **2 of 559
boxes (0.36%)** are lost to two centers landing in the same 16×16 cell → **recall ceiling 99.64%**.
So the design the task sheet prescribes is a fine fit for this dataset, and we can say so with a
number instead of a hunch. (It would *not* be fine on dense city traffic — that is the point worth
making about why real detectors use anchors and multiple FPN levels.)

Boxes are small: median **100×42 px**, i.e. ~1.6% of the image area. At 512×512 input with stride 32,
the median box is ~76×57 px — comfortably larger than one 32 px cell, so a stride-32 feature map is
not too coarse for this data.

---

## 2026-07-12 — Part 1 detector works; the split was quietly lying to us

**Pipeline verified before trusting any number.** Two checks, both of which a broken detector
fails silently:
1. `tests/test_encoding.py` — encode ground-truth boxes into the grid and decode them straight back.
   Round-trips at **IoU 1.0**. This is what catches an off-by-one in the cell index or a mismatched
   coordinate convention, neither of which would ever raise an error; they would just cap the model
   at mediocre and look like "the model needs more epochs".
2. `train.py --overfit` — one batch, 300 steps: loss **2.68 → 0.029 (−98.9%)**, with *both* the
   objectness and the box term falling. A pipeline that cannot memorise 16 images has a bug, and no
   amount of training on the full set will rescue it.

**The finding that changes our numbers: temporal leakage.** All 1001 images are frames of a *single*
video, sampled every 20 frames — about 0.67 s apart. A random train/test split therefore puts a
frame in train and its near-identical twin in test. The model gets credit for recognising a car it
has already memorised, and every metric is flattered. We only caught it by looking at the filenames
(`vid_4_10000.jpg`, `vid_4_10020.jpg`, …) after the val→test gap looked implausible.

Same model, same hyperparameters, only the split changed:

| Split | Val F1 |
|---|---|
| Random (naive) | **0.779** |
| Temporal (honest) | **0.553** |

**The naive split overstates the detector by +0.23 F1** — about 40% relative. Both are kept in the
code (`config.SPLIT_MODE`) so we can *show* this in the presentation rather than assert it. This is
the single most important slide in Part 1: it is a mistake anyone would make by default, it is
invisible in the loss curves, and in a civil-engineering deployment (a camera on a road you have
never seen) it is exactly the error that matters.

**Failure analysis from looking at the predictions.** Rendering GT vs prediction side by side
(`runs/temporal/predictions.png`) explained the mediocre precision immediately — something no metric
could have told us:

- **Duplicate fragments (the dominant error).** One car excites *two adjacent cells*. Each predicts
  its own, slightly undersized box. The two boxes overlap each other too little for NMS at IoU 0.5
  to merge them → the extra box is a **false positive**, *and* each fragment misses the GT box at
  IoU 0.5 → a **false negative** too. One car, two errors, and it hits precision and recall at once.
  → Fix: tune NMS. Dropping the NMS IoU to **0.1** merges the fragments and lifts precision
  0.50 → 0.68, **val F1 0.553 → 0.613**, at almost no cost in recall.
- **"False positives" that are actually correct.** In one test frame the model finds four vehicles
  at a distant intersection — all real, none labelled. They score as false positives. Our measured
  precision is therefore a **lower bound**; part of our error rate is the dataset being wrong, not
  the model. This frame goes on the failure-analysis slide as-is.
- **Missed distant/small cars** — the genuine remaining recall gap.

**Test-set size is a problem.** The task sheet's 5% test split = 50 images containing just **38
boxes**. One bad frame swings F1 by several points, which is why test (0.42) and val (0.61) diverge
so widely. We keep 80/15/5 as prescribed, but report **validation** as the primary number and say
plainly that test is a spot check. Worth asking the supervisor whether a k-fold over the temporal
blocks would be acceptable — it would give an honest number *and* an error bar.

**Baseline stands at val F1 0.613** (P 0.679 / R 0.559), 166 s to train on the RTX 3090. Ablations
(CIoU vs L1, focal vs pos_weight, MobileNetV3, unfreezing layer4, augmentation off) are next; the
table in `docs/experiments.md` is already laid out for them.

---
