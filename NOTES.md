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

## 2026-07-12 — 12 ablations: test F1 0.423 → 0.904

Full table in `docs/experiments.md`. The prescribed baseline (frozen ResNet18, one positive cell per
box) reaches **test F1 0.423 / AP50 0.213**. The best configuration reaches **test F1 0.904 / AP50
0.871** — AP50 improved 4×. Two changes did essentially all of it.

**Lever 1 — the frozen backbone was the bottleneck, not the head.** Unfreezing `layer4` (lr 1e-4)
alone takes test F1 from 0.423 to **0.838**. In hindsight this is obvious: ImageNet is object-centric
photographs, our vehicles are small, motion-blurred and shot from a moving dashcam. No detection head
can compensate for features that were never tuned for that. The clincher is the MobileNetV3 result —
*frozen*, it scores 0.778 against ResNet18's 0.423, but once both are unfrozen the gap collapses to
0.904 vs 0.880. The problem was never capacity; it was frozen-feature mismatch.

⚠️ **This deviates from the task sheet**, which says "you don't need to train the backbone, only train
your detection head". That is permission, not prohibition — but we therefore report **both**: the
prescribed frozen detector *and* the fine-tuned one, with the gap as the finding. Worth confirming
with the supervisor that the deviation is acceptable, since it is our strongest result.

**Lever 2 — multi-cell assignment.** The task sheet's "one positive cell per box" gives 453 positive
signals against ~205 000 negative cells. The neighbouring cells fire regardless, but were never
taught *which* box to emit — so they produced undersized fragments that NMS could not merge, costing
a false positive *and* a false negative on the same car. Assigning the center cell **plus its two
nearest neighbours**, all regressing the same box (what YOLOv5 does), turns those fragments into
agreeing votes. Frozen: 0.423 → **0.648**, with test false positives dropping 18 → 10.

This required widening the offset activation from `sigmoid` ([0,1]) to `sigmoid*2 − 0.5`
([−0.5, 1.5]), because a neighbouring cell must be able to place a box center *outside itself*. That
is a silent failure mode — with a plain sigmoid the target is simply unreachable and the box loss
plateaus at a suspiciously non-zero value with no error raised — so `tests/test_encoding.py` now
asserts all three assigned cells decode back to the *same* box, and that the widened activation can
actually reach the range the encoder emits.

**What did NOT work (all real, all going in the presentation):**
- **CIoU lost to plain L1**, which we did not expect since CIoU optimises the IoU we are scored on.
  Our read: with objects this small (median 1.6% of image area) and predictions initially far from
  the targets, CIoU's gradient is poorly conditioned; L1 in the bounded sigmoid space is just an
  easier optimisation problem.
- **Focal loss was the worst run of all** (0.421). It is built for the extreme imbalance of dense
  anchor detectors; here `pos_weight=20` already handles a far milder one, and focal's suppression
  of easy negatives starved the objectness head.
- **Plain BCE collapsed recall to 0.395** — it learned to answer "no vehicle" and bank 99% cell
  accuracy. Exactly the failure the imbalance handling exists to prevent.
- **Our augmentation bought nothing.** hflip + colour jitter: val F1 **0.612** vs **0.613** without.
  Identical. The clip is one dashcam pass down one road — mirroring and re-tinting it does not create
  the variety the model lacks. What is missing is *scale and viewpoint* variation, so random
  scale/translate crops are the thing to try, not more colour tricks. Reporting this as a negative
  result rather than quietly dropping the row.

**Bug caught by looking at the pictures, not the metrics (round 1).** The first render of the best model showed
red "false positives" stacked on cars the metrics had scored as clean. The metrics were right and the
*visualiser* was wrong: it decoded a `multi` model with the `center` offset activation and used the
default NMS instead of the tuned one. Both are silent — the boxes come out subtly shifted and
duplicated, with no error. Fixed by making `visualize.py` read `assign` and `nms_iou` from the
checkpoint and metrics, exactly as `evaluate.py` does. A good reminder for the presentation: the
figure and the table must be produced by the same code path, or one of them is lying.

---

## 2026-07-12 — the recall cliff: two failed fixes, and two mistakes of our own

The best model's PR curve is strong but ends in a **cliff** — ~13% of vehicles never detected at
*any* confidence. Rather than guess (more epochs? better loss?), we bucketed recall by ground-truth
box area, as a **% of the image** (`src/part1/analysis.py`):

| GT box area | Recall |
|---|---|
| < 0.5% of image | 0/1 |
| 0.5–1% | **11/15 = 0.73** |
| > 1% | **22/22 = 1.00** |

**Every one of the 5 misses is a small, distant vehicle. Everything above 1% of the image is found,
perfectly.** That reads like a resolution problem — at stride 32 a distant car spans barely one 32 px
cell. So we tried the two obvious fixes. **Both failed.**

**Attempt 1 — finer grid (stride 16): FAILED.** A 32×32 grid dropped test F1 0.904 → 0.822. Spatial
resolution was bought at the price of **semantic depth**: cutting the backbone earlier gives shallower
features that have passed through fewer non-linearities. Resolution and semantics trade off, and here
the trade lost. This is precisely the problem a **Feature Pyramid Network** solves — upsample the deep
stride-32 features and fuse them with the shallow stride-16 ones, so you get both. That, or more data,
is the real fix for small objects; it is the honest next step.

**Attempt 2 — larger input (640 / 768 px): NO REAL EFFECT.** 640 px *looked* like a win (F1 0.904 →
0.917, AP50 0.871 → 0.935) and we nearly reported it as one. Then we looked at the counts:

| Input | TP | FP | FN | Recall |
|---|---|---|---|---|
| 512 (task sheet) | 33 | 2 | 5 | 0.868 |
| 640 | **33** | 1 | **5** | **0.868** |
| 768 | 32 | 2 | 6 | 0.842 |

**640 px finds the exact same 33 vehicles as 512 px.** Identical TP, identical FN — not one additional
small car recovered. All it did was drop a single false positive and re-order the confidence ranking,
which is what moved AP50. On a 38-box test set one box is ~2.6 points of recall, so 512 / 640 / 768 are
**statistically indistinguishable**. We keep the **512 px the task sheet specifies**: the bigger input
bought nothing real, and claiming a win on a one-box difference would be dishonest.

### Two mistakes we made here, recorded because they are the lesson

1. **We evaluated a model that was still training.** The first 768 px numbers (F1 0.818, recall 0.711)
   came from a mid-training checkpoint read while the job was still running. The finished model scores
   0.889. Every model is now evaluated only after training completes.
2. **We bucketed box area in network-input pixels.** Those scale with `img_size` — so the *same* car
   lands in a bigger bucket at 640 px than at 512 px, and the buckets appear to improve when nothing
   has changed. This is what made a bigger input look like it had fixed small-object recall. Areas are
   now a scale-invariant **% of the image**.

Both errors pointed the same way: they flattered the change we were hoping would work. That is not a
coincidence, and it is the reason to check a result that comes out the way you wanted it to.

**Confusion matrices** (`analysis.py` writes both):
- *Detection level*: TP 33 / FP 2 / FN 5. Background-vs-background is **undefined** for a detector —
  there is no "correctly predicted nothing" when the negative class is every possible box — so that
  cell is marked n/a rather than filled with a fake number.
- *Grid-cell objectness*: TP 86 / FP 15 / FN 26 / **TN 12,673**. A real 2×2, because the head literally
  *is* a binary classifier over grid cells. The 12,673 : 86 ratio **is** the class imbalance in raw
  numbers — the reason plain BCE collapses to "no vehicle" and `pos_weight` has to exist.

**Test-suite lesson.** Changing the config defaults broke 5 tests, because they had been silently
inheriting `config.ASSIGN` / `config.IMG_SIZE` instead of pinning the geometry they assert. Fixed by
making each test state its own scheme and grid. A test that reads a mutable global is not testing what
it claims to.

**Final Part 1 result (512 px, exactly as specified): test P 0.943 / R 0.868 / F1 0.904 / AP50 0.871**,
against the prescribed baseline's 0.455 / 0.395 / 0.423 / 0.213.

---
