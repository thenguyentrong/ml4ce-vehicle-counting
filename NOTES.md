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

## 2026-07-20 — Part 2: the video was the hard part, not the tracker

**The tracker and counter went in cleanly.** IoU association with both greedy and Hungarian
matching (`--match`), `TRACK_MIN_HITS` before a track may be counted, `TRACK_MAX_AGE` before it is
terminated, and a counting line tested as a proper segment–segment intersection so a vehicle
crossing the line's *infinite extension* off to the side is not counted. `tracker.py` and
`counter.py` import no ultralytics and no OpenCV, which is what let us unit-test all of it on
synthetic boxes — 28 tests, no video required.

**A bug the tests caught immediately, and would never have been noticed on video.** The first
synthetic vehicle driven across the line was not counted. Cause: `_side()` returns exactly 0 when
a box centre lands *on* the line, and the code treated that as a third state ("neither side"). A
vehicle then steps onto the line in one frame and off it in the next, and no single step ever sees
two different sides — the crossing vanishes. Box centres are computed from integer pixel
coordinates and land on round values constantly, so this is not a corner case. Fixed by folding
zero into the positive side (`>= 0`) so "which side" is a genuine boolean. On the rendered video
this would have looked like nothing at all: a plausible clip with boxes and IDs and a count that
was simply too low, with no way to tell.

### Sourcing the video: three wrong answers before a right one

The course did not supply a video, and the search took far longer than writing the tracker. What
made it hard is that the binding requirement is not the obvious one.

- **Dusk / congested UK dual carriageway** — two directions, static, but 18.7 detections/frame.
  Rejected on *hand-countability*: the ground truth has to be produced by a human once, and 200+
  mutually-occluding vehicles cannot be counted reliably. We checked whether dusk itself broke
  detection before rejecting it — it does not, mean confidence 0.551 at dusk vs 0.542–0.593 in
  daylight. **Density was the problem, not light.**
- **US freeway** — same density problem, plus the Pixabay licence forbids redistributing content
  "on a standalone basis", which rules it out of a public repo.
- **Daylight two-lane, 7.4 det/frame** — perfect on every axis except that all traffic flows one
  way, so "count per direction" is meaningless.

**The mistake worth recording.** We then picked a T-junction clip and were confident: static
camera (verified with a difference image), daylight, both directions visible in the same frame,
9.2 det/frame. Every stated requirement, verified by eye. It was still wrong. Running the tracker
over it and sweeping candidate counting lines showed the best line anywhere in the frame is crossed
by **15 of 34 moving tracks, 14 of them in the same direction** — the traffic *disperses* at the
junction instead of passing through any single cross-section.

The requirement was never "both directions are visible". It is "**there exists one line that both
flows cross**", and that is not visible in a still image, only in the trajectories. The chosen
intersection clip scores 42 of 93 crossings split 31/11 on the same test. Lesson, and it is the
same one as the 640 px episode in Part 1: *we verified the property we could see instead of the
property that mattered.*

The counting line's position (y = 0.65) came out of that same sweep rather than being placed by
eye — the first line we drew by hand was on the wrong axis entirely, because the road's dominant
flow direction is not what the still image suggests.

### Two results that came out opposite to our prediction

**Fine-tuning helped, and we expected it to hurt.** The argument for it hurting was good: Part 1's
data is dashcam footage (rear views, road level), the video is an elevated static street camera
(head-on), and fine-tuning on 355 dashcam frames should specialise the model *away* from the target
domain. Measured: fine-tuned counts 47 vehicles, off-the-shelf COCO counts 29, and the fine-tuned
model creates ~200 fewer tracks. The mechanism is visible in the numbers — the off-the-shelf model
emits nearly **twice** as many detections per frame (10.16 vs 5.64) while counting *fewer*
vehicles. It spends them on parked cars, pedestrians and distant traffic that never cross the line,
and its boxes are less stable, so tracks fragment (21.0 tracks per counted vehicle vs 8.7) and the
fragments fail to reach `TRACK_MIN_HITS` at the moment they cross.

**Hungarian vs greedy is a tie on this footage: 47 vs 47, 29 vs 29.** The only difference is a
handful of tracks (408 vs 412) — a few ID switches avoided. This is worth saying plainly rather
than dressing up: optimal assignment only pays when a track has several plausible detections
competing for it, and at 6–10 well-separated vehicles per frame that ambiguity is rare. The unit
test builds the case where greedy provably loses a track and invents a phantom ID; this video just
does not generate it often. On the congested clip we rejected, it would — which is a better
argument for Hungarian than any number we can produce here.

### Still open

The **manual count is not done**, and it is the only external ground truth this project has. Until
it exists, `evaluate.py` deliberately refuses to print an accuracy: substituting one of the runs as
"truth" would make the evaluation circular. `runs/part2/manual/reference.mp4` (the clip with only
the counting line burned in, plus a frame index) is rendered and ready, and the counting rules are
fixed in advance in `docs/manual_count.md` so that the human and the machine answer the *same*
question — crossing, not presence.

---

## 2026-08-14 — Submission-readiness pass

Ten days before the deadline we stopped adding results and tested the part that gets graded after
submission: someone else running this on data we never saw. 33/33 tests pass and
`src.part1.evaluate` reproduces P 0.943 / R 0.868 / F1 0.904 / AP50 0.871, so the README table is
not stale. Three real problems came out.

**1. Part 1 could not run on anyone else's images.** `evaluate.py` goes through `build_loaders()`,
which needs the Kaggle CSV and our own split, so it can only score our 50 test frames. Added
`src.part1.predict`: folder in, `predictions.csv` in original-image pixels out, plus annotated
images and metrics if a ground truth CSV is given. It picks the checkpoint with the best measured
test F1 instead of a hard-coded name, and takes the threshold and NMS IoU from that run's
`metrics.json`.

While writing it: the first version decoded at the tuned threshold and got AP50 0.846 where
`evaluate.py` says 0.871, with identical P/R/F1. AP integrates the whole PR curve, so dropping the
low-confidence boxes first cuts the tail off and understates it. Fix: decode at threshold 0,
threshold afterwards for the reported boxes. Both paths agree to the digit now.

**2. `run_count` would have counted only the first 60 s of any video, silently.**
`config.VIDEO_SECONDS = 60` was the argparse default; it exists to trim *our* 64 s clip to the
window the manual count covers. On a three-minute video from the course it would have read one
minute and printed a confident number, and the missing vehicles would look like a tracker problem.
Now `config.frames_to_process()` trims only our own clip or when `--seconds` is given, says how
much it reads, and stores `frames_available` next to `frames`. Same class of bug as frames vs
seconds in the tracker: a default that is right for one video and wrong everywhere else.

**3. The default detector was the un-fine-tuned one.** `--weights` defaulted to `stock`, so a run
without flags used the COCO model that counts 29 instead of 47. `run_count` and `suggest_line` now
default to the fine-tuned weights when they exist, print which file they loaded, and fall back to
stock otherwise.

**Checked end to end.** Built the zip, extracted it somewhere else, ran it like a grader would:
tests pass, `predict` reproduces 0.943 / 0.868 / 0.904 / 0.871 from the shipped checkpoint, and
`run_count` counts from the shipped weights and video with no path fixes. On a clip re-encoded to
1280×720 at 25 fps, `suggest_line` proposed a vertical line, `run_count` counted 6 of 11 moving
vehicles, and the tracker thresholds became 8 and 2 frames instead of 10 and 3 — which is exactly
why they are written in seconds. The 60 s clip still gives 32/15/47 with 408 tracks.

Also: the README listed three notebooks in `notebooks/` that do not exist. `yolo26n.pt` in the
root is a stray download, nothing uses it, it is not in the bundle.
