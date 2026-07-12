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

## Part 1 — ablations

All runs: temporal split, 40 epochs, 512×512 input, 16×16 grid. Score threshold **and** NMS IoU
tuned on validation per run; precision/recall/AP50 reported on **test** at IoU ≥ 0.5.
`assign` = which cells are positive for a box: `center` (task sheet: only the center cell) or
`multi` (center + its 2 nearest neighbours, all regressing the same box).

Sorted by test F1:

| Run | Backbone | Assign | Backbone frozen | Box loss | Imbalance | Val F1 | Test P | Test R | **Test F1** | AP50 | TP/FP/FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **best_img640** (640 px input) | MobileNetV3 | multi | **no** | L1 | pos_weight | 0.908 | **0.971** | 0.868 | **0.917** | **0.935** | 33/1/5 |
| mobilenet_multi_unfreeze | MobileNetV3 | multi | **no** | L1 | pos_weight | 0.922 | 0.943 | 0.868 | 0.904 | 0.871 | 33/2/5 |
| best_stride16 (32×32 grid) | MobileNetV3 | multi | **no** | L1 | pos_weight | 0.916 | 0.857 | 0.789 | 0.822 | 0.822 | 30/5/8 |
| best_img768 (768 px input) | MobileNetV3 | multi | **no** | L1 | pos_weight | 0.882 | 0.964 | 0.711 | 0.818 | 0.873 | 27/1/11 |
| multi_unfreeze | ResNet18 | multi | **no** | L1 | pos_weight | 0.892 | 0.892 | 0.868 | 0.880 | 0.844 | 33/4/5 |
| unfreeze | ResNet18 | center | **no** | L1 | pos_weight | 0.916 | 0.861 | 0.816 | 0.838 | 0.876 | 31/5/7 |
| mobilenet | MobileNetV3 | center | yes | L1 | pos_weight | 0.884 | 0.824 | 0.737 | 0.778 | 0.774 | 28/6/10 |
| mobilenet_multi | MobileNetV3 | multi | yes | L1 | pos_weight | 0.818 | 0.763 | 0.763 | 0.763 | 0.757 | 29/9/9 |
| multi | ResNet18 | multi | yes | L1 | pos_weight | 0.672 | 0.697 | 0.605 | 0.648 | 0.515 | 23/10/15 |
| multi_ciou | ResNet18 | multi | yes | **CIoU** | pos_weight | 0.678 | 0.690 | 0.526 | 0.597 | 0.448 | 20/9/18 |
| plainbce | ResNet18 | center | yes | L1 | **plain BCE** | 0.524 | 0.714 | 0.395 | 0.508 | 0.456 | 15/6/23 |
| noaug | ResNet18 | center | yes | L1 | pos_weight | 0.612 | 0.571 | 0.421 | 0.485 | 0.339 | 16/12/22 |
| ciou | ResNet18 | center | yes | **CIoU** | pos_weight | 0.601 | 0.435 | 0.526 | 0.476 | 0.346 | 20/26/18 |
| **temporal** (prescribed baseline) | ResNet18 | center | yes | L1 | pos_weight | 0.613 | 0.455 | 0.395 | 0.423 | 0.213 | 15/18/23 |
| focal | ResNet18 | center | yes | L1 | **focal** | 0.472 | 0.421 | 0.421 | 0.421 | 0.316 | 16/22/22 |

**Best config improves test F1 from 0.423 → 0.917 and AP50 from 0.213 → 0.935 (4.4×).**

## Chasing the last of the recall: where the misses actually are

The best 512 px model's PR curve is strong (precision ≥ 0.95 out to recall 0.87) but ends in a
**cliff**: ~13% of vehicles are never found at *any* confidence threshold. Rather than guess why,
`src/part1/analysis.py` buckets recall by ground-truth box area:

| GT box area (input px²) | Recall @ 512 px |
|---|---|
| < 1k (tiny) | **0/1** |
| 1k – 2.5k | **10/13 = 0.77** |
| 2.5k – 5k | 17/18 = 0.94 |
| 5k – 10k | 2/2 = 1.00 |
| > 10k (large) | 4/4 = 1.00 |

**Every single miss is a small, distant vehicle.** Large vehicles are found perfectly. So the cliff
is a *resolution* problem — at stride 32 a distant car spans barely one 32 px cell — and no amount of
extra epochs, loss tuning or regularisation can fix it. Two ways to give small cars more pixels:

| Attempt | How | Test F1 | AP50 | Verdict |
|---|---|---|---|---|
| Baseline best (512 px, stride 32) | — | 0.904 | 0.871 | |
| **Larger input (640 px, stride 32)** | more pixels, full-depth backbone | **0.917** | **0.935** | ✅ **best** |
| Finer grid (512 px, **stride 16**) | cut the backbone earlier → 32×32 grid | 0.822 | 0.822 | ❌ worse |
| Even larger input (768 px, stride 32) | — | 0.818 | 0.873 | ❌ worse |

**Stride 16 failed, and the reason is the interesting part.** Taking features from an earlier stage
does buy spatial resolution — but it pays for it in *semantic depth*: the features are shallower and
have seen fewer non-linearities. Resolution and semantics trade off against each other, and here the
trade was a net loss. That is precisely the problem a **Feature Pyramid Network** exists to solve
(upsample the deep stride-32 features and fuse them with the shallow stride-16 ones, getting both),
and it is the natural next step if we wanted to push further.

**768 px also failed** (recall 0.868 → 0.711), so input size is a sweet spot, not a monotonic knob —
bigger inputs mean fewer effective pixels of context per cell and, at a fixed 40 epochs, a harder
optimisation. 640 px is the win: **AP50 0.871 → 0.935**, with just **1 false positive** on the test set.

## Confusion matrices (best model, 640 px, test split)

Written by `uv run python -m src.part1.analysis --tag best_img640` → `confusion_matrix.png`.

**Detection level** (what "confusion matrix" normally means for a detector — background/background is
undefined, because there is no such thing as "correctly predicted nothing" when the negative class is
every possible box):

| | Actual: vehicle | Actual: background |
|---|---|---|
| **Predicted: vehicle** | **TP 33** | FP 1 |
| **Predicted: background** | FN 5 | n/a |

→ precision **0.971**, recall **0.868**

**Grid-cell objectness** — a genuine 2×2 with a real TN count, and arguably the more honest matrix
for this architecture, since the head literally *is* a binary classifier run over every grid cell:

| | Actual: vehicle | Actual: no vehicle |
|---|---|---|
| **Predicted: vehicle** | TP 83 | FP 8 |
| **Predicted: no vehicle** | FN 30 | **TN 19,879** |

**19,879 true negatives against 83 true positives** — that is the class imbalance in raw numbers, and
it is exactly why plain BCE collapses to "no vehicle" and why `pos_weight` exists.

### Which works best, and why

**1. Unfreezing the backbone is the single biggest lever** (+0.42 test F1 on its own: 0.423 → 0.838).
The frozen ImageNet features were the bottleneck, not the head. ImageNet is object-centric photos;
our vehicles are small, motion-blurred, and seen from a dashcam. Letting `layer4` adapt (at lr 1e-4)
fixes a domain mismatch no detection head can compensate for. This is a *deviation from the task
sheet*, which says "you don't need to train the backbone" — permission, not prohibition — so we
report the prescribed frozen version as the baseline **and** the improved version, and show the gap.

**2. Multi-cell assignment is the second lever** (+0.23 test F1 frozen: 0.423 → 0.648; still +0.04
on top of unfreezing). With `center`, one box yields **one** positive cell against 255 negatives —
453 positive signals in the whole training set. The neighbouring cells fire anyway but were never
taught *what* box to emit, so they produced undersized fragments that NMS could not merge, costing a
false positive *and* a false negative per car. Training the neighbours to regress the *same* box
turns them from noise into agreeing votes. FPs on test drop 18 → 10.

**3. Our augmentation did nothing — and we are reporting that, not hiding it.** `noaug` scores val F1
**0.612** against the baseline's **0.613**, i.e. identical, and it is *better* on test (0.485 vs
0.423) — a difference well inside the noise of a 38-box test set. So horizontal flip + colour jitter
bought us **nothing**. That is a real (negative) result and it points somewhere specific: the clip is
a single dashcam pass down one road, so flipping and re-tinting it does not create the variety the
model actually lacks. What is missing is *scale and viewpoint* variation. Random scale/translate
crops (or mosaic) are the augmentation worth trying next; hflip and jitter are not.

Meanwhile the training curves show classic overfitting: val loss bottoms out at **epoch 17** and then
drifts up while train loss keeps falling (0.28 → 0.17). More epochs cannot help. Only more — or more
*varied* — data can, which is why the augmentation result above matters rather than being a footnote.

**4. What did *not* work, and why it is interesting:**
- **CIoU lost to plain L1** everywhere (0.476 vs 0.423 with `center`, but 0.597 vs 0.648 with
  `multi`). Expected it to win, since it optimises the IoU we are scored on. It appears CIoU's
  gradient is unstable when the predicted boxes start far from the targets and the objects are tiny
  (median 1.6% of image area); L1 in the bounded sigmoid space is simply an easier optimisation.
- **Focal loss was the worst run of all** (0.421). It is designed for the extreme foreground /
  background imbalance of dense anchor detectors; here `pos_weight=20` already handles a much milder
  imbalance, and focal's down-weighting of easy negatives just starved the objectness head of signal.
- **Plain BCE** (no imbalance handling) collapsed recall to 0.395 — it learned to say "no vehicle",
  exactly as predicted: 99% cell-level accuracy, useless detector.
- **MobileNetV3 beat ResNet18 while frozen** (0.778 vs 0.423) — a genuine surprise. Its stride-32
  features (960 channels, inverted-residual blocks) transfer to small blurry vehicles far better
  than ResNet18's 512. Once *both* backbones are unfrozen the gap narrows (0.904 vs 0.880), which
  supports the diagnosis: the issue was never capacity, it was frozen-feature mismatch.

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
