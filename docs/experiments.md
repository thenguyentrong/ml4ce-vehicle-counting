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

> **The test split is too small to trust on its own.** 5% of 1001 images = 50 images holding only
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
| **mobilenet_multi_unfreeze** | MobileNetV3 | multi | **no** | L1 | pos_weight | 0.922 | 0.943 | 0.868 | **0.904** | **0.871** | 33/2/5 |
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

**Best config improves test F1 from 0.423 → 0.904 and AP50 from 0.213 → 0.871 (4×).**

All runs use the **512 px / stride-32 / 16×16 grid the task sheet specifies**. Larger inputs and a
finer grid were tested and rejected — see the next section.

## Where the remaining misses are — and why we did NOT change the input size

The best model's PR curve is strong (precision ≥ 0.95 out to recall 0.87) but ends in a **cliff**:
~13% of vehicles are never found at *any* confidence threshold. `src/part1/analysis.py` buckets
recall by ground-truth box area (as a **% of the image**, so the buckets stay comparable across
input sizes):

| GT box area | Recall |
|---|---|
| < 0.5% of image (tiny) | **0/1** |
| 0.5 – 1% | **11/15 = 0.73** |
| 1 – 2% | 16/16 = 1.00 |
| 2 – 4% | 2/2 = 1.00 |
| > 4% (large) | 4/4 = 1.00 |

Every one of the 5 missed vehicles sits below 1% of the image area. Above that, recall is **perfect**
(22/22).

**Every miss is a small, distant vehicle; large ones are found perfectly.** That reads like a
resolution problem, so we tested the two obvious fixes. **Both failed**, and we keep them here as
negative results rather than deleting the evidence:

| Attempt | Test F1 | AP50 | TP/FP/FN | Verdict |
|---|---|---|---|---|
| **Task-specified 512 px, stride 32** | **0.904** | 0.871 | 33/2/5 | kept |
| Finer grid: stride 16 (32×32 cells) | 0.822 | 0.822 | 30/5/8 | worse |
| Larger input: 640 px | 0.917 | 0.935 | **33**/1/**5** | see below |
| Larger input: 768 px | 0.889 | 0.911 | 32/2/6 | worse |

**Stride 16 made things worse**, and the reason is the interesting part: taking features from an
earlier stage buys spatial resolution but pays for it in *semantic depth* — shallower features, fewer
non-linearities. Resolution and semantics trade off against each other. That is precisely the problem
a **Feature Pyramid Network** exists to solve (upsample the deep stride-32 features and fuse them with
the shallow stride-16 ones, getting both), and it is the honest next step if this were to be pushed
further.

**640 px did not actually fix anything**, despite the flattering F1/AP50. Look at the counts: it finds
**exactly the same 33 vehicles** as 512 px — identical TP, identical FN. Not one additional small car
was recovered. All it did was drop a single false positive and re-order the confidence ranking (which
is what lifts AP50). On a 38-box test set, one box is ~2.6 points of recall, so 512 / 640 / 768 are
**statistically indistinguishable**. We therefore keep the **512 px the task sheet specifies** — the
larger input bought nothing real, and claiming otherwise on a one-box difference would be dishonest.

> **Method note, learned the hard way.** Our first pass at this table was wrong twice: we read a
> checkpoint from a model that was *still training*, and we bucketed box area in *network-input*
> pixels — which scale with the input size, so the same car lands in a bigger bucket at 640 px and the
> buckets appear to improve when nothing has changed. Both errors flattered the larger input. Areas
> are now a scale-invariant % of the image, and every model is evaluated only after training
> completes.

**The real fix for small objects is not more pixels — it is an FPN, or more data.** That is the
conclusion, and it is better supported than the one we first jumped to.

## Confusion matrices (best model, test split)

Written by `uv run python -m src.part1.analysis --tag mobilenet_multi_unfreeze` → `confusion_matrix.png`.

**Detection level** (what "confusion matrix" normally means for a detector — background/background is
undefined, because there is no such thing as "correctly predicted nothing" when the negative class is
every possible box):

| | Actual: vehicle | Actual: background |
|---|---|---|
| **Predicted: vehicle** | **TP 33** | FP 2 |
| **Predicted: background** | FN 5 | n/a |

→ precision **0.943**, recall **0.868**

**Grid-cell objectness** — a genuine 2×2 with a real TN count, and arguably the more honest matrix
for this architecture, since the head literally *is* a binary classifier run over every grid cell:

| | Actual: vehicle | Actual: no vehicle |
|---|---|---|
| **Predicted: vehicle** | TP 86 | FP 15 |
| **Predicted: no vehicle** | FN 26 | **TN 12,673** |

**12,673 true negatives against 86 true positives** — that is the class imbalance in raw numbers, and
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

## Part 2 — choosing the video

The course did not supply a traffic video, so we sourced one. Four properties had to hold at once
(static camera, two directions, hand-countable density, daylight); the third and fourth are only
checkable by measurement, and the second is the one that looks fine on a still and is not.

Density measured with off-the-shelf yolo11n, sampled every 2 s:

| Candidate | det/frame | mean conf | Verdict |
|---|---|---|---|
| Pexels 2103099 — UK dual carriageway, dusk | **18.7** | 0.551 | too dense to hand-count |
| Pixabay 188613 — US freeway, 5 lanes | 16.2 | 0.504 | too dense; licence forbids redistribution |
| Pexels 854671 — daylight, free-flowing | 7.4 | 0.542 | ideal density, but **single direction** |
| Pexels 4791721 — T-junction | 9.2 | 0.581 | good on paper — see below |
| **Pexels 4791734 — intersection (chosen)** | 8.8 | 0.593 | chosen |

Dusk was *not* the disqualifier we assumed: mean detection confidence at dusk (0.551) is
statistically indistinguishable from daylight (0.542–0.593). Density was.

**The measurement that actually decided it.** "Both directions are visible" is not the
requirement — the requirement is that *one line exists that both flows cross*. We tested that by
running detection + tracking over each clip, collecting every moving track's path, and sweeping
candidate lines:

| Clip | Moving tracks | Best line | Crossings | Direction balance |
|---|---|---|---|---|
| **4791734 intersection** | **93** | horizontal y=0.65 | **+31 / −11 = 42** | **0.35** |
| 4791721 T-junction | 34 | vertical x=0.50 | +1 / −14 = 15 | 0.07 |
| 2103099 dual carriageway | 8 | horizontal y=0.80 | +3 / −0 = 3 | 0.00 |
| 854671 one-way | 22 | horizontal y=0.35 | +12 / −0 = 12 | 0.00 |

We had already selected the T-junction clip on stills and frame inspection — static, daylight,
two-way, sensible density, all true. Tracking it showed the traffic *disperses*: vehicles turn off
in several directions and no single cross-section catches them. Only 15 of 34 moving tracks crossed
the best available line, 14 of them the same way. The intersection clip's best line is crossed by
42 of 93, split 31/11. **The property that mattered was invisible in every still we looked at.**

## Part 2 — detector comparison

Fine-tuned YOLO11n on the Part 1 test split (same images and split as Part 1, via
`src.data.make_splits`), 40 epochs, 640 px:

| Detector | Trained on | P | R | F1 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| **YOLO11n fine-tuned** | Kaggle car-object-detection | **0.974** | **0.994** | **0.984** | **0.992** | 0.688 |
| Part 1 head (ours) | Kaggle car-object-detection | 0.943 | 0.868 | 0.904 | 0.871 | — |

On the video itself, where there is no box-level ground truth, the two detectors are compared
through the count they produce (next section).

## Part 2 — tracking & counting

60 s of `data/traffic.mp4` (1798 frames), counting line at y = 0.65, `TRACK_IOU_THRESH` 0.3,
`TRACK_MAX_AGE` 10, `TRACK_MIN_HITS` 3.

| Detector | Matching | toward camera | away | **total** | tracks created | frag ratio | det/frame |
|---|---|---|---|---|---|---|---|
| fine-tuned | Hungarian | 32 | 15 | **47** | 408 | 8.7 | 5.64 |
| fine-tuned | greedy | 32 | 15 | **47** | 412 | 8.8 | 5.64 |
| off-the-shelf | Hungarian | 18 | 11 | **29** | 610 | 21.0 | 10.16 |
| off-the-shelf | greedy | 18 | 11 | **29** | 621 | 21.4 | 10.16 |

**Manual ground truth: 23 toward, 20 away, 43 total.** Counted by hand on 20.07 (total) and again
on 23.08 (per direction), independently reaching 43 both times.

| run | toward | away | total | error |
|---|---|---|---|---|
| fine-tuned, Hungarian | +9 | −5 | 47 | **+4** |
| fine-tuned, greedy | +9 | −5 | 47 | **+4** |
| off-the-shelf, Hungarian | −5 | −9 | 29 | −14 |
| off-the-shelf, greedy | −5 | −9 | 29 | −14 |

**The net error hides the gross error.** +4 overall is +9 in one direction and −5 in the other:
nine crossings counted that no vehicle made, five vehicles that crossed and were never counted.
Quoting 9.3% alone would overstate how well this works — 14 of 43 vehicles are handled wrongly,
and the two mistakes happen to be of opposite sign. The per-direction split is the only thing that
exposes this, which is why the task sheet asks for it.

**Fine-tuning helped, contrary to our prediction.** We expected it to *hurt*: the Part 1 data is
dashcam footage (rear views, road level) and the video is a static street camera (head-on, elevated),
so fine-tuning on 355 dashcam frames looked like specialising away from the target domain. Instead
the fine-tuned model counts 47 against 29 and creates ~200 fewer tracks. Note it also emits **half
as many detections per frame** (5.64 vs 10.16) while counting *more* vehicles: the off-the-shelf
model spends its extra detections on parked cars, pedestrians and distant traffic that never cross
the line, and its boxes are less stable frame-to-frame, so tracks fragment (frag ratio 21.0 vs 8.7)
and fragments fail to survive `TRACK_MIN_HITS` at the moment of crossing.

**Hungarian vs greedy is a near-tie here — and that is the interesting part.** 47 vs 47 and 29 vs 29;
the only difference is marginally fewer tracks created (408 vs 412, 610 vs 621), i.e. a handful of
ID switches avoided. The theoretical advantage of optimal assignment only materialises when a track
has *several* plausible detections to choose between, and at ~6–10 well-separated vehicles per frame
that ambiguity rarely arises. The unit test `test_hungarian_beats_greedy_on_the_occlusion_case`
constructs the situation where greedy provably fails; this footage just does not produce it often.
On the congested dual-carriageway clip we rejected, it would.

## Part 2 — how our error compares to published systems

Manual ground truth **43**, system **47** → **+4 net = 9.3% over-count**.

Published counting accuracies split into two populations that must not be conflated:

| Source | System | Camera | Count error |
|---|---|---|---|
| Ribeiro & Hirata 2025 ([arXiv 2501.04534](https://arxiv.org/html/2501.04534v1)) | YOLO + visual rhythm | top view | 0.85–1.4% |
| Tashkent 2023 ([PMC10255367](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255367/)) | YOLOv5 + DeepSORT | CCTV **15 m up** | 1.9% |
| **Khazukov et al. 2020** ([J Big Data 7:84](https://doi.org/10.1186/s40537-020-00358-x)) | **YOLOv3 + SORT** | **6 urban intersections, 14–40 m** | **5.5% mean** |
| Fedorov et al. 2019 (J Big Data 6:73) | Faster R-CNN + SORT | urban intersections | MAPE 7.25% |
| Pakdamansavoji et al. 2025 ([arXiv 2511.12342](https://arxiv.org/abs/2511.12342)) | best method, low vantage (<5 m) | GoPro | 9.9–11.0% |
| Majumder & Wilmot 2023 ([PMC10381655](https://pmc.ncbi.nlm.nih.gov/articles/PMC10381655/)) | YOLO, 10 field sites | mixed | ~10% (4.5–23.5%) |
| **Ours** | **YOLO11n + IoU/Hungarian, no Kalman** | **street level** | **9.3%** |
| *Trained humans counting from video* | — | — | *~1.05%, and they under-count* |

**The comparison that matters most.** Mandal & Adu-Gyamfi
([arXiv 2007.16198](https://arxiv.org/abs/2007.16198), *J Big Data Analytics in Transportation* 3,
2021) ablate exactly our architecture choice — same detector, same 546 one-minute clips from 6
DOT cameras, only the tracker changes, scored on genuine count error:

| Tracker | Mean absolute count error | Over-counted in |
|---|---|---|
| **IoU, no Kalman (our design)** | **56.7 pp** | 18 / 18 |
| IoU + Kalman | 17.2 pp | 7 / 18 |
| SORT | 11.7 pp | 18 / 18 |
| Deep SORT | 7.1 pp | 2 / 18 |

Their Kalman-free IoU tracker over-counts by **37–84 percentage points**. Ours over-counts by
**9.3%**, which lands in their *Deep SORT* band with none of Deep SORT's machinery. The direction
of our error (over, not under) matches theory exactly: Bochinski et al. 2017, who introduced the
IoU tracker, note it *"is not able to predict missing detections"* — one dropped frame ends a
track, the vehicle reappears with a new ID, and crosses the line a second time.

**On the camera angle.** Zhang et al. 2024 ([arXiv 2401.07220](https://arxiv.org/abs/2401.07220))
counted the *same* footage with the *same* YOLOv5 + BYTETrack, varying only whether counting
happened in the raw perspective view or a homography-rectified top-down one: **1.81% top-down vs
28.58% perspective, ~16×**, attributed to partial occlusion. Khazukov's "street cameras" are in
fact 14–40 m up at 30–60°. AI City Challenge Track 1 footage is elevated mast-mounted. The FHWA
Traffic Detector Handbook specifies a **40 ft (12 m) minimum camera height** explicitly to control
occlusion. Our street-level angle and 43.5% overlap rate are therefore not a mistake to apologise
for — they are the harder end of a documented spectrum, and the result should be read against the
low-vantage band (~10%), not the top-view band (~1%).

**Caveats we state rather than hide:**

- **Net error hides gross error.** +4 on 43 could be 6 double-counts and 2 misses. A system with 5
  false counts and 5 misses would score 0% net and be badly broken. `docs/crossing_audit.md`
  decomposes it; until that is filled in, 9.3% is a *net* figure only.
- **Quantisation.** At N = 43, one vehicle is 2.33%. Our 9.3% *is* four vehicles. There is no
  meaningful resolution below ~2%, and one clip gives no confidence interval.
- **Not comparable across datasets.** There is no shared protocol; the 98–100% results come from
  authors choosing their own favourable clips. Publication bias here is severe.
- **Our 43.5% overlap figure is our own metric** (pairwise detection IoU > 0.1), not UA-DETRAC's
  occlusion ratio (fraction of a box hidden). The two are not interchangeable.
- **AI City nwRMSE is not a percentage.** Winning 2020 scores (~0.946) do **not** mean 5.4%
  miscount; for a steady error ε, nwRMSE ≈ 1 − 0.707ε, so 0.946 ≈ 7.6%.

**Failure cases:**

1. **Track fragmentation dominates.** 408 tracks for 47 counted vehicles. Most are harmless — parked
   cars and pedestrians that never cross the line — but each fragmentation of a *crossing* vehicle
   risks either a double count (both fragments cross) or a miss (neither is confirmed at the line).
2. **Oversized boxes.** The fine-tuned detector occasionally emits a box several times the size of
   the vehicle, spanning a vehicle plus adjacent background. Its centre is displaced, so it can cross
   the line at the wrong moment or not at all.
3. **The signal cycle.** Traffic stops from ~t=44 s; the count plateaus after frame ~1400. Queued
   vehicles jitter across the line, which the `counted` flag absorbs — the count does not inflate.
4. **Turning traffic at the intersection.** Only ~45% of moving tracks cross the line at all. This is
   not an error, but it makes the counting rule (crossing, not presence) essential to state before
   the manual count, or the human and the machine answer different questions.
