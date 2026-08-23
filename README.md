# Vehicle detection & counting — ML4CE semester project (Topic 2)

Machine Learning for Civil Engineering, RWTH Aachen University — summer semester 2026.

**Team:** The Vinh Nguyen Trong, Ali Awada, Rexhep Azemi

- **Part 1 — detector from scratch.** ImageNet-pretrained CNN backbone plus a small single-class
  head. For each cell of a 16×16 grid it predicts one objectness score and four box values. The
  point is to understand a detector, not to win a benchmark.
- **Part 2 — counting in video.** Fine-tuned YOLO-nano feeding a tracker written from scratch (IoU
  association + Hungarian matching). A vehicle is counted once, when its box center crosses a
  line, and the count is compared against a manual one.

Task description: [`docs/task_spec.md`](docs/task_spec.md). Deadlines and submission:
[`docs/course_info.md`](docs/course_info.md).

## Setup

Needs [`uv`](https://docs.astral.sh/uv/) and an NVIDIA GPU (CUDA 12.6 wheels, CPU works but is
slow). Python 3.12, not 3.13+ — no stable torch/ultralytics wheels there yet.

```bash
uv sync --python 3.12          # creates .venv and installs everything
uv run python -m src.data      # downloads the Kaggle dataset into data/
```

Everything stays in this folder: venv, dataset, checkpoints, videos. All paths and
hyperparameters are in [`config.py`](config.py), nothing is hard-coded elsewhere.

The data is not committed (`data/` is gitignored). `src/data.py` downloads the Kaggle images +
CSV and builds the split in `make_splits()` — sorted by time, 80/15/5, so the **last 50 images
are the test set**. Part 2 reuses the same splits; the video comes from
`python -m src.part2.video`.

## Usage

```bash
# Part 1 — build the detector by hand
uv run python -m src.part1.train                 # train the head (backbone frozen)
uv run python -m src.part1.evaluate              # precision / recall @ IoU 0.5 on the test split
uv run python -m src.part1.visualize             # GT vs prediction, side by side

# Part 2 — detect, track, count
uv run python -m src.part2.video                 # fetch the traffic video (not committed)
uv run python -m src.part2.yolo_data             # CSV -> YOLO labels + data.yaml
uv run python -m src.part2.finetune              # fine-tune YOLO11n
uv run python -m src.part2.run_count             # count, with the fine-tuned weights
uv run python -m src.part2.run_count --weights stock --match greedy   # comparison runs
uv run python -m src.part2.manual_count          # clip for the manual count
uv run python -m src.part2.evaluate              # automatic vs manual, failure diagnostics

uv run pytest                                    # tracker / counter unit tests
```

### On unseen data

The course tests the model on a separate set after submission, so both parts run on data from
anywhere without a code edit:

```bash
# Part 1 — a folder of images
uv run python -m src.part1.predict --images path/to/images
uv run python -m src.part1.predict --images path/to/images --csv ground_truth.csv  # + scoring

# Part 2 — a new video
uv run python -m src.part2.suggest_line --video new.mp4    # where the line belongs
uv run python -m src.part2.run_count --video new.mp4 --line 0.45,0.25,0.45,1
```

`predict.py` takes the best checkpoint and the threshold tuned on validation, and writes
`predictions.csv` in original-image pixels, annotated images, and metrics if a ground truth CSV is
passed. On the project's own test split it gives the same numbers as `evaluate.py`.

`run_count` and `suggest_line` use the fine-tuned weights when they are there, and read the
**whole** video — only the project clip is trimmed, to the 60 s the manual count covers. What
transfers to a new video and what does not: [`docs/unseen_video.md`](docs/unseen_video.md).

## Layout

```
config.py            all paths + hyperparameters
make_submission.py   packs code + weights + docs + output video for gigamove
src/data.py          dataset download, CSV parsing, train/val/test split
src/eda.py           dataset montages (labeled / unlabeled frames)
src/part1/           dataset.py  model.py  losses.py  train.py  infer.py  evaluate.py
                     predict.py  visualize.py  analysis.py
src/part2/           video.py  yolo_data.py  finetune.py  tracker.py  counter.py
                     run_count.py  suggest_line.py  manual_count.py  evaluate.py
tests/               tracker + counter unit tests (synthetic detections)
ali_contribution/    Ali's own implementation of both parts, kept separate (see Attribution)
docs/                task_spec.md  course_info.md  experiments.md
                     manual_count.md  crossing_audit.md  unseen_video.md
NOTES.md             lab notebook: what we tried, what worked, what did not
runs/                checkpoints, figures, videos   (git-ignored)
data/                dataset + traffic video        (git-ignored)
```

## Results

**Part 1.** Test precision/recall at IoU ≥ 0.5, temporal split. Score threshold and NMS IoU tuned
on validation. 12 configurations in [`docs/experiments.md`](docs/experiments.md). All runs use the
**512 × 512 input and 16 × 16 grid from the task sheet**.

| Configuration | Precision | Recall | F1 | AP50 |
|---|---|---|---|---|
| **Best** — MobileNetV3, multi-cell assign, fine-tuned backbone | **0.943** | **0.868** | **0.904** | **0.871** |
| Task-sheet baseline — frozen ResNet18, one positive cell per box | 0.455 | 0.395 | 0.423 | 0.213 |

`python -m src.part1.train` without flags reproduces the **task-sheet baseline**; every deviation
is an explicit flag. Two of them explain almost the whole gain:

- **Unfreezing the backbone** (+0.42 F1 alone). Frozen ImageNet features come from object-centric
  photos and do not fit small, motion-blurred vehicles from a dashcam. No head can fix that.
  *The task sheet says the backbone does not need training, so both results are reported.*
- **Multi-cell assignment** (+0.23 F1 alone). One positive cell per box gives only 453 positive
  signals in the whole training set, and the neighbouring cells — which fire anyway — never learn
  a box, so they emit fragments that cost a false positive and a false negative on the same car.
  Training the center cell plus its two nearest neighbours fixes it.

What did **not** work is in [`docs/experiments.md`](docs/experiments.md) too: CIoU lost to plain
L1, focal loss was the worst run, the augmentation changed nothing, and neither a bigger input
(640 / 768 px) nor a finer stride-16 grid found a single extra vehicle. All 5 remaining misses are
cars smaller than 1% of the image — that needs a feature pyramid, or more data.

Two things to know before reading the numbers:

- **The dataset is one video.** All 1001 images are frames of a single clip, 20 frames (~0.67 s)
  apart. A random split puts near-identical frames on both sides and pushes val F1 from 0.55 to
  0.78. The split runs along the time axis instead, so every number above is the honest one.
  `config.SPLIT_MODE` switches between both, so the effect can be shown.
- **The annotations are incomplete.** Some frames label one car and leave other visible cars
  unboxed. A correct detection there counts as a false positive, so the measured precision is a
  lower bound.

**Part 2.** Fine-tuned YOLO11n + the tracker + line counting, on 60 s of a static intersection
video (1798 frames). The detector gets **F1 0.984 / AP50 0.992** on the same test split where the
hand-built head gets 0.904.

| Detector | Matching | toward camera | away | **total** | tracks created |
|---|---|---|---|---|---|
| fine-tuned | Hungarian | 32 | 15 | **47** | 408 |
| fine-tuned | greedy | 32 | 15 | **47** | 412 |
| off-the-shelf COCO | Hungarian | 18 | 11 | **29** | 610 |
| off-the-shelf COCO | greedy | 18 | 11 | **29** | 621 |

**Against the manual count** — 23 toward, 20 away, 43 total, tallied by hand twice and reaching
the same total both times ([`docs/manual_count.md`](docs/manual_count.md)):

| run | toward | away | total | error |
|---|---|---|---|---|
| fine-tuned, either matcher | **+9** | **−5** | 47 | **+4  (9.3%)** |
| off-the-shelf COCO | −5 | −9 | 29 | −14  (−33%) |

The net +4 is two much larger errors cancelling out: **nine phantom crossings toward the camera
against five real vehicles missed going away** — fourteen wrong, not four. Fragmentation creates
the phantoms (8.7 tracks per counted vehicle) and occlusion at the line hides the misses. This is
the failure mode `NOTES.md` predicted before the split was recorded, and it is why the total on its
own is the wrong number to quote.

Two results came out opposite to the expectation, both explained in [`NOTES.md`](NOTES.md):

- **Fine-tuning helped**, even though Part 1 is dashcam footage and the video is a static street
  camera. The off-the-shelf model emits almost twice as many detections per frame (10.16 vs 5.64)
  and still counts fewer vehicles: it spends them on parked cars and pedestrians that never cross
  the line, and its unstable boxes fragment tracks (21.0 tracks per counted vehicle vs 8.7).
- **Hungarian and greedy tie** (47 vs 47, 29 vs 29). Optimal assignment only pays when several
  detections compete for one track, which sparse traffic rarely produces. The unit tests build the
  case where greedy provably fails; this footage does not.

The hard part of Part 2 was **choosing the video**, not writing the tracker. The real requirement
is not "both directions are visible" but "**one line exists that both flows cross**", and that is
invisible in a still frame. One clip passed every visual check and was dropped after tracking
showed its best line was crossed by 15 of 34 moving tracks, 14 of them the same way. Full story in
[`NOTES.md`](NOTES.md).

## Submission

```bash
uv run python make_submission.py --list      # what goes in, and what is missing
uv run python make_submission.py             # -> submission/ML4CE_Topic2_Nguyen_Awada_Azemi.zip
```

The zip has the code, both sets of weights, the rendered output video, the docs, the metrics of
every run and the source clip. The Kaggle images are left out, `python -m src.data` gets them
back. Slides are taken from `presentation/`, and the script complains when that folder is empty.

Upload to [gigamove](https://gigamove.rwth-aachen.de/en), mail the link to
`ml4cegia@lists.rwth-aachen.de` by **24.08.2026, 23:59**.

## Attribution

The course wants each code snippet to say who worked on it. Every module has an `Author:` tag, and
functions written by someone else in the team are tagged on their own.

**The Vinh Nguyen Trong** — `config.py`, `src/` (both parts), `tests/`, `docs/`, the slides in
`presentation/`, and the manual count of `data/traffic.mp4`.

**Ali Awada** — everything under `ali_contribution/`: a second implementation of both parts written
independently of `src/`, with its own data loader, grid-target encoding, MobileNetV3-Small head,
training loop, YOLOv8-nano fine-tune, IoU tracker and line counter. His write-up is
`ali_contribution/Project_Report.docx`, and the manual count in it (333 vehicles on a motorway clip
filmed from a bridge) is his own.

**Rexhep Azemi** — team member. No file in this repository carries his `Author:` tag; the code
is Vinh's and Ali's, and the tags are the record.

The two implementations are kept apart on purpose instead of being merged into one. They were
written separately, they disagree, and the disagreement is the interesting part: `src/` overcounts
by 9.3% (47 against 43 by hand) while `ali_contribution/` undercounts by roughly 6x (46 against
333). Same task, same Kaggle data, different video and different choices — the comparison is in
[`NOTES.md`](NOTES.md). Merging them would have hidden that.
