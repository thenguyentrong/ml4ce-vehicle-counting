# Vehicle detection & counting — ML4CE semester project (Topic 2)

Machine Learning for Civil Engineering, RWTH Aachen University — summer semester 2026.

**Team:** The Vinh Nguyen Trong, Azemi Rexhep

Detection of vehicles in images and counting of vehicles in a traffic video, in two parts:

- **Part 1 — a detector built from scratch.** A *frozen* ImageNet-pretrained CNN backbone with a
  small single-class detection head bolted on top. The head predicts, for each cell of a 16×16 grid,
  an objectness score and four bounding-box values. The point is to understand what a detector does
  internally, not to win a benchmark.
- **Part 2 — counting in video.** A fine-tuned YOLO-nano detector feeding a tracker we wrote
  ourselves (IoU association + Hungarian matching), which assigns stable IDs across frames. Vehicles
  are counted once each, as their box center crosses a virtual line, and the automatic count is
  compared against a manual ground-truth count.

The full task description is in [`docs/task_spec.md`](docs/task_spec.md); deadlines and submission
details in [`docs/course_info.md`](docs/course_info.md).

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and an NVIDIA GPU (CUDA 12.6 wheels; falls back to CPU,
slowly). Python 3.12 — *not* 3.13+, which lacks stable torch/ultralytics wheels.

```bash
uv sync --python 3.12          # creates .venv and installs everything
uv run python -m src.data      # downloads the Kaggle dataset into data/
```

Everything — venv, dataset, checkpoints, output videos — stays inside this folder. Paths and
hyperparameters live in a single file, [`config.py`](config.py); no path is hard-coded anywhere else.

**Where the data comes from** (it is *not* committed — `data/` is gitignored and created by the
commands above): `src/data.py` downloads the Kaggle images + CSV and builds the train/val/test
split in `make_splits()` — sorted by time, 80/15/5, so the **last 50 images are the test set**.
Part 2 (`src/part2/yolo_data.py`) reuses those exact splits; the traffic video is fetched by
`python -m src.part2.video`.

## Usage

```bash
# Part 1 - build the detector by hand
uv run python -m src.part1.train                 # train the head (backbone stays frozen)
uv run python -m src.part1.evaluate              # precision / recall @ IoU 0.5 on the test split
uv run python -m src.part1.visualize             # GT vs prediction, side by side

# Part 2 - detect, track, count
uv run python -m src.part2.video                 # fetch the traffic video (not committed)
uv run python -m src.part2.yolo_data             # CSV -> YOLO labels + data.yaml
uv run python -m src.part2.finetune              # fine-tune YOLO11n
uv run python -m src.part2.run_count --weights runs/yolo/finetune/weights/best.pt
uv run python -m src.part2.run_count --weights stock --match greedy   # comparison runs
uv run python -m src.part2.manual_count          # clip for the manual ground-truth count
uv run python -m src.part2.evaluate              # automatic vs manual, failure diagnostics

# Part 2 on an UNSEEN video (the course tests on a held-out set) - no code edit needed
uv run python -m src.part2.suggest_line --video new.mp4    # measure where the line belongs
uv run python -m src.part2.run_count --video new.mp4 --line 0,0.65,1,0.65

uv run pytest                                    # tracker/counter unit tests
```

See [`docs/unseen_video.md`](docs/unseen_video.md) for what transfers to a new video and what
must be re-set.

## Layout

```
config.py            all paths + hyperparameters
src/data.py          dataset download, CSV parsing, train/val/test split
src/part1/           dataset.py  model.py  losses.py  train.py  infer.py  evaluate.py  visualize.py
src/part2/           video.py  yolo_data.py  finetune.py  tracker.py  counter.py
                     run_count.py  suggest_line.py  manual_count.py  evaluate.py
tests/               tracker + counter unit tests (synthetic detections)
notebooks/           01_explore_data  02_part1  03_part2
docs/                task_spec.md  course_info.md  experiments.md
                     manual_count.md  crossing_audit.md  unseen_video.md
NOTES.md             lab notebook: what we tried, what worked, what did not
runs/                checkpoints, figures, rendered videos  (git-ignored)
data/                dataset + traffic video                (git-ignored)
```

## Results

**Part 1.** Test-set precision/recall at IoU ≥ 0.5, temporal split. Score threshold and NMS IoU tuned
on validation; 12 configurations compared in [`docs/experiments.md`](docs/experiments.md).

All runs use the **512 × 512 input and 16 × 16 grid the task sheet specifies**.

| Configuration | Precision | Recall | F1 | AP50 |
|---|---|---|---|---|
| **Best** — MobileNetV3, multi-cell assign, fine-tuned backbone | **0.943** | **0.868** | **0.904** | **0.871** |
| Task-sheet baseline — frozen ResNet18, one positive cell per box | 0.455 | 0.395 | 0.423 | 0.213 |

`python -m src.part1.train` with no flags reproduces the **task-sheet baseline**; every deviation we
tested is an explicit flag. Two of them account for almost all of the gain:

- **Unfreezing the backbone** (+0.42 F1 alone). The frozen ImageNet features — object-centric photos —
  simply do not fit small, motion-blurred vehicles seen from a dashcam, and no detection head can
  compensate. *This deviates from the task sheet's "you don't need to train the backbone", so both
  the frozen and fine-tuned results are reported.*
- **Multi-cell assignment** (+0.23 F1 alone). The prescribed one-positive-cell-per-box rule gives only
  453 positive signals across the whole training set, and leaves the neighbouring cells — which fire
  anyway — untrained, so they emit fragments that cost a false positive *and* a false negative on the
  same car. Training the center cell plus its two nearest neighbours on the same box fixes it.

What did **not** work is reported too, in [`docs/experiments.md`](docs/experiments.md): CIoU lost to
plain L1, focal loss was the worst run of all, our augmentation changed nothing, and neither a larger
input (640 / 768 px) nor a finer stride-16 grid recovered a single additional vehicle — every one of
the 5 remaining misses is a car smaller than 1% of the image. The real fix for those is a feature
pyramid, or more data.

Two findings worth knowing before reading those numbers:

- **The dataset is one video.** All 1001 images are frames of a single clip, 20 frames (~0.67 s)
  apart. A random train/test split leaks near-identical frames across the boundary and inflates
  val F1 from 0.55 to 0.78. We split along the **time axis** instead; every number above is the
  honest one. `config.SPLIT_MODE` switches between the two so the effect can be shown, not asserted.
- **The annotations are incomplete.** Some frames label one car and leave other clearly visible
  cars unboxed. A correct detection on those scores as a false positive, so the measured precision
  is a *lower bound*.

**Part 2.** Fine-tuned YOLO11n + our own IoU tracker + line counting, on 60 s of a static
intersection video (1798 frames). The detector reaches **F1 0.984 / AP50 0.992** on the same Part 1
test split our hand-built head scored 0.904 on.

| Detector | Matching | toward camera | away | **total** | tracks created |
|---|---|---|---|---|---|
| fine-tuned | Hungarian | 32 | 15 | **47** | 408 |
| fine-tuned | greedy | 32 | 15 | **47** | 412 |
| off-the-shelf COCO | Hungarian | 18 | 11 | **29** | 610 |
| off-the-shelf COCO | greedy | 18 | 11 | **29** | 621 |

⚠️ **The manual ground-truth count is not yet recorded**, so no accuracy is claimed above — the
runs are only compared against each other. `src.part2.evaluate` refuses to substitute a run for
truth; see [`docs/manual_count.md`](docs/manual_count.md).

Two results came out opposite to our prediction, both reported in full in
[`NOTES.md`](NOTES.md):

- **Fine-tuning helped**, though the Part 1 data is dashcam footage and the video is a static
  street camera. The off-the-shelf model emits nearly twice as many detections per frame (10.16 vs
  5.64) while counting *fewer* vehicles — it spends them on parked cars and pedestrians that never
  cross the line, and its unstable boxes fragment tracks (21.0 tracks per counted vehicle vs 8.7).
- **Hungarian and greedy matching tie** (47 vs 47, 29 vs 29). Optimal assignment only pays when
  several detections compete for one track, which sparse traffic rarely produces. The unit tests
  construct the case where greedy provably fails; this footage does not generate it often.

The hardest part of Part 2 was **choosing the video**, not writing the tracker. The binding
requirement turned out not to be "both directions are visible" but "**one line exists that both
flows cross**" — which is invisible in a still frame. A clip that passed every visual check was
discarded after tracking showed its best possible line was crossed by 15 of 34 moving tracks, 14 of
them the same way. Full account in [`NOTES.md`](NOTES.md).

Full tables in [`docs/experiments.md`](docs/experiments.md); the reasoning behind each decision, and
the failure analysis, in [`NOTES.md`](NOTES.md).

## Attribution

The course requires that each code snippet records who wrote it. Every module carries an
`Author:` tag, and functions written by a different team member are tagged individually.
