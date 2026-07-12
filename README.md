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

## Usage

```bash
# Part 1 - build the detector by hand
uv run python -m src.part1.train                 # train the head (backbone stays frozen)
uv run python -m src.part1.evaluate              # precision / recall @ IoU 0.5 on the test split
uv run python -m src.part1.visualize             # GT vs prediction, side by side

# Part 2 - detect, track, count
uv run python -m src.part2.convert_to_yolo       # CSV -> YOLO labels + data.yaml
uv run python -m src.part2.finetune_yolo         # fine-tune YOLO11n
uv run python -m src.part2.run_video --video data/traffic.mp4   # -> annotated video + counts

uv run pytest                                    # tracker/counter unit tests
```

## Layout

```
config.py            all paths + hyperparameters
src/data.py          dataset download, CSV parsing, train/val/test split
src/part1/           dataset.py  model.py  losses.py  train.py  infer.py  evaluate.py  visualize.py
src/part2/           convert_to_yolo.py  finetune_yolo.py  tracker.py  counting.py  run_video.py
tests/               tracker + counter unit tests (synthetic detections)
notebooks/           01_explore_data  02_part1  03_part2
docs/                task_spec.md  course_info.md  experiments.md
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

Full tables in [`docs/experiments.md`](docs/experiments.md); the reasoning behind each decision, and
the failure analysis, in [`NOTES.md`](NOTES.md).

## Attribution

The course requires that each code snippet records who wrote it. Every module carries an
`Author:` tag, and functions written by a different team member are tagged individually.
