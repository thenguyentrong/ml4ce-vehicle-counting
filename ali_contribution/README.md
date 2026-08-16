# Vehicle Detection — Part 1 Starter Kit

## 1. Environment setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch torchvision pandas numpy opencv-python pillow matplotlib scikit-learn
```

If you have a GPU, install the CUDA build of torch instead (see pytorch.org/get-started).

## 2. Get the dataset

1. Create a free account at kaggle.com if you don't have one.
2. Go to: https://www.kaggle.com/datasets/sshikamaru/car-object-detection
3. Click "Download" (or use the Kaggle CLI: `kaggle datasets download -d sshikamaru/car-object-detection`).
4. Unzip it into `data/` so you end up with something like:

```
data/
  training_images/
    vid_4_1000.jpg
    vid_4_10000.jpg
    ...
  testing_images/
    ...
  train_solution_bounding_boxes.csv
```

5. Open the CSV and check the columns. For this dataset it's typically:

```
image,xmin,ymin,xmax,ymax
```

One row per bounding box, so one image can appear multiple times (multiple vehicles).
**Check this yourself first** — run `pandas.read_csv(...).head()` and adjust `src/dataset.py`
if the column names differ.

## 3. Files in this starter kit

- `src/dataset.py` — loads images + CSV boxes, converts them into 16×16 grid targets
  (objectness + box offsets), and gives you train/val/test splits.
- `src/model.py` — frozen pretrained backbone (MobileNetV3-Small by default) + a small
  trainable detection head on top.
- `src/utils.py` — box format conversions, IoU, non-max suppression (NMS).
- `src/train.py` — training loop skeleton (loss = BCE for objectness + L1 for boxes).
- `src/visualize.py` — draws predicted vs. ground-truth boxes on a few test images.

## 4. Suggested order to run things

```bash
cd vehicle_project
python -m src.dataset       # sanity check: prints dataset size, plots one sample
python -m src.train         # trains the head, saves checkpoints/best.pt
python -m src.visualize     # shows predictions vs ground truth on test images
```

## 5. What you'll still need to decide/tune yourself

- Which backbone to use (MobileNetV3-Small is fast to iterate with; ResNet18/34 may
  give better features).
- Loss weighting between objectness and box regression.
- Objectness threshold and NMS IoU threshold at inference time.
- Learning rate, batch size, number of epochs.

These are exactly the "hyperparameter tuning" decisions the assignment asks you to
justify — so it's fine (expected) that the starter defaults aren't optimal.
