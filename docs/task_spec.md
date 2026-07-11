# Task specification — Topic 2: Vehicle Counting

Extracted verbatim in substance from `Topic_2_Vehicle_Counting/Vehicle Counting Project.docx`
(the official task description). This file is the **authoritative checklist** for the project.

> **Note on the class definition.** The topic slide in `ML4CE_Semesterproject_Introduction_2026.pdf`
> shows a 4-class Roboflow dataset (car / truck / bus / motorcycle) and asks for counts *per class*.
> The DOCX supersedes it: *"a single class is sufficient — you do not need to distinguish between
> cars, trucks and buses"*, and it names a different dataset (Kaggle "Car Object Detection").
> We follow the DOCX: **one class, `vehicle`**.

## Dataset (Part 1)

- Kaggle **"Car Object Detection"**: <https://www.kaggle.com/datasets/sshikamaru/car-object-detection>
- ~1000 images of vehicles on a highway, similar perspective, homogeneous scenes.
- Bounding box annotations in a **single CSV file**.
- The detection problem is deliberately constrained: one object class, reasonably large objects.

## Part 1 — Build the detector yourself

The goal is to **understand how an object detector works internally**. We do *not* train a full
detector; we take a pretrained CNN backbone and build a small single-class detection head on top.

### 1. Conceptualization
- [ ] Choose a pretrained backbone (e.g. ResNet, MobileNetV3). **Do not train the backbone** — train only the head.
- [ ] Decide at which stage to take the feature map. With a **512×512** input and a **stride-32**
      feature map you obtain a **16×16 grid**; every grid cell is responsible for the objects whose
      **center falls into it**.
- [ ] The head predicts **five values per grid cell**: an **objectness score** and **four bounding box
      values** (offset of the box center relative to the cell, plus width and height).

### 2. Data loader
- [ ] Read images and the CSV annotations; resize images to the fixed input size.
- [ ] Convert every ground-truth box into a **grid target**: the cell containing the box center is
      positive, all other cells are negative.
- [ ] Sensible **train / validation / test split** — typical: **80% / 15% / 5%**.

### 3. Training and evaluation
- [ ] **Binary cross-entropy** loss for the objectness output.
- [ ] **Regression loss** (L1 or IoU-based) for the box outputs of the **positive cells only**.
- [ ] **Weight the two loss terms** against each other; tune hyperparameters for robust results.
- [ ] At inference: **threshold** the objectness scores, then apply **non-maximum suppression**.
- [ ] Evaluate on the test set with **precision and recall at IoU ≥ 0.5**.
- [ ] **Visualize** predicted and ground-truth boxes **side by side** for a few test images.

## Part 2 — From images to video

Here we do **not** build the detector ourselves. We fine-tune a small pretrained model from an
established framework and connect it to a tracking algorithm that **we implement ourselves**.

### 4. Conceptualization
- [ ] Fine-tune a **small pretrained model** (e.g. the **nano** variant of a recent YOLO release)
      on the Part 1 dataset, or on another small vehicle dataset of our choice.
      Few epochs suffice — the goal is a detector that works reliably on the traffic video.
- [ ] Run the fine-tuned detector **frame by frame** on a static traffic video; inspect the raw
      detections qualitatively.

### 5. Tracking and counting
- [ ] **Implement a simple tracker ourselves**: match detections of the current frame to the active
      tracks of the previous frame using **IoU** between boxes (**greedy or Hungarian** matching).
- [ ] Assign a **new ID** to every unmatched detection.
- [ ] **Terminate** tracks that have not been matched for a few consecutive frames.
- [ ] Define a **virtual counting line**; count a track **exactly once**, at the moment its **box
      center crosses** the line.

### 6. Evaluation
- [ ] Count the vehicles in the video **manually once** — this is the ground truth.
- [ ] Compare the automatic counts **per direction** against the manual counts.
- [ ] Render an **output video** showing detected boxes, **track IDs**, the **counting line** and the
      **running count**.
- [ ] **Failure cases**: where do ID switches or missed vehicles occur, and how do they affect the
      final count?

## Deliverable requirements (from the course PDF)

- [ ] **Document the code** — explain what functions do.
- [ ] **Comment who took part in each code snippet** (who did what?).
- [ ] Presentation: explain **what we tried and what did and did not work**.
- [ ] Presentation: **compare different methods**, evaluate and explain **which works best and why**.
- [ ] Presentation: **visualize results clearly** (plots, tables, test predictions).
