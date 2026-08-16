"""
- Decodes grid predictions back into image-space boxes
- Applies objectness thresholding + NMS
- Computes precision/recall @ IoU 0.5 on the test set
- Plots predicted vs. ground-truth boxes for a few test images
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from src.dataset import VehicleDataset, make_splits, IMG_SIZE, GRID, CSV_PATH, IMG_DIR
from src.model import VehicleDetector
from src.utils import nms, box_iou

OBJ_THRESHOLD = 0.5
NMS_IOU = 0.5
EVAL_IOU = 0.5   # IoU threshold to count a prediction as correct


def decode_predictions(pred, obj_threshold=OBJ_THRESHOLD):
    """
    pred: (GRID, GRID, 5) raw model output (logits for objectness).
    Returns boxes (N, 4) in xyxy image-space pixels, and scores (N,).
    """
    cell_size = IMG_SIZE / GRID
    obj_scores = torch.sigmoid(pred[..., 0])

    rows, cols = torch.where(obj_scores > obj_threshold)
    boxes, scores = [], []
    for r, c in zip(rows.tolist(), cols.tolist()):
        off_x, off_y, w, h = pred[r, c, 1:].tolist()
        cx = c * cell_size + off_x * cell_size
        cy = r * cell_size + off_y * cell_size
        w_px = w * IMG_SIZE
        h_px = h * IMG_SIZE
        boxes.append([cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2])
        scores.append(obj_scores[r, c].item())

    if not boxes:
        return torch.empty((0, 4)), torch.empty((0,))
    return torch.tensor(boxes), torch.tensor(scores)


def decode_target(target):
    """Same decoding, but for ground-truth targets (objectness is already 0/1)."""
    cell_size = IMG_SIZE / GRID
    rows, cols = torch.where(target[..., 0] > 0.5)
    boxes = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        off_x, off_y, w, h = target[r, c, 1:].tolist()
        cx = c * cell_size + off_x * cell_size
        cy = r * cell_size + off_y * cell_size
        w_px = w * IMG_SIZE
        h_px = h * IMG_SIZE
        boxes.append([cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2])
    return torch.tensor(boxes) if boxes else torch.empty((0, 4))


def evaluate(model, dataset, device, iou_threshold=EVAL_IOU):
    model.eval()
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for img, target in dataset:
            pred = model(img.unsqueeze(0).to(device))[0].cpu()
            boxes, scores = decode_predictions(pred)
            if boxes.numel() > 0:
                keep = nms(boxes, scores, NMS_IOU)
                boxes = boxes[keep]

            gt_boxes = decode_target(target)

            if boxes.numel() == 0:
                fn += gt_boxes.shape[0]
                continue
            if gt_boxes.numel() == 0:
                fp += boxes.shape[0]
                continue

            ious = box_iou(boxes, gt_boxes)  # (num_pred, num_gt)
            matched_gt = set()
            for i in range(boxes.shape[0]):
                best_j = ious[i].argmax().item()
                if ious[i, best_j] >= iou_threshold and best_j not in matched_gt:
                    tp += 1
                    matched_gt.add(best_j)
                else:
                    fp += 1
            fn += gt_boxes.shape[0] - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def plot_sample(model, dataset, idx, device):
    img, target = dataset[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))[0].cpu()
    boxes, scores = decode_predictions(pred)
    if boxes.numel() > 0:
        keep = nms(boxes, scores, NMS_IOU)
        boxes = boxes[keep]
    gt_boxes = decode_target(target)

    # undo normalization for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    disp_img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(disp_img)
    for box in gt_boxes:
        x0, y0, x1, y1 = box.tolist()
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        linewidth=2, edgecolor="lime", facecolor="none", label="GT"))
    for box in boxes:
        x0, y0, x1, y1 = box.tolist()
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        linewidth=2, edgecolor="red", facecolor="none", label="pred"))
    ax.set_title(f"sample {idx} (green=GT, red=pred)")
    plt.savefig(f"sample_{idx}.png")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_ids = make_splits(CSV_PATH)
    test_ds = VehicleDataset(CSV_PATH, IMG_DIR, test_ids)

    model = VehicleDetector().to(device)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))

    precision, recall = evaluate(model, test_ds, device)
    print(f"Precision @ IoU {EVAL_IOU}: {precision:.3f}")
    print(f"Recall    @ IoU {EVAL_IOU}: {recall:.3f}")

    for i in range(min(5, len(test_ds))):
        plot_sample(model, test_ds, i, device)
    print("Saved sample_*.png visualizations.")
