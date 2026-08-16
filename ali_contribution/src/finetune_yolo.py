"""
Fine-tunes a small pretrained YOLO detector (YOLOv8-nano) on the vehicle
dataset prepared by prepare_yolo_dataset.py.

Requires: pip install ultralytics

Run:
  python -m src.prepare_yolo_dataset      # once, to build data/yolo_dataset/
  python -m src.finetune_yolo             # fine-tune

Produces a trained checkpoint at:
  runs/detect/vehicle_yolo/weights/best.pt
"""
from ultralytics import YOLO

DATA_YAML = "data/yolo_dataset/data.yaml"
EPOCHS = 30
IMG_SIZE = 640
BASE_MODEL = "yolov8n.pt"   # "nano" -- smallest/fastest YOLOv8 variant, downloaded automatically


def main():
    model = YOLO(BASE_MODEL)  # loads pretrained COCO weights
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project="runs/detect",
        name="vehicle_yolo",
        exist_ok=True,
    )
    # quick validation summary (precision/recall/mAP) on the val split
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()
