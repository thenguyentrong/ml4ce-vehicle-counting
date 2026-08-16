"""
Runs the fine-tuned YOLO detector on a video, tracks vehicles frame-to-frame
with the IoU tracker, counts each track once when its center crosses a
horizontal counting line, and writes an annotated output video.

Run:
  python -m src.count_traffic

Produces:
  outputs/counted_traffic.mp4   -- video with boxes, IDs, counting line, running count
"""
import cv2
from ultralytics import YOLO

from src.tracker import IoUTracker

VIDEO_PATH = "data/traffic_video.mp4"
MODEL_PATH = "runs/detect/runs/detect/vehicle_yolo/weights/best.pt"
OUTPUT_PATH = "outputs/counted_traffic.mp4"
CONF_THRESHOLD = 0.4

# counting line as a fraction of frame height (0.0 = top, 1.0 = bottom)
LINE_Y_FRACTION = 0.6


def main():
    model = YOLO(MODEL_PATH)
    tracker = IoUTracker(iou_threshold=0.3, max_missed_frames=10)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * LINE_Y_FRACTION)

    import os
    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    prev_centers = {}   # track_id -> previous center_y, to detect crossing direction
    count_down = 0      # vehicles crossing top-to-bottom
    count_up = 0        # vehicles crossing bottom-to-top

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []
        for box in results.boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            detections.append([xmin, ymin, xmax, ymax])

        tracks = tracker.update(detections)

        for track in tracks:
            xmin, ymin, xmax, ymax = track.box
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)

            prev_cy = prev_centers.get(track.id)
            if prev_cy is not None and not track.counted:
                if prev_cy < line_y <= cy:
                    count_down += 1
                    track.counted = True
                elif prev_cy > line_y >= cy:
                    count_up += 1
                    track.counted = True
            prev_centers[track.id] = cy

            # only draw the box if this track was actually matched to a
            # detection this frame -- otherwise it's "coasting" on its last
            # known position, which looks like a lagging/ghost box
            if track.missed_frames == 0:
               color = (0, 255, 0) if track.counted else (0, 0, 255)
               cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
               cv2.putText(frame, f"ID {track.id}", (int(xmin), int(ymin) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw counting line and running totals
        cv2.line(frame, (0, line_y), (width, line_y), (255, 255, 0), 2)
        cv2.putText(frame, f"Down: {count_down}  Up: {count_up}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        writer.write(frame)

        if frame_idx % 100 == 0:
            print(f"frame {frame_idx}  down={count_down}  up={count_up}")

    cap.release()
    writer.release()
    print(f"Done. Final counts -- down: {count_down}  up: {count_up}")
    print(f"Total tracks created: {tracker._next_id - 1}  (vs. {count_down + count_up} vehicles counted)")
    print(f"Saved annotated video to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
