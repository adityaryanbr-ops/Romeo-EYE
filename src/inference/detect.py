import argparse
import csv
import datetime
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to YOLO .pt weights')
    parser.add_argument('--source', default='examples/images', help='Image/Video/Webcam (0) or folder path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--img', type=int, default=640, help='Inference image size')
    parser.add_argument('--save', action='store_true', help='Save annotated results to runs/predict')
    parser.add_argument('--show', action='store_true', help='Show results in window')
    args = parser.parse_args()

    # Load YOLO model
    model = YOLO(args.weights)

    # Create a unique log filename for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"detection_log_{timestamp}.csv")

    # Open log file
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Frame", "Class", "Confidence", "BBox[x1,y1,x2,y2]"])

        # Run inference
        results = model.predict(
            source=args.source,
            conf=args.conf,
            imgsz=args.img,
            save=args.save,
            show=args.show
        )

        # Loop through detections
        for frame in results:
            for box in frame.boxes:
                cls = model.names[int(box.cls)]
                conf = float(box.conf)
                xyxy = box.xyxy.tolist()[0]
                frame_path = getattr(frame, "path", "live_feed")

                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    frame_path,
                    cls,
                    conf,
                    xyxy
                ])

    print(f"[✓] Detection complete. Log saved to {log_file}")

if __name__ == "__main__":
    main()
