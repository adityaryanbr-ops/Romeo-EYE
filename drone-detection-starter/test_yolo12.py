from ultralytics import YOLO

# Try to load YOLOv8 instead of YOLOv12 (since YOLOv12 weights are not published yet)
# Change "yolov8s.pt" to your own trained weights path if you have them
model = YOLO("yolov8s.pt")

# Run detection on webcam
results = model.predict(source=0, conf=0.5, show=True)

print("Detection complete")
