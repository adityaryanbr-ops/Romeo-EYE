# Drone Detection – End‑to‑End Starter

This repository gives you a production‑ready template to **train, evaluate, and deploy** a drone detector using the Ultralytics YOLO family (v8/v11). You supply images + annotations; this repo handles the rest.

## Features
- ⚙️ **Training** with Ultralytics YOLO (simple, fast, reproducible)
- 🧪 **Validation**/**mAP** reporting + confusion matrix
- 🚀 **Inference**:
  - batch: images / folders / videos
  - **RTSP/USB webcam** live detection
  - **FastAPI** server for programmatic inference
- 📦 **Export** to ONNX + (optionally) TensorRT
- 🐳 **Dockerfile** for consistent deployment
- 🧰 Utility scripts: dataset split, YOLO data config, basic converters

> Works on Windows / Linux / macOS. For GPU, install CUDA per your system and use a compatible PyTorch build.

---

## 1) Install
```bash
# (Recommended) Create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt
```

## 2) Prepare your dataset
Organize images + labels in YOLO format:

```
datasets/drone_yolo/
├── images/
│   ├── train/  *.jpg|*.png
│   ├── val/    *.jpg|*.png
│   └── test/   *.jpg|*.png (optional)
└── labels/
    ├── train/  *.txt  (YOLO txt per image)
    ├── val/    *.txt
    └── test/   *.txt
```

Each `labels/*.txt` file contains lines:  
`<class_id> <x_center> <y_center> <width> <height>` normalized to 0..1.

Update `configs/data.yaml`:
```yaml
path: datasets/drone_yolo
train: images/train
val: images/val
test: images/test
names: [drone]  # or multiple, e.g. [quadcopter, fixed_wing, bird]
```

If you have a single bucket of images/labels, run the splitter:
```bash
python src/data/split_dataset.py   --images_dir datasets/raw/images   --labels_dir datasets/raw/labels   --out_dir datasets/drone_yolo --val 0.2 --test 0.0
```

## 3) Train
```bash
# Quick start with small model (good baseline)
python src/train.py --model yolov8n.pt --data configs/data.yaml --epochs 100 --img 640 --batch 16
```

Artifacts will appear under `runs/detect/<exp_name>` with `weights/best.pt`.

## 4) Validate
```bash
python src/eval/evaluate.py --weights runs/detect/exp/weights/best.pt --data configs/data.yaml --img 640
```

## 5) Inference (batch)
```bash
# Images folder or a single image/video file
python src/inference/detect.py --weights runs/detect/exp/weights/best.pt --source examples/images --conf 0.5
```

## 6) Live stream (RTSP/USB)
```bash
# RTSP
python src/inference/stream.py --weights runs/detect/exp/weights/best.pt --rtsp rtsp://user:pass@cam-ip:554/stream1

# USB cam index (0 by default)
python src/inference/stream.py --weights runs/detect/exp/weights/best.pt --camera 0
```

## 7) FastAPI server
```bash
uvicorn src.inference.server:app --host 0.0.0.0 --port 8000
# POST /predict with an image -> JSON detections
```

## 8) Export to ONNX
```bash
python scripts/export_onnx.py --weights runs/detect/exp/weights/best.pt --img 640 --opset 12
```

## 9) Docker
```bash
docker build -t drone-detector:latest -f docker/Dockerfile .
docker run --rm -it -p 8000:8000 -v $PWD:/app drone-detector:latest
```

---

## Tips for better accuracy
- Annotate hard negatives (birds, kites) as a separate class if often confused.
- Add **mosaic**/**mixup** augmentations (Ultralytics has strong defaults).
- Train at a resolution close to your deployment input (e.g., 640 or 960).
- Curate a val set from the same **domain** (camera, altitude, lighting).

## License
MIT for this scaffold. Your dataset/models remain yours.
