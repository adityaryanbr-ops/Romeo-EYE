import argparse, yaml, time
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO checkpoint or cfg (e.g., yolov8n.pt)')
    p.add_argument('--data', type=str, default='configs/data.yaml', help='YOLO data.yaml')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--img', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--name', type=str, default='drone')
    p.add_argument('--hyp', type=str, default='configs/hyperparams.yaml')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    model = YOLO(args.model)
    print(f"[i] Training {args.model} for {args.epochs} epochs @ {args.img}px ...")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        workers=args.workers,
        name=args.name,
        verbose=True,
        cfg=args.hyp
    )

    print('[✓] Training complete. Check runs/detect/ for artifacts.')

if __name__ == '__main__':
    main()
