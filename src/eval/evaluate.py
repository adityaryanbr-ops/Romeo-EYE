import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--data', type=str, default='configs/data.yaml')
    p.add_argument('--img', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    args = p.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.img, conf=args.conf)
    print('[✓] Validation summary:')
    print(metrics)

if __name__ == '__main__':
    main()
