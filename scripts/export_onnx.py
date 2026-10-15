import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--img', type=int, default=640)
    p.add_argument('--opset', type=int, default=12)
    args = p.parse_args()

    model = YOLO(args.weights)
    model.export(format='onnx', imgsz=args.img, opset=args.opset)
    print('[✓] Exported ONNX to the same runs/*/weights folder.')

if __name__ == '__main__':
    main()
