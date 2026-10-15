import argparse, cv2, time
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--camera', type=int, default=None, help='USB cam index (e.g., 0)')
    p.add_argument('--rtsp', type=str, default=None, help='rtsp://user:pass@ip:port/path')
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--img', type=int, default=640)
    args = p.parse_args()

    src = args.rtsp if args.rtsp is not None else args.camera
    if src is None:
        raise SystemExit('Provide --camera or --rtsp')
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit('Cannot open video source')

    model = YOLO(args.weights)

    while True:
        ok, frame = cap.read()
        if not ok: break
        results = model(frame, imgsz=args.img, conf=args.conf)
        annotated = results[0].plot()  # draw boxes
        cv2.imshow('Drone Detection', annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
