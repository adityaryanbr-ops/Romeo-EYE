import argparse, asyncio, base64, json, time
import cv2
import numpy as np
import websockets

def jpg_to_b64(frame, quality=80):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def b64_to_bgr(b64):
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

async def run(url, source, conf, width):
    # open source (0 = default camera, or a path/rtsp/http)
    try:
        src_int = int(source)
        cap = cv2.VideoCapture(src_int)
    except ValueError:
        cap = cv2.VideoCapture(source)

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width * 9 // 16)

    async with websockets.connect(url, max_size=25_000_000) as ws:
        # set confidence on server
        await ws.send(json.dumps({"cmd": "set_conf", "conf": conf}))

        t0, frames = time.time(), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Source ended or not available.")
                break

            # optional resize to keep bandwidth reasonable
            if width and frame.shape[1] > width:
                h = int(frame.shape[0] * (width / frame.shape[1]))
                frame = cv2.resize(frame, (width, h))

            b64 = jpg_to_b64(frame, quality=80)
            if b64 is None:
                continue

            await ws.send(json.dumps({"frame_b64_jpg": b64}))
            msg = await ws.recv()
            data = json.loads(msg)

            # server returns an annotated JPEG
            out = data.get("image_b64_jpg")
            if out:
                img = b64_to_bgr(out)
                # show fps
                frames += 1
                dt = time.time() - t0
                if dt >= 1:
                    fps = frames / dt
                    t0, frames = time.time(), 0
                else:
                    fps = 0
                cv2.putText(img, f"FPS ~ {fps:.1f}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow("Romeo Eye (client view)", img)

            # quit with q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws", help="WebSocket endpoint")
    ap.add_argument("--source", default="0", help="0 (camera), or file path, or rtsp/http url")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold on server")
    ap.add_argument("--width", type=int, default=640, help="resize width before sending")
    args = ap.parse_args()
    asyncio.run(run(args.url, args.source, args.conf, args.width))
