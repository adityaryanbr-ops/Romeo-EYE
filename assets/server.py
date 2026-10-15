# server.py
import base64, io, json, os
from typing import Annotated
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

app = FastAPI()

# ---- Load your models once (adjust paths) ----
# General YOLOv8
MODEL_GENERAL = YOLO("yolov8m.pt")  # or any v8 .pt you have
# Bird vs Drone (your trained)
MODEL_DRONE = YOLO(r"runs/detect/birds_vs_drone_a100/weights/best.pt")
# SARD human
MODEL_SARD  = YOLO(r"runs/detect/sard_humans_a100/weights/best.pt")
# VisDrone pretrained (HuggingFace best.pt you downloaded)
MODEL_VISDRONE = YOLO(r"assets/visdrone/best.pt")

MODELS = {
    "general": MODEL_GENERAL,
    "drone_vs_bird": MODEL_DRONE,
    "sard_human": MODEL_SARD,
    "visdrone": MODEL_VISDRONE,
}

@app.get("/")
def info():
    return {"status": "ok", "ws": "/ws", "models": list(MODELS.keys())}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    model_key = "general"
    conf = 0.4
    model = MODELS[model_key]

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            # update model/conf
            if data.get("cmd") == "set_model":
                mk = data.get("model", "general")
                if mk in MODELS:
                    model_key = mk
                    model = MODELS[mk]
                    await ws.send_text(json.dumps({"event": "model", "value": mk}))
                continue

            if data.get("cmd") == "set_conf":
                try:
                    conf = float(data.get("conf", conf))
                    await ws.send_text(json.dumps({"event": "conf", "value": conf}))
                except Exception:
                    pass
                continue

            # got a frame
            b64 = data.get("frame_b64_jpg")
            if not b64:
                continue

            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")

            # run YOLO
            results = model.predict(img, conf=conf, imgsz=640, verbose=False)
            dets = []
            # draw boxes
            draw = ImageDraw.Draw(img)
            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                for b in r.boxes:
                    xyxy = b.xyxy[0].tolist()
                    cls  = int(b.cls[0].item())
                    confv = float(b.conf[0].item())
                    name = model.names.get(cls, str(cls))
                    dets.append({"cls": name, "conf": confv, "xyxy": xyxy})
                    x1,y1,x2,y2 = xyxy
                    draw.rectangle([x1,y1,x2,y2], outline=(34,197,94), width=3)
                    draw.text((x1,y1-14), f"{name} {confv:.2f}", fill=(255,255,255))

            buff = io.BytesIO()
            img.save(buff, format="JPEG", quality=85)
            out_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

            await ws.send_text(json.dumps({"detections": dets, "image_b64_jpg": out_b64}))
    except WebSocketDisconnect:
        pass
