from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title='Drone Detector API')

# Lazy global model (load once on first request)
_model = None
WEIGHTS = 'runs/detect/exp/weights/best.pt'  # change as needed

def get_model():
    global _model
    if _model is None:
        _model = YOLO(WEIGHTS)
    return _model

@app.post('/predict')
async def predict(file: UploadFile = File(...), conf: float = 0.5, img: int = 640):
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert('RGB')

    model = get_model()
    results = model.predict(image, conf=conf, imgsz=img, verbose=False)

    dets = []
    for r in results:
        boxes = r.boxes
        names = r.names
        for i in range(len(boxes)):
            b = boxes[i]
            cls_id = int(b.cls)
            dets.append({
                'class_id': cls_id,
                'class_name': names.get(cls_id, str(cls_id)),
                'confidence': float(b.conf),
                'xyxy': [float(x) for x in b.xyxy[0].tolist()]
            })

    return JSONResponse({'detections': dets})
