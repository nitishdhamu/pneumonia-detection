import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PIL import Image

from backend.config import ServeConfig
from backend.inference import Predictor

# Put your full name here (will appear in OpenAPI description)
AUTHOR_FULL_NAME = "Nitish Dhamu"

app = FastAPI(
    title="Pneumonia Detection API",
    version="1.0",
    description=f"Minor Project - Developed by Nitish Dhamu"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

cfg = ServeConfig()

# Ensure checkpoint exists when starting predictor; Predictor will raise if not.
# Predictor loads model from cfg.model_path and uses CPU-only.
try:
    predictor = Predictor(cfg.arch, cfg.model_path, cfg.img_size)
except Exception as e:
    predictor = None
    # App will still start; endpoints will return friendly error if predictor is not ready.
    start_warning = str(e)

OUT_DIR = "served_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=OUT_DIR), name="static")

@app.get("/health")
def health():
    """
    Health-check endpoint.
    """
    status = {"status": "ok"}
    if predictor is None:
        status["model"] = "not_loaded"
    else:
        status["model"] = "ready"
    return status

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Single-file prediction endpoint.
    Returns JSON: {"label": "Pneumonia"/"Normal", "confidence": float, "gradcam_url": "/static/.."}
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {start_warning if 'start_warning' in globals() else 'no checkpoint'}")

    try:
        data = await file.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        out_name = os.path.splitext(os.path.basename(file.filename))[0] + "_gradcam.png"
        out_path = os.path.join(OUT_DIR, out_name)

        label, conf = predictor.predict_with_cam(img, out_path)
        return {"label": label, "confidence": conf, "gradcam_url": f"/static/{out_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run only if module executed directly
    uvicorn.run(app, host=cfg.host, port=cfg.port)
