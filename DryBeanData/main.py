import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model import CLASS_NAMES, FEATURE_NAMES, DryBeanClassifier

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "artifacts" / "drybean_classification_model.pth"
SCALER_PATH = BASE_DIR / "artifacts" / "scaler.json"
DATA_PATH = BASE_DIR / "Data" / "dry_bean.csv"

app = FastAPI(title="Dry Bean Classifier", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

device = torch.device("cpu")
model = DryBeanClassifier().to(device)
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found at {WEIGHTS_PATH}. "
        "Run `python train_model.py` or copy your .pth file there."
    )
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler stats not found at {SCALER_PATH}. Run `python train_model.py`."
    )
_scaler = json.loads(SCALER_PATH.read_text())
SCALER_MEAN = np.array(_scaler["mean"], dtype=np.float32)
SCALER_SCALE = np.array(_scaler["scale"], dtype=np.float32)


class BeanFeatures(BaseModel):
    Area: float = Field(..., gt=0)
    Perimeter: float = Field(..., gt=0)
    MajorAxisLength: float = Field(..., gt=0)
    MinorAxisLength: float = Field(..., gt=0)
    AspectRation: float = Field(..., gt=0)
    Eccentricity: float = Field(..., ge=0, le=1)
    ConvexArea: float = Field(..., gt=0)
    EquivDiameter: float = Field(..., gt=0)
    Extent: float = Field(..., gt=0, le=1)
    Solidity: float = Field(..., gt=0, le=1)
    roundness: float = Field(..., gt=0, le=1)
    Compactness: float = Field(..., gt=0)
    ShapeFactor1: float = Field(..., gt=0)
    ShapeFactor2: float = Field(..., gt=0)
    ShapeFactor3: float = Field(..., gt=0)
    ShapeFactor4: float = Field(..., gt=0, le=1)


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    probabilities: dict[str, float]
    features: dict[str, float]


def features_to_tensor(payload: BeanFeatures) -> torch.Tensor:
    raw = np.array(
        [getattr(payload, name) for name in FEATURE_NAMES],
        dtype=np.float32,
    )
    scaled = (raw - SCALER_MEAN) / SCALER_SCALE
    return torch.tensor([scaled], dtype=torch.float32, device=device)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": True, "classes": CLASS_NAMES}


@app.get("/sample")
def sample(cls: Optional[str] = None) -> dict:
    import csv

    with DATA_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if cls:
        cls = cls.upper()
        rows = [row for row in rows if row["Class"] == cls]
        if not rows:
            raise HTTPException(status_code=404, detail=f"Unknown class: {cls}")
    row = random.choice(rows)
    return {
        "Class": row["Class"],
        **{name: float(row[name]) for name in FEATURE_NAMES},
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: BeanFeatures) -> PredictionResponse:
    x = features_to_tensor(payload)
    try:
        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    probabilities = {
        name: round(float(probs[i].item()) * 100, 2) for i, name in enumerate(CLASS_NAMES)
    }
    features = {name: getattr(payload, name) for name in FEATURE_NAMES}
    return PredictionResponse(
        predicted_class=CLASS_NAMES[pred_idx],
        predicted_index=pred_idx,
        probabilities=probabilities,
        features=features,
    )
