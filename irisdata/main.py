from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model import CLASS_NAMES, FEATURE_NAMES, IrisClassifier

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "artifacts" / "iris_classifier.pth"

app = FastAPI(title="Iris Classifier", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

device = torch.device("cpu")
model = IrisClassifier().to(device)
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found at {WEIGHTS_PATH}. "
        "Run `python train_model.py` or copy your .pth file there."
    )
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=15, description="cm")
    sepal_width: float = Field(..., ge=0, le=15, description="cm")
    petal_length: float = Field(..., ge=0, le=15, description="cm")
    petal_width: float = Field(..., ge=0, le=15, description="cm")


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    probabilities: dict[str, float]
    features: dict[str, float]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": True, "classes": CLASS_NAMES}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: IrisFeatures) -> PredictionResponse:
    features = [
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]
    x = torch.tensor([features], dtype=torch.float32, device=device)

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
    return PredictionResponse(
        predicted_class=CLASS_NAMES[pred_idx],
        predicted_index=pred_idx,
        probabilities=probabilities,
        features={name: value for name, value in zip(FEATURE_NAMES, features)},
    )
