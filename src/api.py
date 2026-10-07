from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "purchase_intent_model.joblib"
THRESHOLD_PATH = ARTIFACT_DIR / "decision_threshold.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not THRESHOLD_PATH.exists():
    raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

model = joblib.load(MODEL_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))

app = FastAPI(
    title="E-Commerce Purchase Intent Prediction API",
    version="1.0",
)


class SessionData(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": round(threshold, 2),
    }


@app.get("/")
def root():
    return {
        "message": "E-Commerce Purchase Intent Prediction API",
        "docs": "/docs",
    }


@app.post("/predict")
def predict(session: SessionData):
    data = pd.DataFrame([session.model_dump()])

    data["TotalPages"] = (
        data["Administrative"]
        + data["Informational"]
        + data["ProductRelated"]
    )
    data["TotalDuration"] = (
        data["Administrative_Duration"]
        + data["Informational_Duration"]
        + data["ProductRelated_Duration"]
    )
    data["AvgTimePerPage"] = data["TotalDuration"] / (data["TotalPages"] + 1)
    data["ProductEngagementRatio"] = data["ProductRelated"] / (data["TotalPages"] + 1)
    data["ProductTimeRatio"] = data["ProductRelated_Duration"] / (data["TotalDuration"] + 1)
    data["ReturningVisitorFlag"] = data["VisitorType"].eq("Returning_Visitor").astype(int)

    probability = float(model.predict_proba(data)[0, 1])
    prediction = int(probability >= threshold)
    label = "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"

    return {
        "prediction": prediction,
        "label": label,
        "purchase_probability": round(probability, 4),
        "purchase_probability_percent": round(probability * 100, 2),
        "decision_threshold": round(threshold, 2),
    }
