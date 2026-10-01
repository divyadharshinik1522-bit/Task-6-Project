from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="Real-Time ML Inference REST API",
    version="1.0.0",
    description="FastAPI service for Iris flower prediction."
)

model = joblib.load("model/model.joblib")

class PredictionRequest(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: list[float]

LABELS = ["setosa", "versicolor", "virginica"]

@app.get("/")
def root():
    return {"message": "ML inference API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    features = np.array([[
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width
    ]])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0].tolist()

    return {
        "predicted_class": prediction,
        "predicted_label": LABELS[prediction],
        "probabilities": [round(float(p), 6) for p in probabilities]
    }
