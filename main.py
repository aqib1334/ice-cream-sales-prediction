from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Ice Cream Sales Predictor", version="1.0.0")

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model ---
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None


# --- Request/Response Schemas ---
class PredictRequest(BaseModel):
    DayOfWeek: str   # e.g. "Monday"
    Month: str       # e.g. "April"
    Temperature: float  # °F
    Rainfall: float     # inches

class PredictResponse(BaseModel):
    predicted_sales: int
    inputs: dict


# --- Routes ---
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([{
        "DayOfWeek": req.DayOfWeek,
        "Month": req.Month,
        "Temperature": req.Temperature,
        "Rainfall": req.Rainfall,
    }])

    try:
        result = model.predict(df)
        predicted = max(0, int(round(result[0])))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictResponse(
        predicted_sales=predicted,
        inputs=req.dict()
    )
