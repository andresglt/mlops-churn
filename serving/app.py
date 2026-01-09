import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# Ruta del modelo versionado con DVC
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")

# Cargar el modelo desde archivo local
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Modelo cargado desde {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"❌ Modelo no encontrado en {MODEL_PATH}. Asegúrate de ejecutar 'dvc pull' antes de iniciar el servicio.")

app = FastAPI(title="Churn Model API")

class Record(BaseModel):
    features: Dict[str, Any]

class Batch(BaseModel):
    records: List[Record]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(batch: Batch):
    if model is None:
        return {"error": "Modelo no cargado"}
    df = pd.DataFrame([r.features for r in batch.records])
    preds = model.predict(df)
    # Asegurar formato de salida
    try:
        proba = preds if isinstance(preds, (list, tuple)) else preds.tolist()
    except Exception:
        proba = preds
    return {"predictions": proba}

