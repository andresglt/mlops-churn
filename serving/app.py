import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# Configurar tracking (URI y token por variables de entorno en Render)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
MLMODEL_URI = os.environ.get("MLMODEL_URI", "models:/churn-model/Production")

# Carga modelo como PyFunc
model = mlflow.pyfunc.load_model(MLMODEL_URI)

app = FastAPI(title="Churn Model API")

class Record(BaseModel):
    features: Dict[str, Any]

class Batch(BaseModel):
    records: List[Record]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(batch: Batch):
    df = pd.DataFrame([r.features for r in batch.records])
    preds = model.predict(df)
    # Si el modelo devuelve probabilidades, asegurar formato
    try:
        proba = preds if isinstance(preds, (list, tuple)) else preds.tolist()
    except Exception:
        proba = preds
    return {"predictions": proba}