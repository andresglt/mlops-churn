from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI()

# 📦 Cargar el modelo entrenado
MODEL_PATH = "models/best_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print(f"❌ Modelo no encontrado en {MODEL_PATH}")

# 🧾 Definir el esquema de entrada
class ChurnInput(BaseModel):
    age: int
    income: float
    contract_type: str
    tenure_months: int
    has_internet: bool
    num_support_calls: int

# 🔮 Endpoint de predicción
@app.post("/predict")
def predict_churn(data: ChurnInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    # Convertir entrada a DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predecir
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {str(e)}")

    return {
        "churn_prediction": bool(prediction),
        "churn_probability": round(probability, 4)
    }