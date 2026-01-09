import pandas as pd
import numpy as np
import pickle
import json
import pathlib

# Cargar el preprocesador entrenado
with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Cargar metadata de columnas
meta = json.load(open("data/processed/meta.json"))
numeric_cols = meta["numeric_cols"]
categorical_cols = meta["categorical_cols"]

def preprocess_new_data(raw_records):
    """
    raw_records: lista de diccionarios con las variables en bruto
    Ejemplo:
    [{"customerID":"1234","gender":"Male","SeniorCitizen":0,...}]
    """
    # Convertir a DataFrame
    df = pd.DataFrame(raw_records)

    # Asegurar que las columnas estén en el mismo orden
    expected_cols = numeric_cols + categorical_cols
    df = df[expected_cols]

    # Transformar con el preprocesador entrenado
    X = preprocessor.transform(df)

    return X