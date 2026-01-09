import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml
import pathlib

# Carga parámetros
params = yaml.safe_load(open("params.yaml"))
test_size = params["preprocess"]["test_size"]
random_state = params["preprocess"]["random_state"]

# Carga dataset
df = pd.read_csv("data/raw/churn.csv", delimiter=";")

# Objetivo
target = "Churn"
y = df[target].map({"Yes": 1, "No": 0})
#y = df[target].astype(int)
X = df.drop(columns=[target])

# Detecta tipos
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

# Pipelines de transformación
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# Ajusta y transforma
preprocessor.fit(X_train_raw)
X_train = preprocessor.transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# Guarda transformados y objetos en disco
pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
pd.DataFrame(X_train).to_parquet("data/processed/X_train.parquet", index=False)
pd.DataFrame(y_train).to_parquet("data/processed/y_train.parquet", index=False)
pd.DataFrame(X_test).to_parquet("data/processed/X_test.parquet", index=False)
pd.DataFrame(y_test).to_parquet("data/processed/y_test.parquet", index=False)

# Guarda columnas para servir
meta = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
}
pd.Series(meta).to_json("data/processed/meta.json")

print("Preprocesamiento completo.")

import pickle

# Crear carpeta models si no existe
pathlib.Path("models").mkdir(parents=True, exist_ok=True)

# Guardar el preprocesador entrenado
with open("models/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("✅ Preprocesador guardado en models/preprocessor.pkl")