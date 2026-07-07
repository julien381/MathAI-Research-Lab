from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas import Transaction

# C'est cette variable "app" qu'Uvicorn cherche !
app = FastAPI(
    title="MathAI Fraud Detection API",
    version="1.0.0",
    description="API de production pour la détection de fraudes bancaires."
)

# Chemin dynamique vers le modèle sauvegardé
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modèle introuvable à l'emplacement : {MODEL_PATH}")

# Chargement du cerveau de l'IA
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {
        "project": "MathAI Fraud Detection",
        "status": "online",
        "docs": "/docs"
    }

@app.post("/predict")
def predict(transaction: Transaction):
    features = transaction.features

    # Sécurité : le modèle a besoin de ses 30 variables d'entrée
    if len(features) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Format incorrect. Reçu {len(features)} variables, attendu exactement 30."
        )

    # Conversion en tableau NumPy pour traitement
    data = np.array(features).reshape(1, -1)

    # RE-NORMALISATION : Application manuelle du StandardScaler sur Time (index 0) et Amount (index 29)
    # On utilise les approximations issues du dataset pour calibrer l'API en direct
    data[0, 0] = (data[0, 0] - 94813.0) / 47488.0
    data[0, 29] = (data[0, 29] - 88.34) / 250.12

    # Calcul des prédictions
    prediction = model.predict(data)[0]
    probability = float(model.predict_proba(data)[0][1])

    # Retour JSON propre
    return {
        "prediction": int(prediction),  # 0 = Légitime, 1 = Fraude
        "fraud_probability": round(probability, 4)
    }