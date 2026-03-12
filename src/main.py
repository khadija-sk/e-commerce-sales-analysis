

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from preprocessing import (
    load_data, clean_data, create_time_features,
    calculate_metrics, segment_customers, prepare_ml_data, display_summary
)
from model import SalesPredictor

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="E-Commerce Sales Analysis API",
    description="API d'analyse et prediction des ventes - Real Kaggle Dataset",
    version="3.0.0"
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_real.csv')

# ── Startup : load data & train model ─────────────────────────────────────────
try:
    data = load_data(DATA_PATH)
    data = clean_data(data)
    data = create_time_features(data)
    data = calculate_metrics(data)
    X, y, feature_names = prepare_ml_data(data)
    predictor = SalesPredictor(algorithm="gradient_boosting")
    predictor.train(X, y, feature_names=feature_names)
    print("[OK] API ready - using real Kaggle dataset")
except Exception as e:
    print(f"[ERR] Startup error: {e}")
    data = None
    predictor = None


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    prix_unitaire: float = Field(..., gt=0, example=1000.0, description="Prix unitaire du produit")
    quantite: float = Field(default=1.0, gt=0, example=2.0, description="Quantite commandee")

class PredictResponse(BaseModel):
    prix_unitaire: float
    quantite: float
    predicted_revenue: float
    algorithm_used: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "E-Commerce Sales Analysis API",
        "version": "3.0.0",
        "dataset": "Real Kaggle dataset - 6,978 transactions (2021-2022)",
        "endpoints": {
            "GET  /summary":    "Resume statistique global",
            "POST /predict":    "Prediction du CA selon prix et quantite",
            "GET  /segments":   "Segmentation clients RFM",
            "GET  /model/info": "Metriques et importance du modele",
        }
    }


@app.get("/summary", tags=["Statistiques"])
def get_summary():
    """Resume statistique global des ventes reelles."""
    if data is None:
        raise HTTPException(status_code=500, detail="Dataset non charge")

    top_products = (
        data.groupby("Produit")["CA"].sum()
        .sort_values(ascending=False)
        .head(3)
        .round(2)
        .to_dict()
    )

    top_customers = (
        data.groupby("Customer_ID")["CA"].sum()
        .sort_values(ascending=False)
        .head(3)
        .round(2)
        .to_dict()
    )

    return {
        "transactions": int(len(data)),
        "clients_uniques": int(data["Customer_ID"].nunique()),
        "produits_uniques": int(data["Produit"].nunique()),
        "periode": {
            "debut": str(data["Date"].min().date()),
            "fin": str(data["Date"].max().date()),
        },
        "finance": {
            "ca_total": round(float(data["CA"].sum()), 2),
            "ca_moyen_par_transaction": round(float(data["CA"].mean()), 2),
            "profit_total": round(float(data["Profit"].sum()), 2),
            "marge_moyenne_%": round(float(data["Marge_%"].mean()), 2),
        },
        "ventes": {
            "quantite_totale": round(float(data["Quantite"].sum()), 0),
            "prix_unitaire_moyen": round(float(data["Prix_Unitaire"].mean()), 2),
        },
        "top_categorie": str(data["Categorie"].value_counts().idxmax()),
        "top_3_produits_ca": top_products,
        "top_3_clients_ca": top_customers,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_revenue(request: PredictRequest):
    """Predit le chiffre d'affaires selon le prix unitaire et la quantite."""
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=500, detail="Modele non entraine")

    try:
        avg_mois = float(data["Mois_Num"].mean())
        avg_jour = float(data["Jour_Semaine"].mean())
        X_input = np.array([[request.prix_unitaire, request.quantite, avg_mois, avg_jour]])
        predicted = float(predictor.model.predict(X_input)[0])

        return PredictResponse(
            prix_unitaire=request.prix_unitaire,
            quantite=request.quantite,
            predicted_revenue=round(predicted, 2),
            algorithm_used=predictor.algorithm
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments", tags=["Clients"])
def get_segments():
    """Segmentation RFM des clients : VIP / Regulier / Occasionnel."""
    if data is None:
        raise HTTPException(status_code=500, detail="Dataset non charge")

    customers = segment_customers(data)
    counts = customers["Segment"].value_counts().to_dict()
    total = len(customers)

    segments = []
    for segment, count in counts.items():
        group = customers[customers["Segment"] == segment]
        segments.append({
            "segment": segment,
            "nombre_clients": int(count),
            "pourcentage": round(count / total * 100, 1),
            "montant_moyen": round(float(group["Montant_Total"].mean()), 2),
            "frequence_moyenne": round(float(group["Frequence"].mean()), 2),
            "quantite_moyenne": round(float(group["Quantite_Totale"].mean()), 2),
        })

    return {"total_clients": total, "segments": segments}


@app.get("/model/info", tags=["Modele"])
def get_model_info():
    """Metriques de performance et importance des features du modele ML."""
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=500, detail="Modele non entraine")

    return {
        "algorithm": predictor.algorithm,
        "dataset": "Real Kaggle dataset",
        "metrics": predictor.metrics,
        "feature_importance": predictor.get_feature_importance(),
    }