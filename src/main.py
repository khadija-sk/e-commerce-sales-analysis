"""
main.py - API FastAPI pour l'analyse des ventes e-commerce.

Endpoints:
    GET  /              - Info API
    GET  /summary       - Resume statistique global
    POST /predict       - Prediction des ventes selon le budget pub
    GET  /segments      - Segmentation des clients RFM
    GET  /model/info    - Metriques et importance des features du modele

Lancement:
    uvicorn main:app --reload
Puis ouvre : http://127.0.0.1:8000/docs
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from preprocessing import (
    load_data, clean_data, create_time_features,
    calculate_metrics, segment_customers, prepare_ml_data
)
from model import SalesPredictor

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sales Analysis API",
    description="API d'analyse et prediction des ventes e-commerce",
    version="2.0.0"
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')

# ── Startup : load data & train model ─────────────────────────────────────────
try:
    data = load_data(DATA_PATH)
    data = clean_data(data)
    data = create_time_features(data)
    data = calculate_metrics(data)
    X, y, feature_names = prepare_ml_data(data)
    predictor = SalesPredictor(algorithm="gradient_boosting")
    predictor.train(X, y, feature_names=feature_names)
    print("[OK] API ready")
except Exception as e:
    print(f"[ERR] Startup error: {e}")
    data = None
    predictor = None


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    budget_pub: float = Field(..., gt=0, example=300.0, description="Budget publicitaire en EUR")

class PredictResponse(BaseModel):
    budget_pub: float
    predicted_revenue: float
    estimated_profit: float
    algorithm_used: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Sales Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "GET  /summary":    "Resume statistique global",
            "POST /predict":    "Prediction du CA selon budget pub",
            "GET  /segments":   "Segmentation clients RFM",
            "GET  /model/info": "Metriques et importance du modele",
        }
    }


@app.get("/summary", tags=["Statistiques"])
def get_summary():
    """Resume statistique global des ventes."""
    if data is None:
        raise HTTPException(status_code=500, detail="Dataset non charge")

    roi = ((data["Marge_Brute"].sum() - data["Budget_Pub"].sum()) / data["Budget_Pub"].sum()) * 100

    top_products = (
        data.groupby("Produit")["Prix_Final"].sum()
        .sort_values(ascending=False)
        .head(3)
        .round(2)
        .to_dict()
    )

    return {
        "transactions": int(len(data)),
        "clients_uniques": int(data["Customer_ID"].nunique()),
        "periode": {
            "debut": str(data["Date"].min().date()),
            "fin": str(data["Date"].max().date()),
        },
        "finance": {
            "ca_total": round(float(data["CA"].sum()), 2),
            "ca_moyen_par_transaction": round(float(data["CA"].mean()), 2),
            "profit_total": round(float(data["Profit"].sum()), 2),
            "budget_pub_total": round(float(data["Budget_Pub"].sum()), 2),
            "roi_global_%": round(float(roi), 2),
        },
        "ventes": {
            "quantite_totale": int(data["Quantite"].sum()),
            "prix_moyen": round(float(data["Prix_Final"].mean()), 2),
            "reduction_moyenne_%": round(float(data["Reduction_%"].mean()), 2),
        },
        "top_categorie": str(data["Categorie"].value_counts().idxmax()),
        "top_pays": str(data["Pays"].value_counts().idxmax()),
        "top_3_produits_ca": top_products,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_sales(request: PredictRequest):
    """Predit le chiffre d'affaires pour un budget publicitaire donne."""
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=500, detail="Modele non entraine")

    try:
        predicted = predictor.predict(request.budget_pub)
        profit = round(predicted - request.budget_pub, 2)
        return PredictResponse(
            budget_pub=request.budget_pub,
            predicted_revenue=round(predicted, 2),
            estimated_profit=profit,
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
            "montant_moyen_eur": round(float(group["Montant_Total"].mean()), 2),
            "frequence_moyenne_achats": round(float(group["Frequence"].mean()), 2),
        })

    return {"total_clients": total, "segments": segments}


@app.get("/model/info", tags=["Modele"])
def get_model_info():
    """Metriques de performance et importance des features du modele ML."""
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=500, detail="Modele non entraine")

    return {
        "algorithm": predictor.algorithm,
        "metrics": predictor.metrics,
        "feature_importance": predictor.get_feature_importance(),
    }