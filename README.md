# E-Commerce Sales Analysis & Prediction

A full end-to-end data science project: data preprocessing, business analytics,
customer segmentation, machine learning, and a REST API built with FastAPI.

> **Real dataset** — 6,978 transactions from an international clothing boutique (2021–2022)

---

## Project Structure

```
data_project/
├── data/
│   ├── dataset.csv           # Synthetic dataset (10,000 transactions)
│   └── dataset_real.csv      # Real Kaggle dataset (6,978 transactions)
├── src/
│   ├── preprocessing.py      # Data cleaning & feature engineering
│   ├── model.py              # ML model (Gradient Boosting)
│   ├── utils.py              # Charts & summaries
│   ├── main.py               # FastAPI REST API
│   └── analysis.ipynb        # Full analysis notebook
└── requirements.txt
```

---

## Features

- Data cleaning and feature engineering with **pandas**
- Business KPIs: revenue, profit, margin, customer segmentation
- Customer segmentation using a simplified **RFM model** (VIP / Regular / Occasional)
- Sales prediction with **Gradient Boosting** (scikit-learn) — R² > 0.99
- REST API with 4 endpoints (**FastAPI** + Swagger UI)
- Interactive analysis notebook (Jupyter)

---

## Quickstart

**1. Clone the project**
```bash
git clone https://github.com/khadija-sk/e-commerce-sales-analysis.git
cd e-commerce-sales-analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the API**
```bash
uvicorn src.main:app --reload
```
Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the API interactively via Swagger UI.

**4. Open the notebook**
```bash
jupyter notebook src/analysis.ipynb
```

---

## API Endpoints

| Method | Endpoint       | Description                              |
|--------|----------------|------------------------------------------|
| GET    | `/`            | API info & available endpoints           |
| GET    | `/summary`     | Global sales statistics & KPIs           |
| POST   | `/predict`     | Predict revenue from price & quantity    |
| GET    | `/segments`    | RFM customer segmentation                |
| GET    | `/model/info`  | Model metrics & feature importance       |

### Example — Predict Revenue

**Request:**
```json
POST /predict
{
  "prix_unitaire": 1000.0,
  "quantite": 2.0
}
```

**Response:**
```json
{
  "prix_unitaire": 1000.0,
  "quantite": 2.0,
  "predicted_revenue": 2034.57,
  "algorithm_used": "gradient_boosting"
}
```

### Example — Summary (GET /summary)
```json
{
  "transactions": 6978,
  "clients_uniques": 59,
  "produits_uniques": 45,
  "periode": { "debut": "2021-04-01", "fin": "2022-03-31" },
  "finance": {
    "ca_total": 5584334.0,
    "profit_total": 2512950.3,
    "marge_moyenne_%": 45.0
  }
}
```

---

## ML Model

The prediction model uses **Gradient Boosting Regressor** from scikit-learn.

| Feature         | Description                        |
|-----------------|------------------------------------|
| `Prix_Unitaire` | Unit price of the product          |
| `Quantite`      | Number of units ordered            |
| `Mois_Num`      | Month extracted from the date      |
| `Jour_Semaine`  | Day of week extracted from the date|

**Target:** `CA` (Chiffre d'Affaires — total revenue per transaction)

Training split: **80% train / 20% test** with `random_state=42`.

---

## Tech Stack

| Tool            | Usage                        |
|-----------------|------------------------------|
| Python 3.10+    | Core language                |
| pandas          | Data manipulation            |
| scikit-learn    | Machine learning             |
| matplotlib      | Data visualization           |
| FastAPI         | REST API                     |
| Pydantic        | Request validation           |
| uvicorn         | ASGI server                  |
| Jupyter         | Interactive analysis         |

---

## Author

**Khadija Sayoukh**

[![LinkedIn](https://www.linkedin.com/in/khadija-sayoukh-1a1a94288)
[![GitHub](https://github.com/khadija-sk)