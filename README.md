# e-commerce-sales-analysis
# E-Commerce Sales Analysis & Prediction

A full end-to-end data science project : data preprocessing, business analytics,
customer segmentation, machine learning, and a REST API built with FastAPI.

---

## Project Structure

```
data_project/
├── data/
│   └── dataset.csv          # 10,000 e-commerce transactions
├── src/
│   ├── generate_dataset.py  # Generates the dataset
│   ├── preprocessing.py     # Data cleaning & feature engineering
│   ├── model.py             # ML model (Gradient Boosting)
│   ├── utils.py             # Charts & summaries
│   ├── main.py              # FastAPI REST API
│   └── analysis.ipynb       # Full analysis notebook
└── requirements.txt
```

---

## Features

- Data cleaning and feature engineering with pandas
- Business KPIs : revenue, profit, ROI, customer segmentation
- Customer segmentation using a simplified RFM model (VIP / Regular / Occasional)
- Sales prediction with Gradient Boosting (scikit-learn)
- REST API with 4 endpoints (FastAPI + Swagger UI)
- Interactive analysis notebook (Jupyter)

---

## Quickstart

**1. Clone the project**
```bash
git clone https://github.com/YOUR_USERNAME/data_project.git
cd data_project
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate the dataset**
```bash
python src/generate_dataset.py
```

**4. Launch the API**
```bash
uvicorn src.main:app --reload
```
Then open http://127.0.0.1:8000/docs to explore the API interactively.

**5. Open the notebook**
```bash
jupyter notebook src/analysis.ipynb
```

---

## API Endpoints

| Method | Endpoint       | Description                          |
|--------|----------------|--------------------------------------|
| GET    | /summary       | Global sales statistics & KPIs       |
| POST   | /predict       | Predict revenue from ad budget       |
| GET    | /segments      | RFM customer segmentation            |
| GET    | /model/info    | Model metrics & feature importance   |

**Example prediction request:**
```json
POST /predict
{
  "budget_pub": 300.0
}
```
```json
{
  "budget_pub": 300.0,
  "predicted_revenue": 87.42,
  "estimated_profit": -212.58,
  "algorithm_used": "gradient_boosting"
}
```

---

## Tech Stack

| Tool         | Usage                        |
|--------------|------------------------------|
| Python 3.14  | Core language                |
| pandas       | Data manipulation            |
| scikit-learn | Machine learning             |
| matplotlib   | Data visualization           |
| FastAPI      | REST API                     |
| Jupyter      | Interactive analysis         |

---

## Author : khadija sayoukh 

**Your Name**
[LinkedIn](www.linkedin.com/in/khadija-sayoukh-1a1a94288) | [GitHub](https://github.com/khadija-sk)
