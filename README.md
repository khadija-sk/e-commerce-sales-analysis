# E-Commerce Sales Analysis & Revenue Prediction

An end-to-end **Data Science and Machine Learning project** built on a real e-commerce dataset containing **6,978 transactions** from an international clothing boutique between 2021 and 2022.

The project combines **data preprocessing, business analysis, customer segmentation, machine learning, and a REST API** to turn raw sales data into usable insights and predictions.

---

## Project Overview

The goal of this project is to analyze e-commerce sales performance and build a machine learning model capable of predicting revenue.

The project covers the complete workflow:

**Raw Data → Cleaning → Analysis → Customer Segmentation → Machine Learning → REST API**

---

## Key Features

* Data cleaning and preprocessing with **Pandas**
* Feature engineering for sales analysis
* Business KPIs:

  * Revenue
  * Profit
  * Profit margin
  * Sales performance
* Customer segmentation using simplified **RFM analysis**

  * VIP
  * Regular
  * Occasional
* Revenue prediction using **Gradient Boosting**
* Comparison of multiple regression algorithms
* Interactive exploratory analysis with **Jupyter Notebook**
* REST API built with **FastAPI**
* Input validation with **Pydantic**
* Automatic API documentation with **Swagger UI**

---

## Dataset

The project uses a real Kaggle dataset containing:

| Metric         |                   Value |
| -------------- | ----------------------: |
| Transactions   |                   6,978 |
| Customers      |                      59 |
| Products       |                      45 |
| Period         | April 2021 – March 2022 |
| Total Revenue  |               5,584,334 |
| Total Profit   |             2,512,950.3 |
| Average Margin |                   45.0% |

The repository also contains a synthetic dataset used during development and testing.

---

## Data Analysis

The analysis focuses on understanding sales performance through:

* Revenue trends
* Profit and margin analysis
* Product performance
* Customer behavior
* Monthly and weekly patterns
* Customer segmentation

### Customer Segmentation

A simplified RFM-based approach is used to classify customers into three groups:

* **VIP** — high-value customers
* **Regular** — recurring customers with moderate activity
* **Occasional** — lower-frequency customers

---

## Machine Learning

Three regression algorithms were evaluated:

* Linear Regression
* Random Forest
* Gradient Boosting

The models were evaluated using:

* **R²**
* **RMSE**

An **80/20 train-test split** with a fixed random state was used to make the evaluation reproducible.

### Selected Model

The current implementation uses **GradientBoostingRegressor** for revenue prediction.

The model achieved an **R² of approximately 0.99 on the test set**.

### Features

The model uses:

* `Prix_Unitaire`
* `Quantite`
* `Mois_Num`
* `Jour_Semaine`

Target:

* `CA` — revenue

### Feature Importance

| Feature    | Importance |
| ---------- | ---------: |
| Quantity   |       0.66 |
| Unit Price |       0.34 |

These values represent the model's feature importance in the current implementation.

---

## REST API

The machine learning model is exposed through a **FastAPI REST API**, making it possible to use the prediction system outside the notebook.

### Main Routes

| Method | Endpoint      | Description              |
| ------ | ------------- | ------------------------ |
| GET    | `/`           | API information          |
| GET    | `/summary`    | Dataset summary and KPIs |
| POST   | `/predict`    | Predict revenue          |
| GET    | `/segments`   | Customer segmentation    |
| GET    | `/model/info` | Model information        |

### Example Prediction

**Request**

```json
{
  "prix_unitaire": 1000.0,
  "quantite": 2.0
}
```

**Response**

```json
{
  "predicted_revenue": 2034.57,
  "algorithm": "gradient_boosting"
}
```

---

## API Documentation

Once the API is running, interactive documentation is available through FastAPI's Swagger interface:

```text
http://127.0.0.1:8000/docs
```

This allows the available endpoints to be tested directly from the browser.

---

## Project Structure

```text
e-commerce-sales-analysis/
│
├── data_project/
│   ├── data/
│   │   ├── dataset.csv
│   │   └── dataset_real.csv
│   │
│   ├── src/
│   │   ├── preprocessing.py
│   │   ├── model.py
│   │   ├── utils.py
│   │   ├── main.py
│   │   └── analysis.ipynb
│   │
│   └── requirements.txt
│
└── README.md
```

---

## Technologies

### Data & Machine Learning

* Python
* Pandas
* Scikit-learn
* Matplotlib

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Development & Analysis

* Jupyter Notebook
* Git
* GitHub

---

## Installation

Clone the repository:

```bash
git clone https://github.com/khadija-sk/e-commerce-sales-analysis.git
```

Navigate to the project:

```bash
cd e-commerce-sales-analysis
```

Install the dependencies:

```bash
pip install -r data_project/requirements.txt
```

---

## Run the API

From the project directory:

```bash
uvicorn data_project.src.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Run the Analysis

Open the Jupyter Notebook:

```bash
jupyter notebook data_project/src/analysis.ipynb
```

The notebook contains the exploratory analysis, visualizations, preprocessing, and machine learning workflow.

---

## Project Objective

This project was built to practice an end-to-end **Data Science workflow**, from raw transactional data to a machine learning model exposed through an API.

It focuses not only on model training, but also on **data understanding, business analysis, reproducibility, and API integration**.

---

## Author

**Khadija Sayoukh**

Engineering Student — Digital Transformation & Artificial Intelligence

* GitHub: https://github.com/khadija-sk
* LinkedIn: https://www.linkedin.com/in/khadija-sayoukh-1a1a94288
