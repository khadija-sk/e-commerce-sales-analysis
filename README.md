# E-Commerce Sales Analysis & Revenue Prediction

An end-to-end **Data Science and Machine Learning project** built on a real e-commerce dataset containing **6,978 transactions** from an international clothing boutique between 2021 and 2022.

The project combines **data preprocessing, business analysis, customer segmentation, machine learning, and a REST API** to transform raw sales data into actionable insights and revenue predictions.

---

## Project Overview

The project follows an end-to-end Data Science workflow:

**Data Cleaning → Business Analysis → Customer Segmentation → Machine Learning → REST API**

The main objectives are to:

* Understand sales and customer behavior
* Calculate key business performance indicators
* Segment customers based on their purchasing behavior
* Compare different regression algorithms
* Build a model for revenue prediction
* Expose the model through a REST API

---

## Dataset

The project uses a real Kaggle dataset containing sales transactions from an international clothing boutique.

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

## Features

### Data Preprocessing

* Data cleaning with **Pandas**
* Missing-value handling
* Feature engineering
* Date-based feature extraction
* Preparation of data for analysis and machine learning

### Business Analysis

The project calculates and analyzes:

* Revenue
* Profit
* Profit margin
* Sales performance
* Monthly trends
* Customer behavior
* Product performance

### Customer Segmentation

A simplified **RFM-based approach** is used to classify customers into:

* **VIP**
* **Regular**
* **Occasional**

### Machine Learning

The project compares three regression algorithms:

* Linear Regression
* Random Forest
* Gradient Boosting

The models are evaluated using:

* **R²**
* **RMSE**

The final implementation uses **GradientBoostingRegressor** for revenue prediction.

### REST API

The trained model is exposed through a **FastAPI REST API** with:

* Pydantic validation
* Structured endpoints
* Automatic Swagger documentation
* Revenue prediction
* Dataset summaries
* Customer segmentation
* Model information

---

## Machine Learning

### Approach

The dataset is divided using an **80/20 train-test split** with a fixed random state for reproducibility.

The following models were evaluated:

1. Linear Regression
2. Random Forest
3. Gradient Boosting

### Selected Model

The current implementation uses:

```text
GradientBoostingRegressor
```

The model achieved an **R² of approximately 0.99 on the test set**.

### Features Used

The model uses the following features:

* `Prix_Unitaire`
* `Quantite`
* `Mois_Num`
* `Jour_Semaine`

Target variable:

* `CA` — Revenue

### Feature Importance

| Feature    | Importance |
| ---------- | ---------: |
| Quantity   |       0.66 |
| Unit Price |       0.34 |

These values represent the feature importance reported by the model in the current implementation.

---

## REST API

The machine learning model is integrated into a **FastAPI** application.

### API Endpoints

| Method | Endpoint      | Description                       |
| ------ | ------------- | --------------------------------- |
| GET    | `/`           | API information                   |
| GET    | `/summary`    | Dataset summary and business KPIs |
| POST   | `/predict`    | Predict revenue                   |
| GET    | `/segments`   | Customer segmentation             |
| GET    | `/model/info` | Model information                 |

### Example Prediction

**Request:**

```json
{
  "prix_unitaire": 1000.0,
  "quantite": 2.0
}
```

**Response:**

```json
{
  "predicted_revenue": 2034.57,
  "algorithm": "gradient_boosting"
}
```

---

## Swagger Documentation

FastAPI automatically provides interactive API documentation.

After starting the application, open:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface allows you to test the API endpoints directly from the browser.

---

## Project Structure

```text
data_project/
├── data/
│   ├── dataset.csv
│   └── dataset_real.csv
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── utils.py
│   ├── main.py
│   └── analysis.ipynb
│
└── requirements.txt
```

### Main Files

| File               | Description                                  |
| ------------------ | -------------------------------------------- |
| `preprocessing.py` | Data cleaning and preprocessing              |
| `model.py`         | Machine learning models and prediction logic |
| `utils.py`         | Utility functions                            |
| `main.py`          | FastAPI application                          |
| `analysis.ipynb`   | Exploratory data analysis and visualizations |
| `requirements.txt` | Python dependencies                          |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/khadija-sk/e-commerce-sales-analysis.git
```

Navigate to the project directory:

```bash
cd e-commerce-sales-analysis
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the API

Start the FastAPI server:

```bash
uvicorn src.main:app --reload
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

Launch the Jupyter Notebook:

```bash
jupyter notebook src/analysis.ipynb
```

The notebook contains the exploratory analysis, visualizations, data processing, and machine learning workflow.

---

## Technologies

### Programming & Data

* Python
* Pandas
* Scikit-learn
* Matplotlib

### Machine Learning

* Linear Regression
* Random Forest
* Gradient Boosting
* R²
* RMSE

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Tools

* Jupyter Notebook
* Git
* GitHub

---

## Project Objective

This project was developed to practice a complete **Data Science and Machine Learning workflow**, from raw transactional data to a model exposed through a REST API.

The focus was not only on building a predictive model, but also on:

* Understanding the underlying business data
* Extracting meaningful KPIs
* Segmenting customers
* Evaluating different models
* Building a reusable API
* Documenting the resulting system

---

## Author

**Khadija Sayoukh**

Engineering Student — Digital Transformation & Artificial Intelligence

[GitHub](https://github.com/khadija-sk) · [LinkedIn](https://www.linkedin.com/in/khadija-sayoukh-1a1a94288)
