import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    data = pd.read_csv(filepath)
    print(f"[OK] Dataset loaded: {data.shape[0]:,} rows, {data.shape[1]} columns")
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, drop nulls and duplicates."""
    cleaned = data.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])
    before = len(cleaned)
    cleaned = cleaned.dropna().drop_duplicates()
    removed = before - len(cleaned)
    print(f"[OK] Cleaning done: {len(cleaned):,} rows kept ({removed} removed)")
    return cleaned


def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Derive year, month, weekday columns from the Date column."""
    data = data.copy()
    data["Annee"] = data["Date"].dt.year
    data["Mois"] = data["Date"].dt.month
    data["Mois_Nom"] = data["Date"].dt.month_name()
    data["Jour_Semaine"] = data["Date"].dt.dayofweek
    data["Jour_Semaine_Nom"] = data["Date"].dt.day_name()
    print("[OK] Time features created")
    return data


def calculate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Compute business KPIs: revenue, margins, and advertising ROI."""
    data = data.copy()
    data["CA"] = data["Prix_Final"]
    data["Marge_Brute"] = data["Prix_Final"] - data["Cout_Total"]
    data["Marge_Nette"] = data["Profit"]
    data["ROI_Pub_%"] = ((data["Marge_Brute"] - data["Budget_Pub"]) / data["Budget_Pub"]) * 100
    print("[OK] Business metrics calculated")
    return data


def segment_customers(data: pd.DataFrame) -> pd.DataFrame:
    """Segment customers using a simplified RFM model (VIP / Regular / Occasional)."""
    customers = (
        data.groupby("Customer_ID")
        .agg(Frequence=("Transaction_ID", "count"),
             Montant_Total=("Prix_Final", "sum"),
             Derniere_Achat=("Date", "max"))
        .reset_index()
    )

    def classify(row):
        if row["Frequence"] >= 3 and row["Montant_Total"] >= 150:
            return "VIP"
        elif row["Frequence"] >= 2:
            return "Regulier"
        return "Occasionnel"

    customers["Segment"] = customers.apply(classify, axis=1)

    print("\n[>>] CUSTOMER SEGMENTATION")
    print(customers["Segment"].value_counts().to_string())
    return customers


def prepare_ml_data(data: pd.DataFrame):
    """Return feature matrix X, target vector y, and feature names for ML training."""
    features = ["Budget_Pub", "Prix_Unitaire", "Quantite", "Reduction_%"]
    X = data[features].values
    y = data["CA"].values
    print(f"\n[ML] ML data ready — X: {X.shape}, y: {y.shape}  |  features: {features}")
    return X, y, features


def display_summary(data: pd.DataFrame) -> None:
    """Print a high-level business summary to stdout."""
    roi = ((data["Marge_Brute"].sum() - data["Budget_Pub"].sum()) / data["Budget_Pub"].sum()) * 100

    print("\n" + "=" * 70)
    print("[STAT] BUSINESS SUMMARY")
    print("=" * 70)
    print(f"  [SHOP] Transactions      : {len(data):,}")
    print(f"  [>>] Unique customers  : {data['Customer_ID'].nunique():,}")
    print(f"\n  [EUR] Total revenue     : {data['CA'].sum():>12,.2f} €")
    print(f"  [STAT] Avg. revenue      : {data['CA'].mean():>12.2f} €")
    print(f"  [REV] Total profit      : {data['Profit'].sum():>12,.2f} €")
    print(f"  [UP] Total ad spend    : {data['Budget_Pub'].sum():>12,.2f} €")
    print(f"\n  [ROI] Global ROI        : {roi:>11.2f} %")
    print("=" * 70)


if __name__ == "__main__":
    print("[OK] preprocessing module loaded")











