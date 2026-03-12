import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load the real e-commerce dataset."""
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
    data["Mois_Num"] = data["Date"].dt.month
    data["Mois_Nom"] = data["Date"].dt.month_name()
    data["Jour_Semaine"] = data["Date"].dt.dayofweek
    data["Jour_Semaine_Nom"] = data["Date"].dt.day_name()
    data["Trimestre"] = data["Date"].dt.quarter
    print("[OK] Time features created")
    return data


def calculate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Compute business KPIs."""
    data = data.copy()
    data["CA"] = data["Prix_Final"]
    data["Revenu_Par_Unite"] = (data["Prix_Final"] / data["Quantite"]).round(2)
    print("[OK] Business metrics calculated")
    return data


def segment_customers(data: pd.DataFrame) -> pd.DataFrame:
    """Segment customers using RFM model."""
    customers = (
        data.groupby("Customer_ID")
        .agg(
            Frequence=("Transaction_ID", "count"),
            Montant_Total=("Prix_Final", "sum"),
            Derniere_Achat=("Date", "max"),
            Quantite_Totale=("Quantite", "sum")
        )
        .reset_index()
    )

    def classify(row):
        if row["Frequence"] >= 10 and row["Montant_Total"] >= 50000:
            return "VIP"
        elif row["Frequence"] >= 5:
            return "Regulier"
        return "Occasionnel"

    customers["Segment"] = customers.apply(classify, axis=1)
    print("\n[>>] CUSTOMER SEGMENTATION")
    print(customers["Segment"].value_counts().to_string())
    return customers


def prepare_ml_data(data: pd.DataFrame):
    """Prepare features for ML model."""
    features = ["Prix_Unitaire", "Quantite", "Mois_Num", "Jour_Semaine"]
    data = data.copy()
    df_clean = data[features + ["CA"]].dropna()
    X = df_clean[features].values
    y = df_clean["CA"].values
    print(f"\n[ML] ML data ready - X: {X.shape}, y: {y.shape} | features: {features}")
    return X, y, features


def display_summary(data: pd.DataFrame) -> None:
    """Print business summary."""
    print("\n" + "=" * 70)
    print("[STAT] BUSINESS SUMMARY - REAL DATASET")
    print("=" * 70)
    print(f"  [SHOP] Transactions      : {len(data):,}")
    print(f"  [>>]  Unique customers   : {data['Customer_ID'].nunique():,}")
    print(f"  [PKG] Unique products    : {data['Produit'].nunique():,}")
    print(f"\n  [EUR] Total revenue      : {data['CA'].sum():>14,.2f}")
    print(f"  [STAT] Avg per transaction: {data['CA'].mean():>13,.2f}")
    print(f"  [UP]  Total profit       : {data['Profit'].sum():>14,.2f}")
    print(f"  [ROI] Avg margin         : {data['Marge_%'].mean():>13.2f} %")
    print(f"\n  [TOP] Top category       : {data['Categorie'].value_counts().idxmax()}")
    print(f"  [TOP] Top customer       : {data['Customer_ID'].value_counts().idxmax()}")
    print("=" * 70)


if __name__ == "__main__":
    print("[OK] preprocessing module loaded")











