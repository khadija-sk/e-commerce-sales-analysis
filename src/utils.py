import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import os
import sys

# ── Shared style ─────────────────────────────────────────────────────────────
_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

def _apply_style():
    plt.rcParams.update(_STYLE)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_sales_evolution(data: pd.DataFrame) -> None:
    """Line chart: daily sales over time."""
    _apply_style()
    daily = data.groupby("Date")["Quantite"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["Date"], daily["Quantite"], marker="o", linewidth=2,
            markersize=4, color="#2563EB", label="Ventes")
    ax.set(xlabel="Date", ylabel="Ventes (unites)",
           title="Evolution des ventes dans le temps")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_profit_evolution(data: pd.DataFrame) -> None:
    """Bar chart: daily profit over time."""
    _apply_style()
    daily = data.groupby("Date")["Profit"].sum().reset_index()
    colors = ["#16A34A" if v > 0 else "#DC2626" for v in daily["Profit"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(daily["Date"], daily["Profit"], color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="Date", ylabel="Profit (EUR)", title="Evolution du profit")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_pub_vs_sales(data: pd.DataFrame) -> None:
    """Scatter plot: advertising budget vs revenue with trend line."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data["Budget_Pub"], data["Prix_Final"], alpha=0.3, s=20,
               color="#7C3AED", label="Transactions")
    m, b = np.polyfit(data["Budget_Pub"], data["Prix_Final"], 1)
    x_line = [data["Budget_Pub"].min(), data["Budget_Pub"].max()]
    ax.plot(x_line, [m * x + b for x in x_line], color="orange",
            linewidth=2, linestyle="--", label="Tendance")
    ax.set(xlabel="Budget Publicitaire (EUR)", ylabel="Chiffre d affaires (EUR)",
           title="Relation entre Publicite et Ventes")
    ax.legend()
    plt.tight_layout()
    plt.show()


def display_summary(data: pd.DataFrame) -> None:
    """Print a formatted sales and financial summary."""
    roi = (data["Profit"].sum() / data["Budget_Pub"].sum()) * 100

    print("\n[UP] SALES SUMMARY")
    print("=" * 50)
    print(f"  [PKG] Total units sold     : {data['Quantite'].sum():>10,}")
    print(f"  [STAT] Avg. per transaction: {data['Quantite'].mean():>10.2f}")
    print(f"  [UP] Max per transaction   : {data['Quantite'].max():>10,}")
    print(f"  [DOWN] Min per transaction : {data['Quantite'].min():>10,}")

    print("\n[EUR] FINANCIAL SUMMARY")
    print("=" * 50)
    print(f"  [REV] Total revenue        : {data['Prix_Final'].sum():>12,.2f} EUR")
    print(f"  [EUR] Total profit         : {data['Profit'].sum():>12,.2f} EUR")
    print(f"  [STAT] Avg. profit/tx      : {data['Profit'].mean():>12.2f} EUR")
    print(f"  [AD] Total ad spend        : {data['Budget_Pub'].sum():>12,.2f} EUR")
    print(f"\n  [ROI] Ad ROI               : {roi:>10.2f} %")
    print("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[OK] Testing utils module...\n")

    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import load_data, clean_data, calculate_metrics

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
    data = load_data(data_path)
    data = clean_data(data)
    data = calculate_metrics(data)

    display_summary(data)

    print("\n[STAT] Rendering charts (close each window to continue)...\n")
    plot_sales_evolution(data)
    plot_profit_evolution(data)
    plot_pub_vs_sales(data)

    print("\n[OK] utils module OK!")