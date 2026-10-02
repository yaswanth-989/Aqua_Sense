"""
AquaSense Municipal Water Demand EDA Module
Analyzes temporal trends, multi-zone variations, seasonal dynamics,
weather correlations, and usage patterns on the synthetic demand dataset.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.loader import load_raw_demand_data

FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures", "demand")
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_demand_eda() -> Dict[str, Any]:
    """Execute complete EDA on the municipal water demand dataset."""
    df = load_raw_demand_data()
    total_records = len(df)
    df["Date_dt"] = pd.to_datetime(df["Date"])

    # 1. Dataset overview
    overview = {
        "dataset_name": "Synthetic Municipal Water Demand",
        "total_records": total_records,
        "date_start": str(df["Date"].min()),
        "date_end": str(df["Date"].max()),
        "zones_count": int(df["Zone_ID"].nunique()),
        "zones": sorted(list(df["Zone_ID"].unique())),
        "overall_mean_demand_kL": round(float(df["Water_Demand_kL"].mean()), 2),
        "overall_min_demand_kL": round(float(df["Water_Demand_kL"].min()), 2),
        "overall_max_demand_kL": round(float(df["Water_Demand_kL"].max()), 2),
        "overall_std_demand_kL": round(float(df["Water_Demand_kL"].std()), 2)
    }

    # 2. Missing value audit
    missing_analysis = []
    for col in df.columns:
        if col == "Date_dt":
            continue
        cnt = int(df[col].isnull().sum())
        pct = round((cnt / total_records) * 100, 2)
        missing_analysis.append({
            "feature": col,
            "missing_count": cnt,
            "missing_pct": pct
        })

    # 3. Zone-level demand profiling
    zone_profiles = []
    for zone, group in df.groupby("Zone_ID"):
        zone_profiles.append({
            "zone_id": zone,
            "record_count": int(len(group)),
            "population": int(group["Population"].iloc[0]),
            "mean_connections": round(float(group["Active_Connections"].dropna().mean()), 1),
            "mean_demand_kL": round(float(group["Water_Demand_kL"].mean()), 2),
            "std_demand_kL": round(float(group["Water_Demand_kL"].std()), 2),
            "min_demand_kL": round(float(group["Water_Demand_kL"].min()), 2),
            "median_demand_kL": round(float(group["Water_Demand_kL"].median()), 2),
            "max_demand_kL": round(float(group["Water_Demand_kL"].max()), 2)
        })
    zone_profiles = sorted(zone_profiles, key=lambda x: x["mean_demand_kL"], reverse=True)

    # 4. Day of Week & Weekend Analysis
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_profiles = []
    for day in dow_order:
        sub = df[df["Day_of_Week"] == day]
        if len(sub) > 0:
            dow_profiles.append({
                "day": day,
                "mean_demand_kL": round(float(sub["Water_Demand_kL"].mean()), 2),
                "std_demand_kL": round(float(sub["Water_Demand_kL"].std()), 2)
            })

    weekend_comparison = {
        "weekday_mean_kL": round(float(df[df["Is_Weekend"] == 0]["Water_Demand_kL"].mean()), 2),
        "weekend_mean_kL": round(float(df[df["Is_Weekend"] == 1]["Water_Demand_kL"].mean()), 2),
        "weekend_uplift_kL": round(
            float(df[df["Is_Weekend"] == 1]["Water_Demand_kL"].mean() - df[df["Is_Weekend"] == 0]["Water_Demand_kL"].mean()),
            2
        )
    }

    # 5. Seasonal and Monthly Analysis
    month_profiles = []
    for m in range(1, 13):
        sub = df[df["Month"] == m]
        if len(sub) > 0:
            month_profiles.append({
                "month": m,
                "month_name": pd.to_datetime(f"2023-{m:02d}-01").strftime("%B"),
                "mean_demand_kL": round(float(sub["Water_Demand_kL"].mean()), 2),
                "std_demand_kL": round(float(sub["Water_Demand_kL"].std()), 2)
            })

    season_profiles = []
    for season, group in df.groupby("Season"):
        season_profiles.append({
            "season": season,
            "mean_demand_kL": round(float(group["Water_Demand_kL"].mean()), 2),
            "std_demand_kL": round(float(group["Water_Demand_kL"].std()), 2)
        })
    season_profiles = sorted(season_profiles, key=lambda x: x["mean_demand_kL"], reverse=True)

    # 6. Weather Relationships & Intra-zone correlations
    weather_corrs = {
        "overall": {
            "temperature_corr": round(float(df[["Avg_Temperature_C", "Water_Demand_kL"]].dropna().corr().iloc[0, 1]), 4),
            "humidity_corr": round(float(df[["Humidity_pct", "Water_Demand_kL"]].dropna().corr().iloc[0, 1]), 4),
            "rainfall_corr": round(float(df[["Rainfall_mm", "Water_Demand_kL"]].dropna().corr().iloc[0, 1]), 4)
        },
        "intra_zone_avg": {}
    }
    # Within each zone (controlling for zone size)
    temp_zone_corrs = []
    for _, zdf in df.groupby("Zone_ID"):
        temp_zone_corrs.append(float(zdf[["Avg_Temperature_C", "Water_Demand_kL"]].dropna().corr().iloc[0, 1]))
    weather_corrs["intra_zone_avg"]["temperature_corr"] = round(float(np.mean(temp_zone_corrs)), 4)

    # 7. Outlier analysis (IQR per zone)
    zone_outliers = []
    for zone, group in df.groupby("Zone_ID"):
        dem = group["Water_Demand_kL"]
        q1 = float(dem.quantile(0.25))
        q3 = float(dem.quantile(0.75))
        iqr = q3 - q1
        lower_bound = round(q1 - 1.5 * iqr, 2)
        upper_bound = round(q3 + 1.5 * iqr, 2)
        outlier_cnt = int(((dem < lower_bound) | (dem > upper_bound)).sum())
        zone_outliers.append({
            "zone_id": zone,
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_cnt,
            "outlier_pct": round((outlier_cnt / len(dem)) * 100, 2)
        })

    demand_summary = {
        "overview": overview,
        "missing_analysis": missing_analysis,
        "zone_profiles": zone_profiles,
        "dow_profiles": dow_profiles,
        "weekend_comparison": weekend_comparison,
        "month_profiles": month_profiles,
        "season_profiles": season_profiles,
        "weather_correlations": weather_corrs,
        "zone_outliers": zone_outliers
    }

    # Save summary JSON
    with open(os.path.join(RESULTS_DIR, "demand_eda_summary.json"), "w") as f:
        json.dump(demand_summary, f, indent=2)

    # Generate figures
    _generate_demand_figures(df, zone_profiles, dow_profiles, month_profiles)

    return demand_summary


def _generate_demand_figures(df: pd.DataFrame, zone_profiles: list, dow_profiles: list, month_profiles: list):
    """Generate static plots for demand analysis."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Zone Comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    zones = [z["zone_id"] for z in zone_profiles]
    means = [z["mean_demand_kL"] for z in zone_profiles]
    bars = ax.bar(zones, means, color="#0d9488", width=0.55)
    ax.set_title("Mean Daily Water Demand by Zone (kL)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Demand (kL)")
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "demand_zone_comparison.png"), dpi=150)
    plt.close(fig)

    # 2. Monthly Trend
    fig, ax = plt.subplots(figsize=(8, 4))
    months = [m["month_name"][:3] for m in month_profiles]
    m_means = [m["mean_demand_kL"] for m in month_profiles]
    ax.plot(months, m_means, marker="o", color="#0284c7", linewidth=2)
    ax.set_title("Monthly Water Demand Seasonality", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Demand (kL)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "demand_monthly_seasonality.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    results = compute_demand_eda()
    print("Demand EDA computed successfully.")
    print("Overview:", results["overview"])
    print("Weekend comparison:", results["weekend_comparison"])
