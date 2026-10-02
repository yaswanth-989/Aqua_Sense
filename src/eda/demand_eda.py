"""
AquaSense Municipal Water Demand EDA Module
Comprehensive exploratory analysis covering:
1. Univariate distributions, skewness, kurtosis, and precomputed histograms.
2. Temporal trends (Day-of-Week, Monthly seasonality, Weekend uplift).
3. Spatial multi-zone comparisons and demographic capacity.
4. Weather correlations (Temperature, Humidity, Rainfall).
5. Outlier detection across zones using Tukey's IQR boundaries.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
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

    # 2. Missing value audit & unique counts
    missing_analysis = []
    unique_counts = []
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

        unq = int(df[col].nunique())
        unique_counts.append({
            "feature": col,
            "unique_count": unq,
            "unique_pct": round((unq / total_records) * 100, 2),
            "dtype": str(df[col].dtype)
        })

    # 3. Univariate numerical statistics & histograms
    num_cols = ["Water_Demand_kL", "Avg_Temperature_C", "Humidity_pct", "Rainfall_mm", "Active_Connections", "Population"]
    univariate_profiles = []
    histograms = {}

    for col in num_cols:
        s = df[col].dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        skew_val = float(s.skew())
        kurt_val = float(stats.kurtosis(s))

        univariate_profiles.append({
            "feature": col,
            "count": int(len(s)),
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "q25": round(q1, 2),
            "median": round(float(s.median()), 2),
            "q75": round(q3, 2),
            "max": round(float(s.max()), 2),
            "iqr": round(iqr, 2),
            "skewness": round(skew_val, 3),
            "kurtosis": round(kurt_val, 3)
        })

        # Precompute 10 histogram bins
        bin_counts, bin_edges = np.histogram(s, bins=10)
        bin_labels = [
            f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
            for i in range(len(bin_counts))
        ]
        histograms[col] = {
            "labels": bin_labels,
            "counts": [int(c) for c in bin_counts]
        }

    # 4. Zone-level demand profiling
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

    # 5. Day of Week & Weekend Analysis
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

    # 6. Monthly & Seasonal Analysis
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

    # 7. Weather Correlations
    weather_corrs = {
        "overall": {
            "temperature_corr": round(float(df[["Avg_Temperature_C", "Water_Demand_kL"]].dropna().corr().iloc[0, 1]), 4),
            "humidity_corr": round(float(df[["Humidity_pct", "Water_Demand_kL"]].dropna().corr().iloc[0, 1]), 4),
            "rainfall_corr": round(float(df[["Rainfall_mm", "Water_Demand_kL"]].dropna().corr().iloc[0, 1]), 4)
        },
        "intra_zone_avg": {}
    }
    temp_zone_corrs = []
    for _, zdf in df.groupby("Zone_ID"):
        temp_zone_corrs.append(float(zdf[["Avg_Temperature_C", "Water_Demand_kL"]].dropna().corr().iloc[0, 1]))
    weather_corrs["intra_zone_avg"]["temperature_corr"] = round(float(np.mean(temp_zone_corrs)), 4)

    # 8. Outliers per zone
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
        "unique_counts": unique_counts,
        "univariate_profiles": univariate_profiles,
        "histograms": histograms,
        "zone_profiles": zone_profiles,
        "dow_profiles": dow_profiles,
        "weekend_comparison": weekend_comparison,
        "month_profiles": month_profiles,
        "season_profiles": season_profiles,
        "weather_correlations": weather_corrs,
        "zone_outliers": zone_outliers
    }

    with open(os.path.join(RESULTS_DIR, "demand_eda_summary.json"), "w") as f:
        json.dump(demand_summary, f, indent=2)

    return demand_summary


if __name__ == "__main__":
    res = compute_demand_eda()
    print("Demand EDA computed successfully with full univariate and seasonal metrics.")
    print("Sample univariate profile:", res["univariate_profiles"][0])
