"""
AquaSense Water Potability EDA Module
Adapts coursework EDA principles (distribution profiling, missingness, correlation matrix,
mutual information, and IQR outlier boundaries) for the water potability dataset.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.feature_selection import mutual_info_classif

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.loader import load_raw_potability_data

FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures", "potability")
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_potability_eda() -> Dict[str, Any]:
    """Execute complete EDA on the real water potability dataset and return structured analysis."""
    df = load_raw_potability_data()
    total_samples = len(df)
    features = [c for c in df.columns if c != "Potability"]

    # 1. Target distribution
    target_counts = df["Potability"].value_counts().to_dict()
    target_pct = {k: round((v / total_samples) * 100, 2) for k, v in target_counts.items()}
    target_analysis = {
        "non_potable_0": int(target_counts.get(0, 0)),
        "potable_1": int(target_counts.get(1, 0)),
        "non_potable_pct": target_pct.get(0, 0.0),
        "potable_pct": target_pct.get(1, 0.0),
        "imbalance_ratio": round(target_counts.get(0, 0) / max(1, target_counts.get(1, 1)), 2)
    }

    # 2. Missing values analysis
    missing_analysis = []
    for col in df.columns:
        cnt = int(df[col].isnull().sum())
        pct = round((cnt / total_samples) * 100, 2)
        missing_analysis.append({
            "feature": col,
            "missing_count": cnt,
            "missing_pct": pct,
            "complete_count": total_samples - cnt
        })

    # 3. Univariate statistical profiling
    stat_profiles = []
    outlier_profiles = []
    for col in features:
        s = df[col].dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((s < lower_bound) | (s > upper_bound)).sum())
        outlier_pct = round((outlier_count / len(s)) * 100, 2)

        mean_val = float(s.mean())
        std_val = float(s.std())
        skew_val = float(s.skew())

        stat_profiles.append({
            "feature": col,
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "min": round(float(s.min()), 2),
            "q25": round(q1, 2),
            "median": round(float(s.median()), 2),
            "q75": round(q3, 2),
            "max": round(float(s.max()), 2),
            "skewness": round(skew_val, 3)
        })

        outlier_profiles.append({
            "feature": col,
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct
        })

    # 4. Correlation matrix
    corr_matrix = df.corr()
    corr_dict = {}
    for r in corr_matrix.index:
        corr_dict[r] = {c: round(float(corr_matrix.loc[r, c]), 4) for c in corr_matrix.columns}

    # Correlation specifically with Potability target
    target_corrs = []
    for col in features:
        r_val = float(corr_matrix.loc[col, "Potability"])
        target_corrs.append({
            "feature": col,
            "correlation_with_target": round(r_val, 4),
            "abs_correlation": round(abs(r_val), 4)
        })
    target_corrs = sorted(target_corrs, key=lambda x: x["abs_correlation"], reverse=True)

    # 5. Mutual Information Scores (using complete cases)
    clean_df = df.dropna()
    mi_scores = mutual_info_classif(
        clean_df[features], clean_df["Potability"], random_state=42
    )
    mi_ranking = []
    for col, mi in zip(features, mi_scores):
        mi_ranking.append({
            "feature": col,
            "mutual_information": round(float(mi), 4)
        })
    mi_ranking = sorted(mi_ranking, key=lambda x: x["mutual_information"], reverse=True)

    # 6. Grouped statistics (Potable vs Non-Potable mean comparison)
    grouped_stats = []
    for col in features:
        m0 = float(df[df["Potability"] == 0][col].mean())
        m1 = float(df[df["Potability"] == 1][col].mean())
        grouped_stats.append({
            "feature": col,
            "mean_non_potable": round(m0, 2),
            "mean_potable": round(m1, 2),
            "diff": round(m1 - m0, 2)
        })

    eda_summary = {
        "dataset_name": "Water Potability",
        "sample_count": total_samples,
        "feature_count": len(features),
        "features": features,
        "target_analysis": target_analysis,
        "missing_analysis": missing_analysis,
        "univariate_profiles": stat_profiles,
        "outlier_profiles": outlier_profiles,
        "target_correlations": target_corrs,
        "mutual_information": mi_ranking,
        "grouped_stats": grouped_stats,
        "correlation_matrix": corr_dict
    }

    # Save summary JSON
    with open(os.path.join(RESULTS_DIR, "potability_eda_summary.json"), "w") as f:
        json.dump(eda_summary, f, indent=2)

    # Generate and save publication-grade figures
    _generate_potability_figures(df, features, corr_matrix)

    return eda_summary


def _generate_potability_figures(df: pd.DataFrame, features: list, corr_matrix: pd.DataFrame):
    """Generate static plots for reports and verification."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Potability"].value_counts()
    colors = ["#64748b", "#0284c7"]
    bars = ax.bar(["Non-Potable (0)", "Potable (1)"], counts.values, color=colors, width=0.5)
    ax.set_title("Water Potability Class Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    for b in bars:
        ax.annotate(f"{b.get_height()}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "potability_class_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax, linewidths=0.5)
    ax.set_title("Water Quality Physicochemical Correlation Heatmap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "potability_correlation_heatmap.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    results = compute_potability_eda()
    print("Potability EDA computed successfully.")
    print("Target distribution:", results["target_analysis"])
    print("Top correlated feature:", results["target_correlations"][0])
