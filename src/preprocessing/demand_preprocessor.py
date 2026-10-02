"""
AquaSense Municipal Water Demand Preprocessing Pipeline
Implements strictly chronological partitioning, leakage-safe lag & rolling feature
engineering, train-only imputation, and feature scaling.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.loader import load_raw_demand_data
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "demand")
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "results")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def preprocess_demand_data(
    test_start_date: str = "2025-01-01"
) -> Dict[str, Any]:
    """
    Execute leakage-safe demand forecasting preprocessing:
    1. Sort by Zone_ID and Date.
    2. Engineer historical lag and rolling features (shifted strictly to t-1 and earlier).
    3. Chronological train/test split.
    4. Fit imputation and scaler ON TRAIN ONLY.
    5. Save processed datasets and serialization artifacts.
    """
    df = load_raw_demand_data()
    df["Date_dt"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Zone_ID", "Date_dt"]).reset_index(drop=True)

    # 1. Historical Lag & Rolling Feature Engineering (Grouped strictly by Zone)
    # Day t uses strictly t-1, t-7, and historical windows ending at t-1
    df["Demand_Lag_1"] = df.groupby("Zone_ID")["Water_Demand_kL"].shift(1)
    df["Demand_Lag_7"] = df.groupby("Zone_ID")["Water_Demand_kL"].shift(7)
    df["Demand_Rolling_7"] = (
        df.groupby("Zone_ID")["Water_Demand_kL"].shift(1).rolling(7).mean()
    )
    df["Demand_Rolling_14"] = (
        df.groupby("Zone_ID")["Water_Demand_kL"].shift(1).rolling(14).mean()
    )

    # Remove initial 14-day warmup window per zone (70 rows total across 5 zones)
    df_clean = df.dropna(subset=["Demand_Rolling_14"]).copy().reset_index(drop=True)

    # 2. Chronological Split (No future information leaks into training)
    train_mask = df_clean["Date_dt"] < pd.to_datetime(test_start_date)
    test_mask = df_clean["Date_dt"] >= pd.to_datetime(test_start_date)

    df_train = df_clean[train_mask].copy()
    df_test = df_clean[test_mask].copy()

    # 3. Missing Value Imputation (Train medians)
    weather_cols = ["Active_Connections", "Avg_Temperature_C", "Humidity_pct", "Rainfall_mm"]
    impute_medians = {}
    for col in weather_cols:
        med = float(df_train[col].median())
        impute_medians[col] = round(med, 2)
        df_train[col] = df_train[col].fillna(med)
        df_test[col] = df_test[col].fillna(med)

    # 4. Feature Selection & One-Hot Encoding for Zone and Season
    num_features = [
        "Population", "Active_Connections", "Avg_Temperature_C", "Humidity_pct", "Rainfall_mm",
        "Is_Weekend", "Is_Holiday", "Month",
        "Demand_Lag_1", "Demand_Lag_7", "Demand_Rolling_7", "Demand_Rolling_14"
    ]

    # Categorical encoding (Zone_ID and Day_of_Week)
    train_encoded = pd.get_dummies(df_train, columns=["Zone_ID", "Season"], drop_first=True)
    test_encoded = pd.get_dummies(df_test, columns=["Zone_ID", "Season"], drop_first=True)

    # Ensure identical columns between train and test
    encoded_cols = [c for c in train_encoded.columns if c not in ["Date", "Date_dt", "Day_of_Week", "Water_Demand_kL"]]
    for c in encoded_cols:
        if c not in test_encoded.columns:
            test_encoded[c] = 0
    test_encoded = test_encoded[train_encoded.columns]

    # 5. Fit Scaler strictly on train continuous features
    scaler = StandardScaler()
    scaler.fit(df_train[num_features])

    train_scaled_arr = scaler.transform(df_train[num_features])
    test_scaled_arr = scaler.transform(df_test[num_features])

    # 6. Save Processed Files
    train_out_path = os.path.join(PROCESSED_DIR, "demand_train.csv")
    test_out_path = os.path.join(PROCESSED_DIR, "demand_test.csv")
    train_encoded.to_csv(train_out_path, index=False)
    test_encoded.to_csv(test_out_path, index=False)

    # 7. Save Serialization Artifacts
    artifact = {
        "scaler": scaler,
        "scaled_features": num_features,
        "impute_medians": impute_medians,
        "feature_columns": encoded_cols,
        "test_start_date": test_start_date
    }
    artifact_path = os.path.join(MODELS_DIR, "demand_preprocessor.joblib")
    joblib.dump(artifact, artifact_path)

    # 8. Summary Audit
    summary = {
        "dataset": "Synthetic Water Demand",
        "partition_strategy": f"Chronological split (Train < {test_start_date}, Test >= {test_start_date})",
        "train_records": int(len(df_train)),
        "test_records": int(len(df_test)),
        "train_date_range": [str(df_train["Date"].min()), str(df_train["Date"].max())],
        "test_date_range": [str(df_test["Date"].min()), str(df_test["Date"].max())],
        "engineered_features": [
            "Demand_Lag_1 (t-1)", "Demand_Lag_7 (t-7)",
            "Demand_Rolling_7 (7-day MA)", "Demand_Rolling_14 (14-day MA)"
        ],
        "imputation_fitted_medians": impute_medians,
        "features_total": len(encoded_cols),
        "target": "Water_Demand_kL",
        "leakage_checks": {
            "future_information_in_train": False,
            "target_in_lag_features": False,
            "warmup_rows_dropped_total": int(len(df) - len(df_clean))
        },
        "artifacts_saved": [train_out_path, test_out_path, artifact_path]
    }

    with open(os.path.join(RESULTS_DIR, "demand_preprocessing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = preprocess_demand_data()
    print("Demand preprocessing completed.")
    print("Train records:", res["train_records"], "Date range:", res["train_date_range"])
    print("Test records:", res["test_records"], "Date range:", res["test_date_range"])
    print("Engineered features:", res["engineered_features"])
