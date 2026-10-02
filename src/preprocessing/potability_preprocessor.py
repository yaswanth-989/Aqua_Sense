"""
AquaSense Potability Preprocessing Pipeline
Handles train-only imputation, outlier investigation, scaling, and train/test partition.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.loader import load_raw_potability_data

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "potability")
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "results")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def preprocess_potability_data(
    test_size: float = 0.2,
    random_state: int = 42,
    use_robust_scaling: bool = False
) -> Dict[str, Any]:
    """
    Execute leakage-free preprocessing on water potability dataset:
    1. Stratified train/test split.
    2. Fit SimpleImputer(median) ON TRAIN ONLY.
    3. Fit Scaler (Standard or Robust) ON TRAIN ONLY.
    4. Save processed train & test datasets.
    5. Save fitted preprocessor artifacts for consistent inference.
    """
    df = load_raw_potability_data()
    feature_cols = [c for c in df.columns if c != "Potability"]

    # 1. Stratified Split (prevents test data leakage into imputation/scaling)
    X = df[feature_cols].copy()
    y = df["Potability"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 2. Imputation (Median is robust against extreme chemical sensor values)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    train_imputed = imputer.transform(X_train)
    test_imputed = imputer.transform(X_test)

    # Record imputed median statistics for transparency
    imputed_medians = {
        col: round(float(imputer.statistics_[i]), 2)
        for i, col in enumerate(feature_cols)
    }

    # 3. Scaling (Fitted solely on train)
    scaler = RobustScaler() if use_robust_scaling else StandardScaler()
    scaler.fit(train_imputed)

    train_scaled = scaler.transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    # 4. Form processed DataFrames
    df_train_proc = pd.DataFrame(train_scaled, columns=feature_cols, index=X_train.index)
    df_train_proc["Potability"] = y_train.values

    df_test_proc = pd.DataFrame(test_scaled, columns=feature_cols, index=X_test.index)
    df_test_proc["Potability"] = y_test.values

    # 5. Save processed data to data/processed/
    train_out = os.path.join(PROCESSED_DIR, "potability_train.csv")
    test_out = os.path.join(PROCESSED_DIR, "potability_test.csv")
    df_train_proc.to_csv(train_out, index=False)
    df_test_proc.to_csv(test_out, index=False)

    # 6. Save preprocessing pipeline artifacts for serving
    pipeline_artifact = {
        "imputer": imputer,
        "scaler": scaler,
        "features": feature_cols,
        "medians": imputed_medians
    }
    artifact_path = os.path.join(MODELS_DIR, "potability_preprocessor.joblib")
    joblib.dump(pipeline_artifact, artifact_path)

    # 7. Preprocessing summary audit
    summary = {
        "dataset": "Water Potability",
        "split_strategy": "Stratified random split (train-only fitting)",
        "train_samples": len(df_train_proc),
        "test_samples": len(df_test_proc),
        "features_processed": feature_cols,
        "imputation_strategy": "Median (train-fitted)",
        "fitted_medians": imputed_medians,
        "scaler_used": "RobustScaler" if use_robust_scaling else "StandardScaler",
        "missing_remaining_train": int(df_train_proc.isnull().sum().sum()),
        "missing_remaining_test": int(df_test_proc.isnull().sum().sum()),
        "train_class_balance": {
            "0": int((y_train == 0).sum()),
            "1": int((y_train == 1).sum())
        },
        "test_class_balance": {
            "0": int((y_test == 0).sum()),
            "1": int((y_test == 1).sum())
        },
        "artifacts_saved": [train_out, test_out, artifact_path]
    }

    with open(os.path.join(RESULTS_DIR, "potability_preprocessing_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = preprocess_potability_data()
    print("Potability preprocessing completed.")
    print(f"Train: {res['train_samples']}, Test: {res['test_samples']}")
    print(f"Fitted Medians: {res['fitted_medians']}")
