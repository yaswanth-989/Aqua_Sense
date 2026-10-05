"""
AquaSense Water Potability Classification — Linear & Regularized Models
Implements:
1. Baseline Logistic Regression
2. L1-Regularized Logistic Regression (Sparsity / Feature Selection)
3. L2-Regularized Logistic Regression (Ridge Penalty)
4. Balanced Regularized Logistic Regression (Class Imbalance Mitigation)
Evaluates on unseen test partition (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix).
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "potability")
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_evaluate_potability_linear_models() -> Dict[str, Any]:
    """Train, tune, and evaluate all linear potability models on unseen test data."""
    train_path = os.path.join(PROCESSED_DIR, "potability_train.csv")
    test_path = os.path.join(PROCESSED_DIR, "potability_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        from src.preprocessing.potability_preprocessor import preprocess_potability_data
        preprocess_potability_data()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    features = [c for c in train_df.columns if c != "Potability"]
    X_train = train_df[features].values
    y_train = train_df["Potability"].values
    X_test = test_df[features].values
    y_test = test_df["Potability"].values

    models = {
        "Logistic Regression (Baseline OLS)": LogisticRegression(
            C=1.0, solver="lbfgs", random_state=42
        ),
        "Logistic Regression (L1 Lasso Penalty, C=0.5)": LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5, random_state=42
        ),
        "Logistic Regression (L2 Ridge Penalty, C=0.5)": LogisticRegression(
            penalty="l2", solver="lbfgs", C=0.5, random_state=42
        ),
        "Logistic Regression (Balanced Class Weight)": LogisticRegression(
            class_weight="balanced", C=1.0, random_state=42
        )
    }

    metrics_comparison = []
    trained_artifacts = {}
    primary_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        roc = float(roc_auc_score(y_test, probs))
        cm = confusion_matrix(y_test, preds).tolist()

        metrics_comparison.append({
            "model": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc, 4),
            "confusion_matrix": {
                "true_negative": int(cm[0][0]),
                "false_positive": int(cm[0][1]),
                "false_negative": int(cm[1][0]),
                "true_positive": int(cm[1][1])
            }
        })

        trained_artifacts[name] = model
        if "Balanced" in name:
            primary_model = model

    # Feature Coefficients & Odds Ratios for Model Interpretation (from Balanced model)
    coefficients = []
    for feat, coef in zip(features, primary_model.coef_[0]):
        odds_ratio = float(np.exp(coef))
        coefficients.append({
            "feature": feat,
            "coefficient": round(float(coef), 4),
            "odds_ratio": round(odds_ratio, 4),
            "impact": "Increases Potability Odds" if coef > 0 else "Decreases Potability Odds"
        })
    coefficients = sorted(coefficients, key=lambda x: abs(x["coefficient"]), reverse=True)

    # Probability Distribution Deciles for Calibration Insights
    test_probs = primary_model.predict_proba(X_test)[:, 1]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    prob_hist, _ = np.histogram(test_probs, bins=bins)
    prob_deciles = {
        "bins": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        "counts": [int(c) for c in prob_hist]
    }

    results = {
        "task": "Water Potability Assessment (Binary Classification)",
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "features_used": features,
        "metrics_comparison": metrics_comparison,
        "feature_coefficients": coefficients,
        "probability_distribution": prob_deciles,
        "baseline_confusion_matrix": metrics_comparison[3]["confusion_matrix"]
    }

    # Save artifacts and evaluation summary
    joblib.dump(trained_artifacts, os.path.join(MODELS_DIR, "linear_potability_models.joblib"))

    with open(os.path.join(RESULTS_DIR, "linear_potability_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = train_evaluate_potability_linear_models()
    print("Linear Potability Models Trained & Evaluated Successfully.")
    for m in res["metrics_comparison"]:
        print(f"  {m['model']:45} | Acc: {m['accuracy']} | Prec: {m['precision']} | Rec: {m['recall']} | F1: {m['f1']} | ROC: {m['roc_auc']}")
