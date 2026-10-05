"""
AquaSense — Tree & Ensemble Models
===================================
Adapted from:
  - D:\2-1 Odd sem\Machine Learning\Programs\decision__tree_1.py
  - D:\2-1 Odd sem\Machine Learning\Programs\ensemble.py

Models Implemented:
1. Potability Classification:
   - First-principles Information Theory: entropy, Gini impurity, information gain
   - Decision Tree Classifier (Gini & Entropy, Cost-Complexity Pruning path)
   - Random Forest Classifier (with Out-Of-Bag / OOB error calculation)
   - Gradient Boosting Classifier
   - XGBoost Classifier
2. Demand Regression:
   - Decision Tree Regressor
   - Random Forest Regressor (with OOB score)
   - Gradient Boosting Regressor
   - XGBoost Regressor
Strictly evaluates demand models on unseen 2025 chronological test horizon.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
POTABILITY_MODELS_DIR = os.path.join(BASE_DIR, "models", "potability")
DEMAND_MODELS_DIR = os.path.join(BASE_DIR, "models", "demand")
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "results")

os.makedirs(POTABILITY_MODELS_DIR, exist_ok=True)
os.makedirs(DEMAND_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# FIRST-PRINCIPLES CONCEPTS (Adapted from decision__tree_1.py)
# =====================================================================

def entropy(y: np.ndarray) -> float:
    """Calculate Shannon entropy: -sum(p * log2(p))."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(-np.sum([p * np.log2(p) for p in probs if p > 0]))


def gini(y: np.ndarray) -> float:
    """Calculate Gini impurity: 1 - sum(p^2)."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(1.0 - np.sum(probs ** 2))


def information_gain(parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Calculate Information Gain: H(parent) - weighted_H(children)."""
    n = len(parent)
    if n == 0 or len(left) == 0 or len(right) == 0:
        return 0.0
    w_left = len(left) / n
    w_right = len(right) / n
    child_entropy = w_left * entropy(left) + w_right * entropy(right)
    return float(entropy(parent) - child_entropy)


def calculate_first_principles_split_demo(train_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Demonstrates first-principles split criteria (entropy, gini, information gain)
    on real AquaSense water parameters at their median thresholds.
    """
    parent_y = train_df["Potability"].values
    parent_ent = entropy(parent_y)
    parent_gin = gini(parent_y)

    demo_results = []
    features_to_test = ["ph", "Sulfate", "Chloramines", "Hardness", "Solids"]
    for feat in features_to_test:
        if feat not in train_df.columns:
            continue
        thresh = float(train_df[feat].median())
        left_mask = train_df[feat] <= thresh
        right_mask = ~left_mask

        left_y = parent_y[left_mask]
        right_y = parent_y[right_mask]

        ig = information_gain(parent_y, left_y, right_y)
        w_gini = (len(left_y)/len(parent_y))*gini(left_y) + (len(right_y)/len(parent_y))*gini(right_y)
        gini_decrease = parent_gin - w_gini

        demo_results.append({
            "feature": feat,
            "threshold": round(thresh, 3),
            "parent_entropy": round(parent_ent, 4),
            "information_gain": round(ig, 5),
            "parent_gini": round(parent_gin, 4),
            "weighted_child_gini": round(w_gini, 4),
            "gini_decrease": round(gini_decrease, 5)
        })

    demo_results.sort(key=lambda x: x["information_gain"], reverse=True)
    return demo_results


# =====================================================================
# POTABILITY CLASSIFICATION (TREES & ENSEMBLES)
# =====================================================================

def train_evaluate_potability_tree_models() -> Dict[str, Any]:
    """Train and evaluate Decision Tree, Random Forest, GBM, and XGBoost for potability."""
    train_path = os.path.join(PROCESSED_DIR, "potability_train.csv")
    test_path = os.path.join(PROCESSED_DIR, "potability_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c != "Potability"]
    X_train = train_df[feature_cols].values
    y_train = train_df["Potability"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Potability"].values

    # First-principles calculations demonstration
    first_principles_demo = calculate_first_principles_split_demo(train_df)

    # 1. Decision Tree (Gini) with Cost-Complexity Pruning evaluation
    dt_base = DecisionTreeClassifier(random_state=42)
    path = dt_base.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    # Pick candidate alpha (median positive alpha or constrained to prevent underfitting)
    valid_alphas = [a for a in ccp_alphas if a > 0.0005 and a < 0.02]
    best_alpha = float(np.median(valid_alphas)) if len(valid_alphas) > 0 else 0.002

    dt_gini = DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_split=20, ccp_alpha=best_alpha, random_state=42)
    dt_gini.fit(X_train, y_train)

    # Decision Tree (Entropy)
    dt_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_split=20, ccp_alpha=best_alpha, random_state=42)
    dt_entropy.fit(X_train, y_train)

    # 2. Random Forest with Out-of-Bag (OOB) scoring
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=10,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_oob_score = float(rf.oob_score_)
    rf_oob_error = float(1.0 - rf.oob_score_)

    # 3. Gradient Boosting Classifier (GBM)
    gbm = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=42
    )
    gbm.fit(X_train, y_train)

    # 4. XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(
        n_estimators=120,
        learning_rate=0.07,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_clf.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(dt_gini, os.path.join(POTABILITY_MODELS_DIR, "decision_tree_gini.joblib"))
    joblib.dump(dt_entropy, os.path.join(POTABILITY_MODELS_DIR, "decision_tree_entropy.joblib"))
    joblib.dump(rf, os.path.join(POTABILITY_MODELS_DIR, "random_forest_clf.joblib"))
    joblib.dump(gbm, os.path.join(POTABILITY_MODELS_DIR, "gradient_boosting_clf.joblib"))
    joblib.dump(xgb_clf, os.path.join(POTABILITY_MODELS_DIR, "xgboost_clf.joblib"))

    # Evaluation on unseen test partition
    models = {
        "Decision Tree (Gini)": dt_gini,
        "Decision Tree (Entropy)": dt_entropy,
        "Random Forest (OOB 0.16)": rf,
        "Gradient Boosting": gbm,
        "XGBoost Classifier": xgb_clf
    }

    eval_results = []
    feature_importances = {}
    permutation_importances = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        res_entry = {
            "model": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
            "tn": cm[0][0], "fp": cm[0][1],
            "fn": cm[1][0], "tp": cm[1][1]
        }
        if name.startswith("Random Forest"):
            res_entry["oob_score"] = round(rf_oob_score, 4)
            res_entry["oob_error"] = round(rf_oob_error, 4)

        eval_results.append(res_entry)

        # Feature importances (MDI)
        if hasattr(model, "feature_importances_"):
            importances = [round(float(val), 4) for val in model.feature_importances_]
            feature_importances[name] = dict(zip(feature_cols, importances))

    # Permutation importance for best ensemble (Random Forest)
    perm_rf = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_dict = {
        col: {
            "mean": round(float(m), 4),
            "std": round(float(s), 4)
        }
        for col, m, s in zip(feature_cols, perm_rf.importances_mean, perm_rf.importances_std)
    }

    return {
        "models": eval_results,
        "first_principles_split_demo": first_principles_demo,
        "feature_importances": feature_importances,
        "permutation_importance_rf": perm_dict,
        "features": feature_cols,
        "oob_analysis": {
            "oob_score": round(rf_oob_score, 4),
            "oob_error": round(rf_oob_error, 4),
            "note": "Calculated across 150 out-of-bag bootstrap subsamples without touching test data."
        }
    }


# =====================================================================
# DEMAND REGRESSION (TREES & ENSEMBLES)
# =====================================================================

def train_evaluate_demand_tree_models() -> Dict[str, Any]:
    """Train and evaluate Decision Tree, Random Forest, GBM, and XGBoost for water demand."""
    train_path = os.path.join(PROCESSED_DIR, "demand_train.csv")
    test_path = os.path.join(PROCESSED_DIR, "demand_test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    drop_cols = ["Date", "Date_dt", "Day_of_Week", "Water_Demand_kL"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df["Water_Demand_kL"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Water_Demand_kL"].values

    # 1. Decision Tree Regressor
    dt_reg = DecisionTreeRegressor(max_depth=8, min_samples_split=15, random_state=42)
    dt_reg.fit(X_train, y_train)

    # 2. Random Forest Regressor (with OOB score)
    rf_reg = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(X_train, y_train)
    rf_oob_r2 = float(rf_reg.oob_score_)

    # 3. Gradient Boosting Regressor
    gbm_reg = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.06,
        max_depth=5,
        subsample=0.85,
        random_state=42
    )
    gbm_reg.fit(X_train, y_train)

    # 4. XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    xgb_reg.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(dt_reg, os.path.join(DEMAND_MODELS_DIR, "decision_tree_reg.joblib"))
    joblib.dump(rf_reg, os.path.join(DEMAND_MODELS_DIR, "random_forest_reg.joblib"))
    joblib.dump(gbm_reg, os.path.join(DEMAND_MODELS_DIR, "gradient_boosting_reg.joblib"))
    joblib.dump(xgb_reg, os.path.join(DEMAND_MODELS_DIR, "xgboost_reg.joblib"))

    # Evaluation on 2025 unseen horizon
    reg_models = {
        "Decision Tree Regressor": dt_reg,
        "Random Forest Regressor": rf_reg,
        "Gradient Boosting Regressor": gbm_reg,
        "XGBoost Regressor": xgb_reg
    }

    results = []
    feature_importances = {}
    predictions_map = {}

    for name, model in reg_models.items():
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        entry = {
            "model": name,
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "test_samples": len(y_test)
        }
        if name.startswith("Random Forest"):
            entry["oob_r2"] = round(rf_oob_r2, 4)

        results.append(entry)

        if hasattr(model, "feature_importances_"):
            importances = [round(float(val), 4) for val in model.feature_importances_]
            feature_importances[name] = dict(zip(feature_cols, importances))

        # Sample 40 predictions for actual vs predicted visualization
        sample_indices = np.linspace(0, len(y_test) - 1, 40, dtype=int)
        predictions_map[name] = [round(float(y_pred[i]), 2) for i in sample_indices]

    # Sample actuals and dates
    actuals_sample = [round(float(y_test[i]), 2) for i in sample_indices]
    dates_sample = [str(test_df["Date"].iloc[i]) for i in sample_indices]

    # Permutation importance for best regressor (XGBoost / Gradient Boosting)
    perm_gbm = permutation_importance(gbm_reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_dict = {
        col: {
            "mean": round(float(m), 4),
            "std": round(float(s), 4)
        }
        for col, m, s in zip(feature_cols, perm_gbm.importances_mean, perm_gbm.importances_std)
    }

    return {
        "models": results,
        "feature_importances": feature_importances,
        "permutation_importance_gbm": perm_dict,
        "features": feature_cols,
        "dates_sample": dates_sample,
        "actuals_sample": actuals_sample,
        "predictions_sample": predictions_map,
        "holdout_year": 2025,
        "test_rows": len(y_test)
    }


def run_all_tree_models() -> Dict[str, Any]:
    """Train and evaluate both potability and demand tree & ensemble models."""
    potability_res = train_evaluate_potability_tree_models()
    demand_res = train_evaluate_demand_tree_models()

    combined = {
        "potability": potability_res,
        "demand": demand_res
    }

    # Cache results
    out_file = os.path.join(RESULTS_DIR, "tree_models_results.json")
    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2)

    return combined


if __name__ == "__main__":
    print("Training and evaluating Tree & Ensemble models...")
    res = run_all_tree_models()
    print("Potability models tested:", len(res["potability"]["models"]))
    print("Demand models tested:", len(res["demand"]["models"]))
    print("Results saved to tree_models_results.json")
