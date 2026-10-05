"""
AquaSense Municipal Water Demand Forecasting — Linear & Regularized Models
Implements:
1. Multiple Linear Regression Baseline
2. Vectorized From-Scratch Gradient Descent
3. Ridge Regression (L2 Regularization)
4. Lasso Regression (L1 Regularization)
5. Elastic Net (L1 + L2 Regularization)
Evaluates on strictly unseen 2025 chronological test horizon (RMSE, MAE, R²).
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "demand")
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def vectorized_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 300
) -> Tuple[np.ndarray, float, list]:
    """
    Vectorized multivariate Gradient Descent optimizer.
    Adapts coursework from-scratch GD algorithm for multi-feature matrix operations.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = float(np.mean(y))  # Initialize bias at mean target
    cost_history = []

    for epoch in range(epochs):
        # 1. Forward prediction
        y_pred = np.dot(X, weights) + bias

        # 2. Prediction error
        error = y_pred - y

        # 3. Mean Squared Error Cost
        mse = float(np.mean(error ** 2))
        cost_history.append(round(mse, 2))

        # 4. Vectorized parameter gradients
        dw = (2.0 / n_samples) * np.dot(X.T, error)
        db = (2.0 / n_samples) * np.sum(error)

        # 5. Parameter updates
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias, cost_history


def train_evaluate_demand_linear_models() -> Dict[str, Any]:
    """Train, tune, and evaluate all linear demand models on unseen test data."""
    train_path = os.path.join(PROCESSED_DIR, "demand_train.csv")
    test_path = os.path.join(PROCESSED_DIR, "demand_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        from src.preprocessing.demand_preprocessor import preprocess_demand_data
        preprocess_demand_data()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    drop_cols = ["Date", "Date_dt", "Day_of_Week", "Water_Demand_kL"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df["Water_Demand_kL"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Water_Demand_kL"].values

    # Scale continuous predictors (train-only fit)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Multiple Linear Regression Baseline (OLS)
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    ols_preds = ols.predict(X_test_scaled)

    # 2. Vectorized From-Scratch Gradient Descent
    gd_weights, gd_bias, gd_costs = vectorized_gradient_descent(
        X_train_scaled, y_train, learning_rate=0.05, epochs=300
    )
    gd_preds = np.dot(X_test_scaled, gd_weights) + gd_bias

    # 3. Ridge Regression (L2 Penalty)
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    ridge_preds = ridge.predict(X_test_scaled)

    # 4. Lasso Regression (L1 Penalty, Sparsity)
    lasso = Lasso(alpha=0.1, random_state=42, max_iter=2000)
    lasso.fit(X_train_scaled, y_train)
    lasso_preds = lasso.predict(X_test_scaled)

    # 5. Elastic Net (L1 + L2 Balance)
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
    elastic.fit(X_train_scaled, y_train)
    elastic_preds = elastic.predict(X_test_scaled)

    # Helper function to compute metrics
    def calc_metrics(y_true, y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4)
        }

    metrics_comparison = [
        {"model": "Multiple Linear Regression (OLS)", **calc_metrics(y_test, ols_preds)},
        {"model": "Gradient Descent (Vectorized From-Scratch)", **calc_metrics(y_test, gd_preds)},
        {"model": "Ridge Regression (L2, alpha=1.0)", **calc_metrics(y_test, ridge_preds)},
        {"model": "Lasso Regression (L1, alpha=0.1)", **calc_metrics(y_test, lasso_preds)},
        {"model": "Elastic Net (L1+L2, alpha=0.1, ratio=0.5)", **calc_metrics(y_test, elastic_preds)}
    ]

    # Feature Coefficients from OLS for Model Interpretation
    coefficients = []
    for feat, coef in zip(feature_cols, ols.coef_):
        coefficients.append({
            "feature": feat,
            "coefficient": round(float(coef), 2),
            "abs_weight": round(abs(float(coef)), 2)
        })
    coefficients = sorted(coefficients, key=lambda x: x["abs_weight"], reverse=True)

    # Sample Actual vs Predicted Trajectory (First 30 days of test period in Zone Z01)
    z01_indices = (test_df["Zone_ID_Z02"] == 0) & (test_df["Zone_ID_Z03"] == 0) & \
                  (test_df["Zone_ID_Z04"] == 0) & (test_df["Zone_ID_Z05"] == 0)
    z01_test = test_df[z01_indices].head(30)
    z01_actuals = [round(float(v), 1) for v in z01_test["Water_Demand_kL"].values]
    z01_preds = [round(float(v), 1) for v in ols_preds[z01_indices][:30]]
    z01_dates = [str(d) for d in z01_test["Date"].values]

    # Sample GD convergence points (every 15 epochs)
    sampled_gd_epochs = list(range(0, 300, 15))
    sampled_gd_costs = [gd_costs[i] for i in sampled_gd_epochs]

    results = {
        "task": "Water Demand Forecasting (Time-Series Regression)",
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "features_used": feature_cols,
        "metrics_comparison": metrics_comparison,
        "feature_coefficients": coefficients,
        "actual_vs_predicted_sample": {
            "dates": z01_dates,
            "actual": z01_actuals,
            "predicted": z01_preds,
            "sample_zone": "Zone Z01 (First 30 Days of 2025)"
        },
        "gradient_descent_convergence": {
            "epochs": sampled_gd_epochs,
            "cost_mse": sampled_gd_costs,
            "initial_cost": gd_costs[0],
            "final_cost": gd_costs[-1]
        }
    }

    # Save artifacts and summary
    artifacts = {
        "ols": ols,
        "ridge": ridge,
        "lasso": lasso,
        "elastic": elastic,
        "scaler": scaler,
        "features": feature_cols,
        "metrics": metrics_comparison
    }
    joblib.dump(artifacts, os.path.join(MODELS_DIR, "linear_demand_models.joblib"))

    with open(os.path.join(RESULTS_DIR, "linear_demand_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = train_evaluate_demand_linear_models()
    print("Linear Demand Models Trained & Evaluated Successfully.")
    for m in res["metrics_comparison"]:
        print(f"  {m['model']:40} | RMSE: {m['rmse']} | MAE: {m['mae']} | R2: {m['r2']}")
