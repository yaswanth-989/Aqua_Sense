"""
AquaSense Live Prediction & Inference Service
Provides real-time scoring for:
1. Water Potability Assessment (Binary Classification + WHO threshold audit + Risk Scoring)
2. Municipal Water Demand Forecasting (Next-day kL regression + Prediction Intervals + Pumping Guidance)
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

POTABILITY_MODEL_DIR = os.path.join(BASE_DIR, "models", "potability")
DEMAND_MODEL_DIR = os.path.join(BASE_DIR, "models", "demand")

# WHO Drinking Water Guidelines Standards
WHO_STANDARDS = {
    "ph": {"min": 6.5, "max": 8.5, "unit": "pH", "label": "pH Level"},
    "Hardness": {"min": 0.0, "max": 300.0, "unit": "mg/L", "label": "Hardness (CaCO₃)"},
    "Solids": {"min": 0.0, "max": 1000.0, "unit": "mg/L", "label": "Total Dissolved Solids (TDS)"},
    "Chloramines": {"min": 0.0, "max": 4.0, "unit": "mg/L", "label": "Chloramines"},
    "Sulfate": {"min": 0.0, "max": 250.0, "unit": "mg/L", "label": "Sulfate (SO₄²⁻)"},
    "Conductivity": {"min": 0.0, "max": 400.0, "unit": "µS/cm", "label": "Electrical Conductivity"},
    "Organic_carbon": {"min": 0.0, "max": 2.0, "unit": "mg/L", "label": "Organic Carbon (TOC)"},
    "Trihalomethanes": {"min": 0.0, "max": 80.0, "unit": "µg/L", "label": "Trihalomethanes (THM)"},
    "Turbidity": {"min": 0.0, "max": 5.0, "unit": "NTU", "label": "Turbidity"}
}

# Zone Operational Baseline Profiles
ZONE_PROFILES = {
    "Z01": {"name": "Zone 1 — Central Residential", "population": 52076, "connections": 13363, "base_demand": 785.35},
    "Z02": {"name": "Zone 2 — North Suburbs", "population": 40812, "connections": 10936, "base_demand": 579.79},
    "Z03": {"name": "Zone 3 — Downtown Commercial", "population": 67516, "connections": 17113, "base_demand": 1102.33},
    "Z04": {"name": "Zone 4 — East Industrial Belt", "population": 35169, "connections": 9316, "base_demand": 496.41},
    "Z05": {"name": "Zone 5 — South Mixed Municipal", "population": 60082, "connections": 15289, "base_demand": 931.44}
}

POTABILITY_PRESETS = [
    {
        "id": "who_compliant",
        "name": "WHO-Compliant Municipal Tap Water",
        "desc": "Optimal treatment — all 9 parameters strictly inside safe drinking boundaries.",
        "params": {
            "ph": 7.35, "Hardness": 185.0, "Solids": 320.0, "Chloramines": 2.8,
            "Sulfate": 175.0, "Conductivity": 340.0, "Organic_carbon": 1.4,
            "Trihalomethanes": 45.0, "Turbidity": 1.8
        }
    },
    {
        "id": "turbid_river",
        "name": "Monsoon River Runoff (Turbid / Silted)",
        "desc": "High turbidity and suspended particulates from surface runoff.",
        "params": {
            "ph": 6.2, "Hardness": 140.0, "Solids": 1180.0, "Chloramines": 4.8,
            "Sulfate": 280.0, "Conductivity": 460.0, "Organic_carbon": 3.8,
            "Trihalomethanes": 88.0, "Turbidity": 9.6
        }
    },
    {
        "id": "industrial_acidic",
        "name": "Industrial Effluent Spill (Acidic & High Sulfate)",
        "desc": "Severe chemical contamination: sub-6 pH and excessive sulfate load.",
        "params": {
            "ph": 5.1, "Hardness": 380.0, "Solids": 1450.0, "Chloramines": 6.5,
            "Sulfate": 395.0, "Conductivity": 580.0, "Organic_carbon": 4.9,
            "Trihalomethanes": 115.0, "Turbidity": 4.2
        }
    },
    {
        "id": "hard_groundwater",
        "name": "Deep Limestone Borewell (High Hardness)",
        "desc": "Mineralized aquifer water with elevated dissolved salts and calcium.",
        "params": {
            "ph": 8.1, "Hardness": 345.0, "Solids": 720.0, "Chloramines": 1.9,
            "Sulfate": 210.0, "Conductivity": 390.0, "Organic_carbon": 1.1,
            "Trihalomethanes": 32.0, "Turbidity": 2.2
        }
    }
]

DEMAND_PRESETS = [
    {
        "id": "summer_peak",
        "name": "Summer Heatwave — Downtown (Z03)",
        "desc": "Extreme temperature, zero rain, elevated commercial and residential consumption.",
        "params": {
            "zone_id": "Z03", "temperature": 39.5, "humidity": 42.0, "rainfall": 0.0,
            "is_weekend": 0, "is_holiday": 0, "month": 5, "lag_1": 1240.0, "lag_7": 1210.0
        }
    },
    {
        "id": "monsoon_downpour",
        "name": "Monsoon Heavy Rain — Central (Z01)",
        "desc": "High precipitation reducing outdoor, municipal, and garden demand.",
        "params": {
            "zone_id": "Z01", "temperature": 24.0, "humidity": 94.0, "rainfall": 48.5,
            "is_weekend": 0, "is_holiday": 0, "month": 7, "lag_1": 710.0, "lag_7": 735.0
        }
    },
    {
        "id": "winter_weekend",
        "name": "Winter Weekend — North Suburbs (Z02)",
        "desc": "Mild weather, residential weekend routine, steady baseline.",
        "params": {
            "zone_id": "Z02", "temperature": 18.5, "humidity": 55.0, "rainfall": 0.0,
            "is_weekend": 1, "is_holiday": 0, "month": 1, "lag_1": 565.0, "lag_7": 572.0
        }
    },
    {
        "id": "holiday_industrial",
        "name": "Public Holiday — Industrial Belt (Z04)",
        "desc": "Factory shutdown drastically drops industrial zone consumption.",
        "params": {
            "zone_id": "Z04", "temperature": 29.0, "humidity": 60.0, "rainfall": 0.0,
            "is_weekend": 0, "is_holiday": 1, "month": 10, "lag_1": 430.0, "lag_7": 490.0
        }
    }
]


class PredictionEngine:
    """Singleton service for cached model loading and fast real-time inference."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionEngine, cls).__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self):
        # 1. Potability Models & Preprocessor
        self.pot_prep = joblib.load(os.path.join(POTABILITY_MODEL_DIR, "potability_preprocessor.joblib"))
        
        # Load best tree ensemble
        rf_path = os.path.join(POTABILITY_MODEL_DIR, "random_forest_clf.joblib")
        xgb_path = os.path.join(POTABILITY_MODEL_DIR, "xgboost_clf.joblib")
        if os.path.exists(xgb_path):
            self.pot_model = joblib.load(xgb_path)
            self.pot_model_name = "XGBoost Classifier (Tuned)"
        elif os.path.exists(rf_path):
            self.pot_model = joblib.load(rf_path)
            self.pot_model_name = "Random Forest Classifier"
        else:
            lin_pot = joblib.load(os.path.join(POTABILITY_MODEL_DIR, "linear_potability_models.joblib"))
            self.pot_model = lin_pot["Logistic Regression (Baseline OLS)"]
            self.pot_model_name = "Logistic Regression"

        # 2. Demand Models & Preprocessor
        self.dem_prep = joblib.load(os.path.join(DEMAND_MODEL_DIR, "demand_preprocessor.joblib"))
        gb_dem_path = os.path.join(DEMAND_MODEL_DIR, "gradient_boosting_reg.joblib")
        rf_dem_path = os.path.join(DEMAND_MODEL_DIR, "random_forest_reg.joblib")
        lin_dem_path = os.path.join(DEMAND_MODEL_DIR, "linear_demand_models.joblib")

        if os.path.exists(gb_dem_path):
            self.dem_model = joblib.load(gb_dem_path)
            self.dem_model_name = "Gradient Boosting Regressor (Tuned)"
            self.dem_needs_scaling = False  # GB trained on raw features
        elif os.path.exists(rf_dem_path):
            self.dem_model = joblib.load(rf_dem_path)
            self.dem_model_name = "Random Forest Regressor"
            self.dem_needs_scaling = False  # RF trained on raw features
        else:
            lin_dem = joblib.load(lin_dem_path)
            self.dem_model = lin_dem["ols"]
            self.dem_model_name = "Multiple Linear Regression (OLS)"
            self.dem_needs_scaling = True   # Linear models trained on scaled features

        self.dem_rmse = 41.06  # GB Test holdout RMSE


def predict_potability(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a water sample:
    1. Impute missing values with training medians.
    2. Scale features with training standard scaler.
    3. Run model inference for class and probability.
    4. Audit each parameter against WHO standards.
    5. Generate safety verdict, risk category, and action guidelines.
    """
    engine = PredictionEngine()
    features = engine.pot_prep["features"]
    medians = engine.pot_prep["medians"]

    # Extract input values with fallback to fitted training medians
    input_values = {}
    violations = []
    compliance_details = []

    for feat in features:
        val = payload.get(feat)
        if val is None or val == "" or str(val).lower() == "nan":
            val = float(medians.get(feat, 0.0))
            is_imputed = True
        else:
            try:
                val = float(val)
                is_imputed = False
            except (ValueError, TypeError):
                val = float(medians.get(feat, 0.0))
                is_imputed = True

        input_values[feat] = val

        # Check against WHO guidelines
        who = WHO_STANDARDS.get(feat, {"min": 0, "max": 9999, "unit": "", "label": feat})
        is_violating = False
        deviation = 0.0
        status = "COMPLIANT"

        if val < who["min"]:
            is_violating = True
            deviation = round(who["min"] - val, 2)
            status = f"BELOW MIN ({who['min']})"
        elif val > who["max"]:
            is_violating = True
            deviation = round(val - who["max"], 2)
            status = f"EXCEEDS MAX ({who['max']})"

        if is_violating:
            violations.append({
                "feature": feat,
                "label": who["label"],
                "value": round(val, 2),
                "threshold": f"{who['min']} – {who['max']} {who['unit']}",
                "status": status,
                "deviation": deviation
            })

        compliance_details.append({
            "feature": feat,
            "label": who["label"],
            "value": round(val, 2),
            "unit": who["unit"],
            "safe_range": f"{who['min']} – {who['max']}",
            "is_compliant": not is_violating,
            "status": status,
            "is_imputed": is_imputed
        })

    # Prepare array for model
    sample_arr = np.array([[input_values[f] for f in features]])
    X_scaled = engine.pot_prep["scaler"].transform(sample_arr)

    # Inference
    prediction = int(engine.pot_model.predict(X_scaled)[0])
    
    if hasattr(engine.pot_model, "predict_proba"):
        proba = engine.pot_model.predict_proba(X_scaled)[0]
        prob_potable = float(proba[1])
        prob_non_potable = float(proba[0])
    else:
        prob_potable = 1.0 if prediction == 1 else 0.0
        prob_non_potable = 1.0 - prob_potable

    # Risk level: WHO violations are primary signal; probability is secondary
    # If >= 4 violations → HIGH RISK regardless of model output
    # If >= 2 violations or probability < 0.45 → MODERATE RISK
    # Otherwise LOW RISK
    if len(violations) >= 4 or (prob_potable < 0.35 and len(violations) >= 2):
        risk_level = "HIGH RISK / CONTAMINATED"
        risk_color = "var(--red,#ef4444)"
        badge_class = "danger"
        recommendation = "Potentially hazardous water. Chemical or physical parameters exceed safety standards. Immediate corrective coagulation and reverse osmosis treatment required."
    elif len(violations) >= 2 or prob_potable < 0.48:
        risk_level = "MODERATE RISK"
        risk_color = "var(--amber)"
        badge_class = "warning"
        recommendation = "Marginal water quality detected. Secondary filtration, UV disinfection, or pH buffering advised prior to municipal consumption."
    else:
        risk_level = "LOW RISK"
        risk_color = "var(--green)"
        badge_class = "green"
        recommendation = "Water complies with standard potable quality thresholds. Suitable for direct municipal distribution and consumption."

    # Final safety verdict: Safety guardrail overrides raw model prediction if extreme WHO violations occur
    if risk_level == "HIGH RISK / CONTAMINATED" or len(violations) >= 3:
        is_potable = False
        verdict = "UNSAFE / NON-POTABLE (Severe WHO Exceedance)"
    elif prediction == 1 and len(violations) <= 1:
        is_potable = True
        verdict = "SAFE TO DRINK (Potable)"
    elif prediction == 1:
        is_potable = False
        verdict = "MARGINAL QUALITY (Filtration Required)"
    else:
        is_potable = False
        verdict = "UNSAFE / NON-POTABLE"

    return {
        "status": "success",
        "model_used": engine.pot_model_name,
        "is_potable": is_potable,
        "verdict": verdict,
        "potability_probability": round(prob_potable * 100, 1),
        "non_potable_probability": round(prob_non_potable * 100, 1),
        "confidence_score": round(max(prob_potable, prob_non_potable) * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "badge_class": badge_class,
        "recommendation": recommendation,
        "violation_count": len(violations),
        "violations": violations,
        "compliance_details": compliance_details,
        "input_values": input_values
    }


def predict_demand(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Forecast next-day municipal water consumption:
    1. Parse zone selection and load zone infrastructure baseline.
    2. Fill weather, calendar, and autoregressive lag variables.
    3. Scale features via training standard scaler.
    4. Predict next-day consumption in kL.
    5. Calculate 95% confidence interval and pumping guidance.
    """
    engine = PredictionEngine()
    feature_cols = engine.dem_prep["feature_columns"]
    scaled_cols = engine.dem_prep["scaled_features"]
    scaler = engine.dem_prep["scaler"]

    zone_id = str(payload.get("zone_id", "Z01")).strip().upper()
    if zone_id not in ZONE_PROFILES:
        zone_id = "Z01"

    zone_info = ZONE_PROFILES[zone_id]

    # Parse and validate inputs
    population = float(payload.get("population", zone_info["population"]))
    connections = float(payload.get("connections", zone_info["connections"]))
    temperature = float(payload.get("temperature", 28.5))
    humidity = float(payload.get("humidity", 65.0))
    rainfall = float(payload.get("rainfall", 0.0))
    is_weekend = int(bool(payload.get("is_weekend", 0)))
    is_holiday = int(bool(payload.get("is_holiday", 0)))
    month = int(payload.get("month", 6))
    if not (1 <= month <= 12):
        month = 6

    # Lags (fallback to zone baseline if omitted)
    base_demand = zone_info["base_demand"]
    lag_1 = float(payload.get("lag_1", base_demand))
    lag_7 = float(payload.get("lag_7", base_demand))
    roll_7 = float(payload.get("roll_7", (lag_1 + lag_7) / 2))
    roll_14 = float(payload.get("roll_14", base_demand))

    # Derive Season
    # Summer: Mar-May (3,4,5); Monsoon: Jun-Sep (6,7,8,9); Post-Monsoon: Oct-Nov (10,11); Winter: Dec-Feb (12,1,2)
    season_summer = 1 if month in [3, 4, 5] else 0
    season_post_monsoon = 1 if month in [10, 11] else 0
    season_winter = 1 if month in [12, 1, 2] else 0

    # Build input dictionary
    row = {
        "Population": population,
        "Active_Connections": connections,
        "Avg_Temperature_C": temperature,
        "Humidity_pct": humidity,
        "Rainfall_mm": rainfall,
        "Is_Weekend": is_weekend,
        "Is_Holiday": is_holiday,
        "Month": month,
        "Demand_Lag_1": lag_1,
        "Demand_Lag_7": lag_7,
        "Demand_Rolling_7": roll_7,
        "Demand_Rolling_14": roll_14,
        "Zone_ID_Z02": 1 if zone_id == "Z02" else 0,
        "Zone_ID_Z03": 1 if zone_id == "Z03" else 0,
        "Zone_ID_Z04": 1 if zone_id == "Z04" else 0,
        "Zone_ID_Z05": 1 if zone_id == "Z05" else 0,
        "Season_Post-Monsoon": season_post_monsoon,
        "Season_Summer": season_summer,
        "Season_Winter": season_winter
    }

    # Build DataFrame from raw values
    df = pd.DataFrame([row])

    # Only scale if model requires it (linear models need scaling; tree models do not)
    if engine.dem_needs_scaling:
        df[scaled_cols] = scaler.transform(df[scaled_cols])

    X_vector = df[feature_cols].values

    # Run inference
    pred_demand = float(engine.dem_model.predict(X_vector)[0])
    pred_demand = max(50.0, round(pred_demand, 2))  # Physically bounded non-negative

    # Confidence interval (95% CI ~ ± 1.96 * RMSE)
    margin = round(1.96 * engine.dem_rmse, 1)
    lower_bound = round(max(0.0, pred_demand - margin), 1)
    upper_bound = round(pred_demand + margin, 1)

    # Deviation from zone normal
    pct_deviation = round(((pred_demand - base_demand) / base_demand) * 100, 1)

    # Operational Dispatch Recommendation
    if pct_deviation > 10.0:
        pumping_status = "SURGE CAPACITY REQUIRED"
        pumping_color = "var(--red,#ef4444)"
        dispatch_note = f"Demand forecasted {pct_deviation:+}% above zone baseline. Activate auxiliary pump station #2 and reserve 15% supplemental head in clearwater reservoirs."
    elif pct_deviation < -8.0:
        pumping_status = "REDUCED OUTPUT DISPATCH"
        pumping_color = "var(--teal)"
        dispatch_note = f"Demand forecasted {pct_deviation:+}% below zone baseline (weather/holiday attenuation). Throttle main pumps to base speed to conserve energy."
    else:
        pumping_status = "NOMINAL BASE LOAD"
        pumping_color = "var(--green)"
        dispatch_note = f"Demand forecasted within normal range ({pct_deviation:+}% vs baseline). Operate primary variable frequency drives on standard daily schedule."

    return {
        "status": "success",
        "model_used": engine.dem_model_name,
        "zone_id": zone_id,
        "zone_name": zone_info["name"],
        "predicted_demand_kL": pred_demand,
        "confidence_interval": {
            "lower_kL": lower_bound,
            "upper_kL": upper_bound,
            "margin_kL": margin
        },
        "baseline_demand_kL": base_demand,
        "deviation_pct": pct_deviation,
        "pumping_status": pumping_status,
        "pumping_color": pumping_color,
        "dispatch_note": dispatch_note,
        "inputs": {
            "population": int(population),
            "connections": int(connections),
            "temperature_C": temperature,
            "humidity_pct": humidity,
            "rainfall_mm": rainfall,
            "is_weekend": bool(is_weekend),
            "is_holiday": bool(is_holiday),
            "month": month,
            "lag_1_kL": lag_1
        }
    }
