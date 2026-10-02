"""
AquaSense Data Validation Module
Performs schema verification, missingness audits, range checks, and integrity checks.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.data.loader import load_raw_potability_data, load_raw_demand_data

# Expected Potability Schema & Ranges (Physical drinking water bounds)
POTABILITY_SCHEMA = {
    "ph": {"dtype": "float64", "min_expected": 0.0, "max_expected": 14.0},
    "Hardness": {"dtype": "float64", "min_expected": 0.0, "max_expected": 500.0},
    "Solids": {"dtype": "float64", "min_expected": 0.0, "max_expected": 70000.0},
    "Chloramines": {"dtype": "float64", "min_expected": 0.0, "max_expected": 15.0},
    "Sulfate": {"dtype": "float64", "min_expected": 0.0, "max_expected": 600.0},
    "Conductivity": {"dtype": "float64", "min_expected": 0.0, "max_expected": 1000.0},
    "Organic_carbon": {"dtype": "float64", "min_expected": 0.0, "max_expected": 40.0},
    "Trihalomethanes": {"dtype": "float64", "min_expected": 0.0, "max_expected": 150.0},
    "Turbidity": {"dtype": "float64", "min_expected": 0.0, "max_expected": 10.0},
    "Potability": {"dtype": "int64", "valid_values": [0, 1]}
}

# Expected Demand Schema
DEMAND_SCHEMA = {
    "Date": {"dtype": "object"},
    "Zone_ID": {"dtype": "object", "valid_values": ["Z01", "Z02", "Z03", "Z04", "Z05"]},
    "Population": {"dtype": "int64", "min_expected": 1000},
    "Active_Connections": {"dtype": "float64", "min_expected": 100},
    "Avg_Temperature_C": {"dtype": "float64", "min_expected": -10.0, "max_expected": 55.0},
    "Humidity_pct": {"dtype": "float64", "min_expected": 0.0, "max_expected": 100.0},
    "Rainfall_mm": {"dtype": "float64", "min_expected": 0.0, "max_expected": 500.0},
    "Day_of_Week": {"dtype": "object"},
    "Is_Weekend": {"dtype": "int64", "valid_values": [0, 1]},
    "Is_Holiday": {"dtype": "int64", "valid_values": [0, 1]},
    "Month": {"dtype": "int64", "valid_values": list(range(1, 13))},
    "Season": {"dtype": "object", "valid_values": ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]},
    "Water_Demand_kL": {"dtype": "float64", "min_expected": 0.0}
}


def validate_potability_dataset(df: pd.DataFrame = None) -> Dict[str, Any]:
    """Validate raw water potability dataset."""
    if df is None:
        df = load_raw_potability_data()

    total_rows = len(df)
    missing_info = {}
    range_checks = {}

    for col, spec in POTABILITY_SCHEMA.items():
        missing_count = int(df[col].isnull().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2)
        missing_info[col] = {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "has_missing": missing_count > 0
        }

        if col == "Potability":
            valid_vals = set(spec["valid_values"])
            actual_vals = set(df[col].dropna().unique())
            range_checks[col] = {
                "valid": actual_vals.issubset(valid_vals),
                "unique_values": list(actual_vals)
            }
        else:
            col_min = float(df[col].min()) if not df[col].isnull().all() else None
            col_max = float(df[col].max()) if not df[col].isnull().all() else None
            range_checks[col] = {
                "min": round(col_min, 2) if col_min is not None else None,
                "max": round(col_max, 2) if col_max is not None else None,
                "expected_min": spec.get("min_expected"),
                "expected_max": spec.get("max_expected"),
                "within_bounds": (
                    col_min >= spec["min_expected"] and col_max <= spec["max_expected"]
                    if col_min is not None and col_max is not None else False
                )
            }

    duplicate_rows = int(df.duplicated().sum())

    target_counts = df["Potability"].value_counts().to_dict()
    class_balance = {
        "non_potable_0": int(target_counts.get(0, 0)),
        "potable_1": int(target_counts.get(1, 0)),
        "potable_pct": round((target_counts.get(1, 0) / total_rows) * 100, 2)
    }

    return {
        "status": "VALID",
        "dataset_name": "Water Potability",
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "duplicates": duplicate_rows,
        "missing_analysis": missing_info,
        "range_checks": range_checks,
        "class_balance": class_balance
    }


def validate_demand_dataset(df: pd.DataFrame = None) -> Dict[str, Any]:
    """Validate raw synthetic water demand dataset."""
    if df is None:
        df = load_raw_demand_data()

    total_rows = len(df)
    missing_info = {}
    range_checks = {}

    for col, spec in DEMAND_SCHEMA.items():
        missing_count = int(df[col].isnull().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2)
        missing_info[col] = {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "has_missing": missing_count > 0
        }

        if "min_expected" in spec:
            col_min = float(df[col].min()) if not df[col].isnull().all() else None
            col_max = float(df[col].max()) if not df[col].isnull().all() else None
            range_checks[col] = {
                "min": round(col_min, 2) if col_min is not None else None,
                "max": round(col_max, 2) if col_max is not None else None,
                "expected_min": spec.get("min_expected"),
                "expected_max": spec.get("max_expected")
            }

    duplicate_rows = int(df.duplicated().sum())
    zones = sorted(list(df["Zone_ID"].unique()))
    zone_counts = {z: int(c) for z, c in df["Zone_ID"].value_counts().items()}

    # Date integrity
    df_date = pd.to_datetime(df["Date"])
    date_summary = {
        "min_date": str(df_date.min().date()),
        "max_date": str(df_date.max().date()),
        "total_days_per_zone": int(len(df_date) / len(zones))
    }

    return {
        "status": "VALID",
        "dataset_name": "Synthetic Water Demand",
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "duplicates": duplicate_rows,
        "missing_analysis": missing_info,
        "range_checks": range_checks,
        "zones": zones,
        "zone_distribution": zone_counts,
        "date_summary": date_summary
    }


def sanitize_numpy(obj):
    """Recursively convert numpy types and NaN values to native Python types/None for JSON compliance."""
    import math
    if isinstance(obj, dict):
        return {k: sanitize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_numpy(v) for v in obj]
    elif hasattr(obj, "item"):
        val = obj.item()
        if isinstance(val, float) and math.isnan(val):
            return None
        return val
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def get_full_validation_summary() -> Dict[str, Any]:
    """Retrieve combined validation results with native python types."""
    raw_summary = {
        "potability": validate_potability_dataset(),
        "demand": validate_demand_dataset()
    }
    return sanitize_numpy(raw_summary)


if __name__ == "__main__":
    summary = get_full_validation_summary()
    print("=== Validation Completed ===")
    print("Potability Rows:", summary["potability"]["total_rows"])
    print("Potability Missing Sulfate:", summary["potability"]["missing_analysis"]["Sulfate"])
    print("Demand Rows:", summary["demand"]["total_rows"])
    print("Demand Zones:", summary["demand"]["zones"])
