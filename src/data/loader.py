"""
AquaSense Data Ingestion & Validation Module
Loads raw datasets without alteration and validates integrity.
"""
import os
import pandas as pd
from typing import Dict, Any, Tuple

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

POTABILITY_PATH = os.path.join(RAW_DATA_DIR, "water_potability.csv")
DEMAND_PATH = os.path.join(RAW_DATA_DIR, "AquaSense_Synthetic_Water_Demand_5475.csv")


def load_raw_potability_data() -> pd.DataFrame:
    """Load the raw water potability dataset without alteration."""
    if not os.path.exists(POTABILITY_PATH):
        raise FileNotFoundError(f"Water potability dataset not found at {POTABILITY_PATH}")
    return pd.read_csv(POTABILITY_PATH)


def load_raw_demand_data() -> pd.DataFrame:
    """Load the raw synthetic water demand dataset without alteration."""
    if not os.path.exists(DEMAND_PATH):
        raise FileNotFoundError(f"Water demand dataset not found at {DEMAND_PATH}")
    return pd.read_csv(DEMAND_PATH)


def get_dataset_summaries() -> Dict[str, Any]:
    """Retrieve verified summary statistics for ingested datasets."""
    potability_df = load_raw_potability_data()
    demand_df = load_raw_demand_data()

    return {
        "potability": {
            "rows": int(len(potability_df)),
            "columns": list(potability_df.columns),
            "target_distribution": {
                str(k): int(v) for k, v in potability_df["Potability"].value_counts().items()
            },
            "missing_counts": {
                col: int(cnt) for col, cnt in potability_df.isnull().sum().items() if cnt > 0
            }
        },
        "demand": {
            "rows": int(len(demand_df)),
            "columns": list(demand_df.columns),
            "date_range": [str(demand_df["Date"].min()), str(demand_df["Date"].max())],
            "zones": sorted(list(demand_df["Zone_ID"].unique())),
            "missing_counts": {
                col: int(cnt) for col, cnt in demand_df.isnull().sum().items() if cnt > 0
            }
        }
    }


if __name__ == "__main__":
    summary = get_dataset_summaries()
    print("Potability samples:", summary["potability"]["rows"])
    print("Demand samples:", summary["demand"]["rows"])
    print("Zones:", summary["demand"]["zones"])
