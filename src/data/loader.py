"""
AquaSense Data Ingestion & Inspection Module
Loads raw datasets without alteration, computes schema metadata, and extracts preview rows.
"""
import os
import pandas as pd
from typing import Dict, Any

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


def get_data_loading_inspection() -> Dict[str, Any]:
    """Detailed metadata and live data preview for the Data Loading page."""
    pot_df = load_raw_potability_data()
    dem_df = load_raw_demand_data()

    # Potability column metadata
    pot_cols = []
    for col in pot_df.columns:
        null_cnt = int(pot_df[col].isnull().sum())
        pot_cols.append({
            "name": col,
            "dtype": str(pot_df[col].dtype),
            "non_null_count": int(pot_df[col].notnull().sum()),
            "missing_count": null_cnt,
            "missing_pct": round((null_cnt / len(pot_df)) * 100, 2),
            "unique_count": int(pot_df[col].nunique())
        })

    # Demand column metadata
    dem_cols = []
    for col in dem_df.columns:
        null_cnt = int(dem_df[col].isnull().sum())
        dem_cols.append({
            "name": col,
            "dtype": str(dem_df[col].dtype),
            "non_null_count": int(dem_df[col].notnull().sum()),
            "missing_count": null_cnt,
            "missing_pct": round((null_cnt / len(dem_df)) * 100, 2),
            "unique_count": int(dem_df[col].nunique())
        })

    # Preview rows (convert NaN to None for clean JSON/Jinja rendering)
    pot_preview = pot_df.head(10).round(3).where(pd.notnull(pot_df), None).to_dict(orient="records")
    dem_preview = dem_df.head(10).round(2).where(pd.notnull(dem_df), None).to_dict(orient="records")

    return {
        "potability": {
            "total_rows": len(pot_df),
            "total_columns": len(pot_df.columns),
            "memory_kb": round(float(pot_df.memory_usage(deep=True).sum()) / 1024, 1),
            "duplicates": int(pot_df.duplicated().sum()),
            "columns_info": pot_cols,
            "preview_rows": pot_preview,
            "column_names": list(pot_df.columns)
        },
        "demand": {
            "total_rows": len(dem_df),
            "total_columns": len(dem_df.columns),
            "memory_kb": round(float(dem_df.memory_usage(deep=True).sum()) / 1024, 1),
            "duplicates": int(dem_df.duplicated().sum()),
            "columns_info": dem_cols,
            "preview_rows": dem_preview,
            "column_names": list(dem_df.columns)
        }
    }


if __name__ == "__main__":
    insp = get_data_loading_inspection()
    print("Potability rows:", insp["potability"]["total_rows"])
    print("Potability preview count:", len(insp["potability"]["preview_rows"]))
    print("Demand rows:", insp["demand"]["total_rows"])
