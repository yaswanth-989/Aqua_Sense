# AquaSense: AI-Based Water Quality Assessment and Demand Forecasting System

AquaSense is an AI-based water management system designed to evaluate drinking water potability and forecast next-day municipal water consumption. The system provides utility operators and municipal authorities with predictive analytics to optimize distribution, maintain compliance standards, and prevent water shortages.

---

## 1. Project Overview & Tasks

1. **Water Potability Prediction (Classification)**
   * **Dataset:** Real physicochemical water potability dataset (`data/raw/water_potability.csv`).
   * **Target:** `Potability` (Binary: `0 = Not Potable`, `1 = Potable`).
   * **Input Features:** pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity.

2. **Water Demand Forecasting (Time-Series Regression)**
   * **Dataset:** Synthetic municipal water demand dataset (`data/raw/AquaSense_Synthetic_Water_Demand_5475.csv`).
   * **Target:** `Water_Demand_kL` (Daily water demand in kiloliters).
   * **Input Features:** Zone ID, Population, Active Connections, Temperature, Humidity, Rainfall, Calendar features, historical lags and rolling trends.

> **Data Disclosure:** As specified in the project requirements, the municipal water demand dataset is a **synthetic dataset** designed for time-series forecasting research. It is not real field/utility telemetry and is strictly documented as synthetic. The water potability dataset consists of real water quality observations.

---

## 2. Directory Structure

```text
AquaSense/
│
├── data/
│   ├── raw/                  # Unaltered raw source datasets
│   └── processed/            # Leakage-safe preprocessed datasets
│
├── templates/                # HTML5 templates (Jinja2)
│   ├── base.html             # Persistent navy sidebar layout
│   ├── index.html            # System dashboard
│   ├── eda.html              # Data & Insights
│   ├── preprocessing.html    # Data Preparation
│   ├── linear_models.html    # Regression & Classification
│   ├── tree_models.html      # Tree & Ensemble Models
│   ├── clustering.html       # Pattern Discovery
│   ├── evaluation.html       # Model Evaluation
│   ├── prediction.html       # Prediction Portal
│   ├── deployment.html       # Deployment Architecture
│   └── monitoring.html       # Production Monitoring
│
├── static/
│   ├── css/
│   │   └── style.css         # AquaSense Design System CSS
│   ├── js/
│   │   └── script.js         # Shared client-side scripts & Lucide icons
│   └── images/               # Generated charts and diagrams
│
├── src/
│   ├── data/                 # Ingestion & validation
│   ├── preprocessing/        # Imputation, scaling, lag features
│   ├── eda/                  # Exploratory statistical analysis
│   ├── models/               # Training pipelines (Linear, Trees, Clustering)
│   ├── evaluation/           # Metrics, curves, calibration
│   ├── deployment/           # REST schemas and API handlers
│   └── monitoring/           # Data drift & distribution tracking
│
├── models/
│   ├── potability/           # Serialized potability pipeline artifacts
│   └── demand/               # Serialized demand forecasting artifacts
│
├── reports/
│   ├── figures/              # Generated EDA and evaluation plots
│   └── results/              # Evaluation metrics and tables
│
├── app.py                    # FastAPI application entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 3. Running the Application

### Prerequisites
* Python 3.10+
* Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Launch Development Server
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to: `http://127.0.0.1:8000`

### Verify Health API
```bash
curl http://127.0.0.1:8000/api/v1/health
```
