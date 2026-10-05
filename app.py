"""
AquaSense Application Entry Point
FastAPI Web and REST Service for AI-Based Water Quality Assessment and Demand Forecasting.
"""
import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.data.loader import get_dataset_summaries, get_data_loading_inspection
from src.data.validator import get_full_validation_summary, sanitize_numpy
from src.eda.potability_eda import compute_potability_eda
from src.eda.demand_eda import compute_demand_eda
from src.preprocessing.potability_preprocessor import preprocess_potability_data
from src.preprocessing.demand_preprocessor import preprocess_demand_data

# Initialize Application
app = FastAPI(
    title="AquaSense AI System",
    description="AI-Based Water Quality Assessment and Demand Forecasting System",
    version="0.2.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "results")

# Mount Static Assets & Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def load_or_compute_json(filename: str, compute_func):
    """Utility to load cached summary JSON or compute dynamically."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return compute_func()


@app.get("/", response_class=HTMLResponse)
async def dashboard_view(request: Request):
    """System Dashboard View."""
    dataset_summary = get_dataset_summaries()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "active_page": "dashboard",
            "summary": dataset_summary
        }
    )


@app.get("/data_loading", response_class=HTMLResponse)
async def data_loading_view(request: Request):
    """Data Loading & Schema Inspection View."""
    inspection = get_data_loading_inspection()
    return templates.TemplateResponse(
        request=request,
        name="data_loading.html",
        context={
            "active_page": "data_loading",
            "inspection": inspection
        }
    )


@app.get("/eda", response_class=HTMLResponse)
async def eda_view(request: Request):
    """Data & Insights (Exploratory Data Analysis) View."""
    potability_eda = load_or_compute_json("potability_eda_summary.json", compute_potability_eda)
    demand_eda = load_or_compute_json("demand_eda_summary.json", compute_demand_eda)
    return templates.TemplateResponse(
        request=request,
        name="eda.html",
        context={
            "active_page": "eda",
            "potability_eda": potability_eda,
            "demand_eda": demand_eda
        }
    )


@app.get("/preprocessing", response_class=HTMLResponse)
async def preprocessing_view(request: Request):
    """Data Preparation & Feature Pipeline View."""
    potability_prep = load_or_compute_json("potability_preprocessing_summary.json", preprocess_potability_data)
    demand_prep = load_or_compute_json("demand_preprocessing_summary.json", preprocess_demand_data)
    return templates.TemplateResponse(
        request=request,
        name="preprocessing.html",
        context={
            "active_page": "preprocessing",
            "potability_prep": potability_prep,
            "demand_prep": demand_prep
        }
    )


@app.get("/linear_models", response_class=HTMLResponse)
async def linear_models_view(request: Request):
    """Linear & Regularized Regression/Classification View."""
    from src.models.linear_demand import train_evaluate_demand_linear_models
    from src.models.linear_potability import train_evaluate_potability_linear_models

    demand_linear = load_or_compute_json("linear_demand_results.json", train_evaluate_demand_linear_models)
    potability_linear = load_or_compute_json("linear_potability_results.json", train_evaluate_potability_linear_models)

    return templates.TemplateResponse(
        request=request,
        name="linear_models.html",
        context={
            "active_page": "linear_models",
            "demand_linear": demand_linear,
            "potability_linear": potability_linear
        }
    )


@app.get("/tree_models", response_class=HTMLResponse)
async def tree_models_view(request: Request):
    """Tree & Ensemble Models View."""
    from src.models.tree_models import run_all_tree_models
    tree_data = load_or_compute_json("tree_models_results.json", run_all_tree_models)
    return templates.TemplateResponse(
        request=request,
        name="tree_models.html",
        context={
            "active_page": "tree_models",
            "tree_data": tree_data
        }
    )


@app.get("/clustering", response_class=HTMLResponse)
async def clustering_view(request: Request):
    """Pattern Discovery (Clustering & PCA) View."""
    from src.models.clustering_pca import run_all_unsupervised
    clustering_data = load_or_compute_json("clustering_results.json", run_all_unsupervised)
    return templates.TemplateResponse(
        request=request,
        name="clustering.html",
        context={
            "active_page": "clustering",
            "clustering_data": clustering_data
        }
    )


@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation_view(request: Request):
    """Model Evaluation & Benchmark View."""
    from src.models.evaluation_tuning import run_full_evaluation
    eval_data = load_or_compute_json("model_evaluation_results.json", run_full_evaluation)
    return templates.TemplateResponse(
        request=request,
        name="evaluation.html",
        context={
            "active_page": "evaluation",
            "eval_data": eval_data
        }
    )


@app.get("/prediction", response_class=HTMLResponse)
async def prediction_view(request: Request):
    """Interactive Inference & Prediction View."""
    return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={"active_page": "prediction"}
    )


@app.get("/deployment", response_class=HTMLResponse)
async def deployment_view(request: Request):
    """Production Deployment Architecture View."""
    return templates.TemplateResponse(
        request=request,
        name="deployment.html",
        context={"active_page": "deployment"}
    )


@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_view(request: Request):
    """Model Monitoring & Drift Detection View."""
    return templates.TemplateResponse(
        request=request,
        name="monitoring.html",
        context={"active_page": "monitoring"}
    )


# --- REST API Endpoints ---

@app.get("/api/v1/health")
async def health_check():
    """System Health and Data Verification Endpoint."""
    try:
        summary = get_dataset_summaries()
        return {
            "status": "healthy",
            "system": "AquaSense AI Engine",
            "version": "0.2.0",
            "datasets": {
                "potability_samples": summary["potability"]["rows"],
                "demand_records": summary["demand"]["rows"],
                "status": "ready"
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }


@app.get("/api/v1/validation")
async def validation_api():
    """Dataset Schema and Integrity Validation Endpoint."""
    return get_full_validation_summary()


@app.get("/api/v1/data_loading")
async def data_loading_api():
    """Data Loading Metadata and Live Preview API Endpoint."""
    return sanitize_numpy(get_data_loading_inspection())


@app.get("/api/v1/eda/potability")
async def eda_potability_api():
    """Water Potability Statistical and EDA Summary Endpoint."""
    return load_or_compute_json("potability_eda_summary.json", compute_potability_eda)


@app.get("/api/v1/eda/demand")
async def eda_demand_api():
    """Water Demand Statistical and EDA Summary Endpoint."""
    return load_or_compute_json("demand_eda_summary.json", compute_demand_eda)


@app.get("/api/v1/preprocessing/summary")
async def preprocessing_summary_api():
    """Data Preprocessing & Split Integrity Audit Endpoint."""
    pot_summary = load_or_compute_json("potability_preprocessing_summary.json", preprocess_potability_data)
    dem_summary = load_or_compute_json("demand_preprocessing_summary.json", preprocess_demand_data)
    return sanitize_numpy({
        "potability": pot_summary,
        "demand": dem_summary
    })


@app.get("/api/v1/models/linear/demand")
async def linear_demand_api():
    """Water Demand Forecasting Linear Models Evaluation API."""
    from src.models.linear_demand import train_evaluate_demand_linear_models
    return sanitize_numpy(load_or_compute_json("linear_demand_results.json", train_evaluate_demand_linear_models))


@app.get("/api/v1/models/linear/potability")
async def linear_potability_api():
    """Water Potability Assessment Linear Models Evaluation API."""
    from src.models.linear_potability import train_evaluate_potability_linear_models
    return sanitize_numpy(load_or_compute_json("linear_potability_results.json", train_evaluate_potability_linear_models))


@app.get("/api/v1/models/tree/potability")
async def tree_potability_api():
    """Tree & Ensemble Potability Classification Results API."""
    from src.models.tree_models import run_all_tree_models
    data = load_or_compute_json("tree_models_results.json", run_all_tree_models)
    return sanitize_numpy(data.get("potability", data))


@app.get("/api/v1/models/tree/demand")
async def tree_demand_api():
    """Tree & Ensemble Demand Regression Results API."""
    from src.models.tree_models import run_all_tree_models
    data = load_or_compute_json("tree_models_results.json", run_all_tree_models)
    return sanitize_numpy(data.get("demand", data))


@app.get("/api/v1/clustering")
async def clustering_api():
    """Unsupervised Clustering & PCA Results API."""
    from src.models.clustering_pca import run_all_unsupervised
    data = load_or_compute_json("clustering_results.json", run_all_unsupervised)
    return sanitize_numpy(data)


@app.get("/api/v1/evaluation")
async def evaluation_api():
    """Full Model Evaluation & Benchmark Comparison API."""
    from src.models.evaluation_tuning import run_full_evaluation
    data = load_or_compute_json("model_evaluation_results.json", run_full_evaluation)
    return sanitize_numpy(data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
