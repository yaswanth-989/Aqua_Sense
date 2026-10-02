"""
AquaSense Application Entry Point
FastAPI Web and REST Service for AI-Based Water Quality Assessment and Demand Forecasting.
"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.data.loader import get_dataset_summaries

# Initialize Application
app = FastAPI(
    title="AquaSense AI System",
    description="AI-Based Water Quality Assessment and Demand Forecasting System",
    version="0.1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount Static Assets & Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


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


@app.get("/eda", response_class=HTMLResponse)
async def eda_view(request: Request):
    """Data & Insights (Exploratory Data Analysis) View."""
    return templates.TemplateResponse(
        request=request,
        name="eda.html",
        context={"active_page": "eda"}
    )


@app.get("/preprocessing", response_class=HTMLResponse)
async def preprocessing_view(request: Request):
    """Data Preparation & Feature Pipeline View."""
    return templates.TemplateResponse(
        request=request,
        name="preprocessing.html",
        context={"active_page": "preprocessing"}
    )


@app.get("/linear_models", response_class=HTMLResponse)
async def linear_models_view(request: Request):
    """Linear & Regularized Regression/Classification View."""
    return templates.TemplateResponse(
        request=request,
        name="linear_models.html",
        context={"active_page": "linear_models"}
    )


@app.get("/tree_models", response_class=HTMLResponse)
async def tree_models_view(request: Request):
    """Tree & Ensemble Models View."""
    return templates.TemplateResponse(
        request=request,
        name="tree_models.html",
        context={"active_page": "tree_models"}
    )


@app.get("/clustering", response_class=HTMLResponse)
async def clustering_view(request: Request):
    """Pattern Discovery (Clustering & PCA) View."""
    return templates.TemplateResponse(
        request=request,
        name="clustering.html",
        context={"active_page": "clustering"}
    )


@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation_view(request: Request):
    """Model Evaluation & Benchmark View."""
    return templates.TemplateResponse(
        request=request,
        name="evaluation.html",
        context={"active_page": "evaluation"}
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
            "version": "0.1.0",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
