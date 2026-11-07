"""Experiments API endpoints."""

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

router = APIRouter(prefix="/experiments", tags=["experiments"])

EXPERIMENTS_DIR = Path("./experiments")


class ExperimentRequest(BaseModel):
    """Request to start an experiment."""
    symbol: str = Field(description="Symbol to run experiment on (SPY, QQQ, NVDA)")
    start_date: str = Field(default="2018-01-01", description="Start date for data")
    end_date: str | None = Field(default=None, description="End date (defaults to today)")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    n_bootstrap: int = Field(default=500, ge=100, le=5000, description="Bootstrap samples")
    xgb_n_estimators: int = Field(default=100, ge=10, le=500)
    xgb_max_depth: int = Field(default=4, ge=2, le=10)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symbol": "SPY",
                    "start_date": "2018-01-01",
                    "seed": 42,
                    "n_bootstrap": 500,
                }
            ]
        }
    }


class ExperimentStatus(BaseModel):
    """Experiment status response."""
    experiment_id: str
    status: Literal["running", "completed", "error"]
    symbol: str
    message: str | None = None


class ExperimentSummary(BaseModel):
    """Summary of an experiment."""
    experiment_id: str
    symbol: str
    created_at: str
    status: str
    n_models: int


# In-memory tracking of running experiments
_running_experiments: dict[str, str] = {}


def _run_experiment_sync(request: ExperimentRequest) -> str:
    """Run experiment synchronously (for background task)."""
    from volatilitylab.ml.runner import run_experiment, RunConfig
    
    config = RunConfig(
        symbol=request.symbol,
        start_date=request.start_date,
        end_date=request.end_date,
        seed=request.seed,
        n_bootstrap=request.n_bootstrap,
        xgb_n_estimators=request.xgb_n_estimators,
        xgb_max_depth=request.xgb_max_depth,
    )
    
    result = run_experiment(config, str(EXPERIMENTS_DIR))
    return result.experiment_id


@router.post("", response_model=ExperimentStatus)
async def start_experiment(
    request: ExperimentRequest,
    background_tasks: BackgroundTasks,
) -> ExperimentStatus:
    """
    Start a new volatility forecasting experiment.
    
    This runs in the background. Poll GET /experiments/{id} for status.
    """
    # Generate a temporary ID
    import uuid
    from datetime import datetime, timezone
    
    temp_id = f"{request.symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    _running_experiments[temp_id] = "running"
    
    # Run in background
    def run_and_update():
        try:
            result_id = _run_experiment_sync(request)
            _running_experiments[temp_id] = f"completed:{result_id}"
        except Exception as e:
            _running_experiments[temp_id] = f"error:{str(e)}"
    
    background_tasks.add_task(run_and_update)
    
    return ExperimentStatus(
        experiment_id=temp_id,
        status="running",
        symbol=request.symbol,
        message="Experiment started. Poll this endpoint for status.",
    )


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict:
    """
    Get experiment results by ID.
    
    Returns full results if completed, status if running.
    """
    from volatilitylab.ml.experiments import ExperimentTracker
    
    # Check if still running
    if experiment_id in _running_experiments:
        status = _running_experiments[experiment_id]
        if status == "running":
            return {
                "experiment_id": experiment_id,
                "status": "running",
                "message": "Experiment still in progress",
            }
        elif status.startswith("completed:"):
            # Get the real experiment ID
            real_id = status.split(":", 1)[1]
            experiment_id = real_id
            del _running_experiments[experiment_id]
        elif status.startswith("error:"):
            error_msg = status.split(":", 1)[1]
            return {
                "experiment_id": experiment_id,
                "status": "error",
                "error": error_msg,
            }
    
    # Load from disk
    tracker = ExperimentTracker(EXPERIMENTS_DIR)
    result = tracker.get_experiment(experiment_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    # Convert to dict for JSON response
    from dataclasses import asdict
    return asdict(result)


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments(
    symbol: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[ExperimentSummary]:
    """List all experiments, optionally filtered by symbol or status."""
    from volatilitylab.ml.experiments import ExperimentTracker
    
    tracker = ExperimentTracker(EXPERIMENTS_DIR)
    experiments = tracker.list_experiments(symbol=symbol, status=status)
    
    return [
        ExperimentSummary(**exp)
        for exp in experiments[:limit]
    ]


@router.get("/{experiment_id}/predictions")
async def get_predictions(experiment_id: str) -> dict:
    """Get predictions from an experiment."""
    import pandas as pd
    
    pred_path = EXPERIMENTS_DIR / experiment_id / "predictions.parquet"
    
    if not pred_path.exists():
        raise HTTPException(status_code=404, detail="Predictions not found")
    
    df = pd.read_parquet(pred_path)
    
    return {
        "experiment_id": experiment_id,
        "n_samples": len(df),
        "columns": list(df.columns),
        "predictions": df.to_dict(orient="records"),
    }


@router.get("/{experiment_id}/summary")
async def get_experiment_summary(experiment_id: str) -> dict:
    """Get a formatted summary of experiment results."""
    from volatilitylab.ml.experiments import ExperimentTracker
    
    tracker = ExperimentTracker(EXPERIMENTS_DIR)
    result = tracker.get_experiment(experiment_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    # Build summary
    summary = {
        "experiment_id": result.experiment_id,
        "symbol": result.symbol,
        "status": result.status,
        "created_at": result.created_at,
    }
    
    # Extract metrics by model and horizon
    metrics_table = []
    for model_result in result.models:
        name = model_result.model_name
        if '_5d' in name:
            horizon = 5
            metrics = model_result.metrics_5d
        else:
            horizon = 20
            metrics = model_result.metrics_20d
        
        if metrics:
            metrics_table.append({
                "model": model_result.model_type,
                "horizon": f"{horizon}d",
                "mae": metrics.get('mae'),
                "rmse": metrics.get('rmse'),
                "qlike": metrics.get('qlike'),
            })
    
    summary["metrics"] = metrics_table
    
    # Hypothesis test results
    h_tests = []
    for h in result.hypothesis_tests:
        h_tests.append({
            "hypothesis": h.hypothesis,
            "result": h.result,
        })
    summary["hypothesis_tests"] = h_tests
    
    # DM comparisons
    summary["dm_comparisons_5d"] = result.dm_comparisons_5d
    summary["dm_comparisons_20d"] = result.dm_comparisons_20d
    
    return summary
