"""
Experiment Tracking

Tracks ML experiments with:
- Seed, config, and data hash for reproducibility
- Model parameters and metrics
- Walk-forward validation results
- Statistical test results
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    """Results for a single model."""

    model_name: str
    model_type: str
    params: dict
    metrics_5d: dict
    metrics_20d: dict
    feature_importance: dict | None = None


@dataclass
class HypothesisTestResult:
    """Results of hypothesis testing."""

    hypothesis: str
    result: str
    details: dict
    confidence_level: float


@dataclass
class ExperimentResult:
    """Complete experiment result."""

    experiment_id: str
    created_at: str
    symbol: str
    config: dict
    data_hash: str
    seed: int

    # Model results
    models: list[ModelResult]

    # DM test results
    dm_comparisons_5d: list[dict]
    dm_comparisons_20d: list[dict]

    # Bootstrap CIs
    bootstrap_cis: dict

    # Hypothesis tests
    hypothesis_tests: list[HypothesisTestResult]

    # Regime analysis
    regime_analysis: dict | None = None

    # Predictions
    predictions_path: str | None = None

    # Status
    status: str = "completed"
    error: str | None = None


def compute_data_hash(data: pd.DataFrame) -> str:
    """Compute hash of data for reproducibility tracking."""
    # Hash the index and values
    index_str = str(data.index.tolist())
    values_str = np.array2string(data.values, precision=8)
    combined = index_str + values_str
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class ExperimentTracker:
    """
    Track and persist ML experiments.

    Example
    -------
    >>> tracker = ExperimentTracker('./experiments')
    >>> exp_id = tracker.start_experiment('SPY', config, seed=42)
    >>> tracker.log_model_result(exp_id, model_result)
    >>> tracker.finish_experiment(exp_id)
    >>> result = tracker.get_experiment(exp_id)
    """

    def __init__(self, base_dir: str | Path = "./experiments"):
        """
        Initialize experiment tracker.

        Parameters
        ----------
        base_dir : str or Path
            Base directory for experiment artifacts
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._active_experiments: dict[str, dict] = {}

    def start_experiment(
        self,
        symbol: str,
        config: dict,
        seed: int,
        data: pd.DataFrame | None = None,
    ) -> str:
        """
        Start a new experiment.

        Returns
        -------
        str
            Experiment ID
        """
        exp_id = f"{symbol}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        data_hash = compute_data_hash(data) if data is not None else "unknown"

        self._active_experiments[exp_id] = {
            "experiment_id": exp_id,
            "created_at": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "config": config,
            "data_hash": data_hash,
            "seed": seed,
            "models": [],
            "dm_comparisons_5d": [],
            "dm_comparisons_20d": [],
            "bootstrap_cis": {},
            "hypothesis_tests": [],
            "regime_analysis": None,
            "predictions_path": None,
            "status": "running",
            "error": None,
        }

        return exp_id

    def log_model_result(
        self,
        experiment_id: str,
        model_result: ModelResult,
    ) -> None:
        """Log results for a model."""
        if experiment_id not in self._active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        self._active_experiments[experiment_id]["models"].append(asdict(model_result))

    def log_dm_comparison(
        self,
        experiment_id: str,
        horizon: int,
        comparison: dict,
    ) -> None:
        """Log DM test comparison."""
        if experiment_id not in self._active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        key = f"dm_comparisons_{horizon}d"
        self._active_experiments[experiment_id][key].append(comparison)

    def log_bootstrap_ci(
        self,
        experiment_id: str,
        model: str,
        horizon: int,
        metric: str,
        ci: dict,
    ) -> None:
        """Log bootstrap confidence interval."""
        if experiment_id not in self._active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        key = f"{model}_{horizon}d_{metric}"
        self._active_experiments[experiment_id]["bootstrap_cis"][key] = ci

    def log_hypothesis_test(
        self,
        experiment_id: str,
        hypothesis: str,
        result: str,
        details: dict,
        confidence_level: float = 0.95,
    ) -> None:
        """Log hypothesis test result."""
        if experiment_id not in self._active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        self._active_experiments[experiment_id]["hypothesis_tests"].append(
            {
                "hypothesis": hypothesis,
                "result": result,
                "details": details,
                "confidence_level": confidence_level,
            }
        )

    def log_regime_analysis(
        self,
        experiment_id: str,
        analysis: dict,
    ) -> None:
        """Log regime-dependent analysis."""
        if experiment_id not in self._active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        self._active_experiments[experiment_id]["regime_analysis"] = analysis

    def save_predictions(
        self,
        experiment_id: str,
        predictions: pd.DataFrame,
    ) -> str:
        """Save predictions to file."""
        if experiment_id not in self._active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)

        pred_path = exp_dir / "predictions.parquet"
        predictions.to_parquet(pred_path)

        self._active_experiments[experiment_id]["predictions_path"] = str(pred_path)
        return str(pred_path)

    def finish_experiment(
        self,
        experiment_id: str,
        error: str | None = None,
    ) -> ExperimentResult:
        """
        Finish and persist experiment.

        Returns
        -------
        ExperimentResult
            Final experiment result
        """
        if experiment_id not in self._active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp_data = self._active_experiments[experiment_id]
        exp_data["status"] = "error" if error else "completed"
        exp_data["error"] = error

        # Convert to ExperimentResult
        result = ExperimentResult(
            experiment_id=exp_data["experiment_id"],
            created_at=exp_data["created_at"],
            symbol=exp_data["symbol"],
            config=exp_data["config"],
            data_hash=exp_data["data_hash"],
            seed=exp_data["seed"],
            models=[ModelResult(**m) for m in exp_data["models"]],
            dm_comparisons_5d=exp_data["dm_comparisons_5d"],
            dm_comparisons_20d=exp_data["dm_comparisons_20d"],
            bootstrap_cis=exp_data["bootstrap_cis"],
            hypothesis_tests=[HypothesisTestResult(**h) for h in exp_data["hypothesis_tests"]],
            regime_analysis=exp_data["regime_analysis"],
            predictions_path=exp_data["predictions_path"],
            status=exp_data["status"],
            error=exp_data["error"],
        )

        # Persist to disk
        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)

        with open(exp_dir / "result.json", "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Clean up active experiment
        del self._active_experiments[experiment_id]

        return result

    def get_experiment(self, experiment_id: str) -> ExperimentResult | None:
        """Load experiment from disk."""
        exp_dir = self.base_dir / experiment_id
        result_path = exp_dir / "result.json"

        if not result_path.exists():
            return None

        with open(result_path) as f:
            data = json.load(f)

        return ExperimentResult(
            experiment_id=data["experiment_id"],
            created_at=data["created_at"],
            symbol=data["symbol"],
            config=data["config"],
            data_hash=data["data_hash"],
            seed=data["seed"],
            models=[ModelResult(**m) for m in data["models"]],
            dm_comparisons_5d=data["dm_comparisons_5d"],
            dm_comparisons_20d=data["dm_comparisons_20d"],
            bootstrap_cis=data["bootstrap_cis"],
            hypothesis_tests=[HypothesisTestResult(**h) for h in data["hypothesis_tests"]],
            regime_analysis=data.get("regime_analysis"),
            predictions_path=data.get("predictions_path"),
            status=data["status"],
            error=data.get("error"),
        )

    def list_experiments(
        self,
        symbol: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """List all experiments."""
        experiments = []

        for exp_dir in self.base_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            result_path = exp_dir / "result.json"
            if not result_path.exists():
                continue

            with open(result_path) as f:
                data = json.load(f)

            if symbol and data["symbol"] != symbol:
                continue
            if status and data["status"] != status:
                continue

            experiments.append(
                {
                    "experiment_id": data["experiment_id"],
                    "symbol": data["symbol"],
                    "created_at": data["created_at"],
                    "status": data["status"],
                    "n_models": len(data["models"]),
                }
            )

        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x["created_at"], reverse=True)

        return experiments
