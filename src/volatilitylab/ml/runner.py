"""
Experiment Runner

Runs complete volatility forecasting experiments with:
- Walk-forward validation
- Multiple models
- Statistical testing (H1, H3)
- Experiment tracking
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volatilitylab.ml.evaluation import (
    block_bootstrap_ci,
    compare_models_dm,
    compute_metrics,
    diebold_mariano_test,
)
from volatilitylab.ml.experiments import (
    ExperimentResult,
    ExperimentTracker,
    ModelResult,
)
from volatilitylab.ml.features import (
    FeatureEngine,
    VolatilityFeatures,
    load_price_data,
    load_vix_data,
)
from volatilitylab.ml.models import (
    EWMAModel,
    GARCHModel,
    HistoricalVolModel,
    XGBoostVolModel,
)
from volatilitylab.ml.validation import (
    ValidationConfig,
    WalkForwardValidator,
    create_regime_splits,
)


@dataclass
class RunConfig:
    """Configuration for experiment run."""

    symbol: str
    start_date: str = "2015-01-01"
    end_date: str | None = None
    horizons: list[int] = None
    seed: int = 42
    n_bootstrap: int = 1000
    confidence_level: float = 0.95

    # Validation config
    initial_train_years: float = 2.0
    test_months: int = 1
    final_test_years: float = 1.0

    # XGBoost params
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.1

    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [5, 20]


def run_experiment(
    config: RunConfig,
    experiments_dir: str = "./experiments",
    include_vix: bool = True,
) -> ExperimentResult:
    """
    Run a complete volatility forecasting experiment.

    Parameters
    ----------
    config : RunConfig
        Experiment configuration
    experiments_dir : str
        Directory for experiment artifacts
    include_vix : bool
        Whether to include VIX features

    Returns
    -------
    ExperimentResult
        Complete experiment results
    """
    tracker = ExperimentTracker(experiments_dir)

    # Load data
    print(f"Loading data for {config.symbol}...")
    try:
        prices = load_price_data(config.symbol, config.start_date, config.end_date)
        vix = load_vix_data(config.start_date, config.end_date) if include_vix else None
    except Exception as e:
        # Return error result
        exp_id = tracker.start_experiment(
            config.symbol,
            config.__dict__,
            config.seed,
        )
        return tracker.finish_experiment(exp_id, error=str(e))

    # Compute features
    print("Computing features...")
    engine = FeatureEngine(horizons=config.horizons, include_vix=include_vix)
    features = engine.compute_features(prices, vix)

    # Start experiment
    exp_id = tracker.start_experiment(
        config.symbol,
        config.__dict__,
        config.seed,
        features.features,
    )

    try:
        _run_experiment_inner(config, features, tracker, exp_id)
        return tracker.finish_experiment(exp_id)
    except Exception as e:
        return tracker.finish_experiment(exp_id, error=str(e))


def _run_experiment_inner(
    config: RunConfig,
    features: VolatilityFeatures,
    tracker: ExperimentTracker,
    exp_id: str,
) -> None:
    """Inner experiment logic."""

    X = features.features
    y_5d = features.target_5d
    y_20d = features.target_20d

    # Validation config
    val_config = ValidationConfig(
        initial_train_size=int(252 * config.initial_train_years),
        test_size=21 * config.test_months,
        step_size=21,
        purge_gap=5,
        embargo_gap=20,
        final_test_size=int(252 * config.final_test_years),
    )

    validator = WalkForwardValidator(val_config)

    # Check for leakage
    leakage_5d = validator.validate_no_leakage(X, y_5d, horizon=5)
    leakage_20d = validator.validate_no_leakage(X, y_20d, horizon=20)

    if not leakage_5d["passed"] or not leakage_20d["passed"]:
        raise ValueError(f"Leakage detected: 5d={leakage_5d}, 20d={leakage_20d}")

    # Get final test split
    train_idx_5d, test_idx_5d = validator.get_final_test_split(X, y_5d, horizon=5)
    train_idx_20d, test_idx_20d = validator.get_final_test_split(X, y_20d, horizon=20)

    # Store predictions
    all_predictions = {}

    # Run each model
    for horizon, y, train_idx, test_idx in [
        (5, y_5d, train_idx_5d, test_idx_5d),
        (20, y_20d, train_idx_20d, test_idx_20d),
    ]:
        print(f"\n=== Horizon: {horizon}d ===")

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        _returns = np.log(X_train["ret_lag_1"].shift(-1).dropna() + 1)  # Approximate

        horizon_preds = {}

        # 1. Historical Volatility
        print("  Training HV...")
        hv_model = HistoricalVolModel(window=21)
        hv_preds = _predict_baseline(hv_model, X_train, X_test, y_train)
        horizon_preds["hv"] = hv_preds

        metrics = compute_metrics(y_test.values, hv_preds)
        ci = block_bootstrap_ci(y_test.values, hv_preds, "qlike", config.n_bootstrap)

        tracker.log_model_result(
            exp_id,
            ModelResult(
                model_name=f"hv_{horizon}d",
                model_type="historical_volatility",
                params=hv_model.get_params(),
                metrics_5d=_metrics_to_dict(metrics) if horizon == 5 else {},
                metrics_20d=_metrics_to_dict(metrics) if horizon == 20 else {},
            ),
        )
        tracker.log_bootstrap_ci(exp_id, "hv", horizon, "qlike", _ci_to_dict(ci))

        # 2. EWMA
        print("  Training EWMA...")
        ewma_model = EWMAModel(decay=0.94)
        ewma_preds = _predict_baseline(ewma_model, X_train, X_test, y_train)
        horizon_preds["ewma"] = ewma_preds

        metrics = compute_metrics(y_test.values, ewma_preds)
        ci = block_bootstrap_ci(y_test.values, ewma_preds, "qlike", config.n_bootstrap)

        tracker.log_model_result(
            exp_id,
            ModelResult(
                model_name=f"ewma_{horizon}d",
                model_type="ewma",
                params=ewma_model.get_params(),
                metrics_5d=_metrics_to_dict(metrics) if horizon == 5 else {},
                metrics_20d=_metrics_to_dict(metrics) if horizon == 20 else {},
            ),
        )
        tracker.log_bootstrap_ci(exp_id, "ewma", horizon, "qlike", _ci_to_dict(ci))

        # 3. GARCH
        print("  Training GARCH...")
        try:
            garch_model = GARCHModel(p=1, q=1)
            # Use returns for GARCH
            returns_train = X_train["ret_lag_1"].values
            garch_model.fit(pd.Series(returns_train))
            garch_preds = np.full(
                len(X_test), garch_model.predict(pd.Series(returns_train), horizon)
            )
            horizon_preds["garch"] = garch_preds

            metrics = compute_metrics(y_test.values, garch_preds)
            ci = block_bootstrap_ci(y_test.values, garch_preds, "qlike", config.n_bootstrap)

            tracker.log_model_result(
                exp_id,
                ModelResult(
                    model_name=f"garch_{horizon}d",
                    model_type="garch",
                    params=garch_model.get_params(),
                    metrics_5d=_metrics_to_dict(metrics) if horizon == 5 else {},
                    metrics_20d=_metrics_to_dict(metrics) if horizon == 20 else {},
                ),
            )
            tracker.log_bootstrap_ci(exp_id, "garch", horizon, "qlike", _ci_to_dict(ci))
        except Exception as e:
            print(f"    GARCH failed: {e}")
            horizon_preds["garch"] = hv_preds  # Fallback

        # 4. XGBoost
        print("  Training XGBoost...")
        xgb_model = XGBoostVolModel(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            random_state=config.seed,
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        horizon_preds["xgboost"] = xgb_preds

        metrics = compute_metrics(y_test.values, xgb_preds)
        ci = block_bootstrap_ci(y_test.values, xgb_preds, "qlike", config.n_bootstrap)

        tracker.log_model_result(
            exp_id,
            ModelResult(
                model_name=f"xgboost_{horizon}d",
                model_type="xgboost",
                params=xgb_model.get_params(),
                metrics_5d=_metrics_to_dict(metrics) if horizon == 5 else {},
                metrics_20d=_metrics_to_dict(metrics) if horizon == 20 else {},
                feature_importance=xgb_model.get_params().get("feature_importance"),
            ),
        )
        tracker.log_bootstrap_ci(exp_id, "xgboost", horizon, "qlike", _ci_to_dict(ci))

        # DM Tests
        print("  Running DM tests...")
        dm_results = compare_models_dm(
            y_test.values,
            horizon_preds,
            baseline_model="hv",
            loss_fn="qlike",
            horizon=horizon,
        )

        for _, row in dm_results.iterrows():
            tracker.log_dm_comparison(exp_id, horizon, row.to_dict())

        all_predictions[horizon] = (y_test, horizon_preds)

    # Hypothesis testing
    print("\n=== Hypothesis Testing ===")

    # H1: Does ML beat baselines?
    for horizon in config.horizons:
        y_test, preds = all_predictions[horizon]

        # XGBoost vs HV
        dm_xgb_hv = diebold_mariano_test(
            y_test.values, preds["xgboost"], preds["hv"], loss_fn="qlike", horizon=horizon
        )

        # XGBoost vs EWMA
        dm_xgb_ewma = diebold_mariano_test(
            y_test.values, preds["xgboost"], preds["ewma"], loss_fn="qlike", horizon=horizon
        )

        xgb_beats_hv = dm_xgb_hv.model1_better and dm_xgb_hv.p_value < 0.05
        xgb_beats_ewma = dm_xgb_ewma.model1_better and dm_xgb_ewma.p_value < 0.05

        h1_result = "SUPPORTED" if (xgb_beats_hv or xgb_beats_ewma) else "NOT SUPPORTED"

        tracker.log_hypothesis_test(
            exp_id,
            f"H1_{horizon}d: ML beats HV/EWMA baselines",
            h1_result,
            {
                "xgb_vs_hv": {
                    "dm_stat": dm_xgb_hv.dm_statistic,
                    "p_value": dm_xgb_hv.p_value,
                    "xgb_better": dm_xgb_hv.model1_better,
                },
                "xgb_vs_ewma": {
                    "dm_stat": dm_xgb_ewma.dm_statistic,
                    "p_value": dm_xgb_ewma.p_value,
                    "xgb_better": dm_xgb_ewma.model1_better,
                },
            },
        )
        print(f"  H1 ({horizon}d): {h1_result}")

    # H3: Performance differs across regimes
    print("\n  Testing H3 (regime-dependent performance)...")

    low_vol_idx, high_vol_idx = create_regime_splits(X)

    for horizon in config.horizons:
        y_test, preds = all_predictions[horizon]

        # Get test indices in each regime
        test_dates = y_test.index
        low_vol_test = test_dates.isin(low_vol_idx)
        high_vol_test = test_dates.isin(high_vol_idx)

        if low_vol_test.sum() > 10 and high_vol_test.sum() > 10:
            # Metrics in each regime
            metrics_low = compute_metrics(
                y_test.values[low_vol_test], preds["xgboost"][low_vol_test]
            )
            metrics_high = compute_metrics(
                y_test.values[high_vol_test], preds["xgboost"][high_vol_test]
            )

            # Compare
            qlike_diff = abs(metrics_low.qlike - metrics_high.qlike)
            differs = qlike_diff > 0.1 * min(metrics_low.qlike, metrics_high.qlike)

            h3_result = "SUPPORTED" if differs else "NOT SUPPORTED"

            tracker.log_hypothesis_test(
                exp_id,
                f"H3_{horizon}d: Performance differs across regimes",
                h3_result,
                {
                    "low_vol_qlike": metrics_low.qlike,
                    "high_vol_qlike": metrics_high.qlike,
                    "qlike_diff": qlike_diff,
                    "n_low_vol": int(low_vol_test.sum()),
                    "n_high_vol": int(high_vol_test.sum()),
                },
            )
            print(f"  H3 ({horizon}d): {h3_result}")

            tracker.log_regime_analysis(
                exp_id,
                {
                    f"{horizon}d": {
                        "low_vol": _metrics_to_dict(metrics_low),
                        "high_vol": _metrics_to_dict(metrics_high),
                    }
                },
            )

    # Save predictions
    pred_df = pd.DataFrame(
        {
            "actual_5d": all_predictions[5][0],
            "actual_20d": all_predictions[20][0],
            **{f"{m}_5d": all_predictions[5][1][m] for m in all_predictions[5][1]},
            **{f"{m}_20d": all_predictions[20][1][m] for m in all_predictions[20][1]},
        }
    )
    tracker.save_predictions(exp_id, pred_df)


def _predict_baseline(model, X_train, X_test, y_train):
    """Generate predictions for baseline models."""
    # Use last training RV as prediction
    if hasattr(model, "fit") and not isinstance(model, (HistoricalVolModel, EWMAModel)):
        model.fit(X_train, y_train)
        return model.predict(X_test)

    # For HV/EWMA, use rolling approach
    # Simplified: use last training variance as constant prediction
    returns = X_train["ret_lag_1"].values
    model.fit(pd.Series(returns))
    pred_value = model.predict(pd.Series(returns), horizon=5)
    return np.full(len(X_test), pred_value)


def _metrics_to_dict(metrics):
    """Convert MetricResult to dict."""
    return {
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "qlike": metrics.qlike,
        "n_samples": metrics.n_samples,
    }


def _ci_to_dict(ci):
    """Convert BootstrapCI to dict."""
    return {
        "estimate": ci.estimate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
        "std_error": ci.std_error,
        "confidence_level": ci.confidence_level,
    }
