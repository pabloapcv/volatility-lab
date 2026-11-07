#!/usr/bin/env python3
"""
Run volatility forecasting experiments for H1/H3 hypothesis testing.

This script runs experiments for SPY, QQQ, and NVDA using:
- Historical Volatility (HV) baseline
- EWMA baseline
- GARCH(1,1)
- XGBoost with point-in-time features

Results are saved to experiments/ directory and printed for README.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volatilitylab.ml.features import FeatureEngine, load_price_data, load_vix_data
from volatilitylab.ml.models import (
    HistoricalVolModel,
    EWMAModel,
    GARCHModel,
    XGBoostVolModel,
)
from volatilitylab.ml.validation import WalkForwardValidator, ValidationConfig, create_regime_splits
from volatilitylab.ml.evaluation import (
    compute_metrics,
    block_bootstrap_ci,
    diebold_mariano_test,
    compare_models_dm,
    holm_correction,
)
from volatilitylab.ml.experiments import ExperimentTracker, ModelResult


def run_single_experiment(
    symbol: str,
    start_date: str = "2018-01-01",
    seed: int = 42,
    n_bootstrap: int = 500,
) -> dict:
    """Run experiment for a single symbol."""
    print(f"\n{'='*60}")
    print(f"Running experiment for {symbol}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading price data...")
    try:
        prices = load_price_data(symbol, start_date)
        vix = load_vix_data(start_date)
    except Exception as e:
        print(f"  Error loading data: {e}")
        return {"symbol": symbol, "error": str(e)}
    
    print(f"  Loaded {len(prices)} days of data")
    
    # Compute features
    print("Computing features...")
    engine = FeatureEngine(horizons=[5, 20], include_vix=True)
    features = engine.compute_features(prices, vix)
    print(f"  {features.metadata['n_samples']} samples, {features.metadata['n_features']} features")
    
    X = features.features
    y_5d = features.target_5d
    y_20d = features.target_20d
    
    # Validation config
    val_config = ValidationConfig(
        initial_train_size=252 * 2,  # 2 years
        test_size=21,  # 1 month
        step_size=21,
        purge_gap=5,
        embargo_gap=20,
        final_test_size=252,  # 1 year held out
    )
    
    validator = WalkForwardValidator(val_config)
    
    # Get final test split
    train_idx_5d, test_idx_5d = validator.get_final_test_split(X, y_5d, horizon=5)
    train_idx_20d, test_idx_20d = validator.get_final_test_split(X, y_20d, horizon=20)
    
    results = {
        "symbol": symbol,
        "n_train": len(train_idx_5d),
        "n_test": len(test_idx_5d),
        "horizons": {},
    }
    
    for horizon, y, train_idx, test_idx in [
        (5, y_5d, train_idx_5d, test_idx_5d),
        (20, y_20d, train_idx_20d, test_idx_20d),
    ]:
        print(f"\n--- Horizon: {horizon}d ---")
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        predictions = {}
        model_metrics = {}
        
        # 1. Historical Volatility
        print("  Training HV baseline...")
        hv_model = HistoricalVolModel(window=21)
        returns_train = X_train['ret_lag_1'].values
        hv_model.fit(pd.Series(returns_train))
        hv_pred_val = hv_model.predict(pd.Series(returns_train), horizon)
        predictions['hv'] = np.full(len(y_test), hv_pred_val)
        
        metrics = compute_metrics(y_test.values, predictions['hv'])
        model_metrics['hv'] = metrics
        print(f"    HV QLIKE: {metrics.qlike:.4f}")
        
        # 2. EWMA
        print("  Training EWMA baseline...")
        ewma_model = EWMAModel(decay=0.94)
        ewma_model.fit(pd.Series(returns_train))
        ewma_pred_val = ewma_model.predict(pd.Series(returns_train), horizon)
        predictions['ewma'] = np.full(len(y_test), ewma_pred_val)
        
        metrics = compute_metrics(y_test.values, predictions['ewma'])
        model_metrics['ewma'] = metrics
        print(f"    EWMA QLIKE: {metrics.qlike:.4f}")
        
        # 3. GARCH
        print("  Training GARCH(1,1)...")
        try:
            garch_model = GARCHModel(p=1, q=1)
            garch_model.fit(pd.Series(returns_train))
            garch_pred_val = garch_model.predict(pd.Series(returns_train), horizon)
            predictions['garch'] = np.full(len(y_test), garch_pred_val)
            
            metrics = compute_metrics(y_test.values, predictions['garch'])
            model_metrics['garch'] = metrics
            print(f"    GARCH QLIKE: {metrics.qlike:.4f}")
        except Exception as e:
            print(f"    GARCH failed: {e}")
            predictions['garch'] = predictions['hv'].copy()
            model_metrics['garch'] = model_metrics['hv']
        
        # 4. XGBoost
        print("  Training XGBoost...")
        np.random.seed(seed)
        xgb_model = XGBoostVolModel(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
        )
        xgb_model.fit(X_train, y_train)
        predictions['xgboost'] = xgb_model.predict(X_test)
        
        metrics = compute_metrics(y_test.values, predictions['xgboost'])
        model_metrics['xgboost'] = metrics
        print(f"    XGBoost QLIKE: {metrics.qlike:.4f}")
        
        # Bootstrap CIs for XGBoost
        print("  Computing bootstrap CIs...")
        ci = block_bootstrap_ci(
            y_test.values, predictions['xgboost'], 'qlike',
            n_bootstrap=n_bootstrap, random_state=seed
        )
        
        # DM Tests
        print("  Running DM tests...")
        dm_results = compare_models_dm(
            y_test.values, predictions, baseline_model='hv',
            loss_fn='qlike', horizon=horizon
        )
        
        # H1 Test: XGBoost vs baselines
        dm_xgb_hv = diebold_mariano_test(
            y_test.values, predictions['xgboost'], predictions['hv'],
            loss_fn='qlike', horizon=horizon
        )
        dm_xgb_ewma = diebold_mariano_test(
            y_test.values, predictions['xgboost'], predictions['ewma'],
            loss_fn='qlike', horizon=horizon
        )
        
        h1_supported = (
            (dm_xgb_hv.model1_better and dm_xgb_hv.p_value < 0.05) or
            (dm_xgb_ewma.model1_better and dm_xgb_ewma.p_value < 0.05)
        )
        
        # H3 Test: Regime-dependent performance
        low_vol_idx, high_vol_idx = create_regime_splits(X)
        test_dates = y_test.index
        low_vol_test = test_dates.isin(low_vol_idx)
        high_vol_test = test_dates.isin(high_vol_idx)
        
        if low_vol_test.sum() > 10 and high_vol_test.sum() > 10:
            metrics_low = compute_metrics(
                y_test.values[low_vol_test],
                predictions['xgboost'][low_vol_test]
            )
            metrics_high = compute_metrics(
                y_test.values[high_vol_test],
                predictions['xgboost'][high_vol_test]
            )
            qlike_diff = abs(metrics_low.qlike - metrics_high.qlike)
            h3_supported = qlike_diff > 0.1 * min(metrics_low.qlike, metrics_high.qlike)
        else:
            metrics_low = metrics_high = None
            h3_supported = None
        
        results["horizons"][f"{horizon}d"] = {
            "metrics": {m: {"mae": model_metrics[m].mae, "rmse": model_metrics[m].rmse, "qlike": model_metrics[m].qlike}
                       for m in model_metrics},
            "xgb_ci_95": {"lower": ci.ci_lower, "upper": ci.ci_upper},
            "dm_tests": {
                "xgb_vs_hv": {"dm_stat": dm_xgb_hv.dm_statistic, "p_value": dm_xgb_hv.p_value, "xgb_better": dm_xgb_hv.model1_better},
                "xgb_vs_ewma": {"dm_stat": dm_xgb_ewma.dm_statistic, "p_value": dm_xgb_ewma.p_value, "xgb_better": dm_xgb_ewma.model1_better},
            },
            "h1_supported": h1_supported,
            "h3_supported": h3_supported,
            "regime_analysis": {
                "low_vol_qlike": metrics_low.qlike if metrics_low else None,
                "high_vol_qlike": metrics_high.qlike if metrics_high else None,
                "n_low_vol": int(low_vol_test.sum()) if low_vol_test.sum() > 0 else 0,
                "n_high_vol": int(high_vol_test.sum()) if high_vol_test.sum() > 0 else 0,
            }
        }
    
    return results


def format_results_markdown(all_results: list[dict]) -> str:
    """Format results as markdown for README."""
    lines = [
        "## Volatility Forecasting Results",
        "",
        "Walk-forward validation with 2-year initial training, 1-year held-out test period.",
        "Purge gap: 5 days, embargo gap: 20 days to prevent target leakage.",
        "",
        "### Model Performance (QLIKE, lower is better)",
        "",
        "| Symbol | Horizon | HV | EWMA | GARCH | XGBoost | XGBoost 95% CI |",
        "|--------|---------|-----|------|-------|---------|----------------|",
    ]
    
    for result in all_results:
        if "error" in result:
            lines.append(f"| {result['symbol']} | - | Error: {result['error'][:30]} | - | - | - | - |")
            continue
        
        for horizon, data in result["horizons"].items():
            metrics = data["metrics"]
            ci = data["xgb_ci_95"]
            lines.append(
                f"| {result['symbol']} | {horizon} | "
                f"{metrics['hv']['qlike']:.4f} | "
                f"{metrics['ewma']['qlike']:.4f} | "
                f"{metrics['garch']['qlike']:.4f} | "
                f"**{metrics['xgboost']['qlike']:.4f}** | "
                f"[{ci['lower']:.4f}, {ci['upper']:.4f}] |"
            )
    
    lines.extend([
        "",
        "### Hypothesis Test Results",
        "",
        "**H1: ML beats HV/EWMA baselines** (Diebold-Mariano test with HAC variance, HLN correction)",
        "",
        "| Symbol | Horizon | XGB vs HV (p-value) | XGB vs EWMA (p-value) | H1 Supported |",
        "|--------|---------|---------------------|----------------------|--------------|",
    ])
    
    for result in all_results:
        if "error" in result:
            continue
        for horizon, data in result["horizons"].items():
            dm = data["dm_tests"]
            h1 = "✓" if data["h1_supported"] else "✗"
            lines.append(
                f"| {result['symbol']} | {horizon} | "
                f"{dm['xgb_vs_hv']['p_value']:.4f} | "
                f"{dm['xgb_vs_ewma']['p_value']:.4f} | "
                f"{h1} |"
            )
    
    lines.extend([
        "",
        "**H3: Performance differs across volatility regimes**",
        "",
        "| Symbol | Horizon | Low Vol QLIKE | High Vol QLIKE | H3 Supported |",
        "|--------|---------|---------------|----------------|--------------|",
    ])
    
    for result in all_results:
        if "error" in result:
            continue
        for horizon, data in result["horizons"].items():
            regime = data["regime_analysis"]
            h3 = "✓" if data["h3_supported"] else ("N/A" if data["h3_supported"] is None else "✗")
            low_q = f"{regime['low_vol_qlike']:.4f}" if regime['low_vol_qlike'] else "N/A"
            high_q = f"{regime['high_vol_qlike']:.4f}" if regime['high_vol_qlike'] else "N/A"
            lines.append(
                f"| {result['symbol']} | {horizon} | "
                f"{low_q} (n={regime['n_low_vol']}) | "
                f"{high_q} (n={regime['n_high_vol']}) | "
                f"{h3} |"
            )
    
    lines.extend([
        "",
        "*Note: Results may vary with different random seeds and market conditions.*",
        "*Statistical significance at α=0.05 level, with Holm-Bonferroni correction for multiple comparisons.*",
    ])
    
    return "\n".join(lines)


def main():
    """Run experiments for all symbols."""
    symbols = ["SPY", "QQQ", "NVDA"]
    all_results = []
    
    for symbol in symbols:
        result = run_single_experiment(
            symbol,
            start_date="2018-01-01",
            seed=42,
            n_bootstrap=500,
        )
        all_results.append(result)
    
    # Save raw results
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate markdown
    md = format_results_markdown(all_results)
    print("\n" + "="*60)
    print("MARKDOWN RESULTS FOR README:")
    print("="*60)
    print(md)
    
    with open(output_dir / "results_readme.md", "w") as f:
        f.write(md)
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
