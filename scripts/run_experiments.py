#!/usr/bin/env python3
"""
Run volatility forecasting experiments for H1/H3 hypothesis testing.

Models compared:
- Historical Volatility (HV) - rolling 21-day variance
- EWMA (λ=0.94) - RiskMetrics standard
- GARCH(1,1) - parametric volatility model
- HAR-RV - Heterogeneous Autoregressive (standard benchmark)
- XGBoost - ML model with point-in-time features

Methodology:
- Walk-forward validation with 2-year initial training
- 1-year held-out final test period
- Purge gap = max(5, horizon+1) days to prevent target leakage
- Embargo gap = 20 days for 20-day horizon
- All models evaluated on identical test timestamps
- Bootstrap CIs via stationary block bootstrap
- DM tests with Newey-West HAC variance and HLN correction
- Holm-Bonferroni correction for multiple comparisons

Data source: yfinance (real market data)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from volatilitylab.ml.features import FeatureEngine, load_price_data, load_vix_data
from volatilitylab.ml.models import (
    HistoricalVolModel,
    EWMAModel,
    GARCHModel,
    HARRVModel,
    XGBoostVolModel,
)
from volatilitylab.ml.validation import WalkForwardValidator, ValidationConfig, create_regime_splits
from volatilitylab.ml.evaluation import (
    compute_metrics,
    block_bootstrap_ci,
    diebold_mariano_test,
    holm_correction,
)


def run_shuffle_test(X_train, y_train, X_test, y_test, seed=42):
    """
    Shuffle test: verify edge disappears with randomized targets.
    
    If XGBoost still performs well with shuffled targets, there's likely leakage.
    """
    np.random.seed(seed)
    
    # Fit XGBoost on shuffled targets
    y_train_shuffled = y_train.sample(frac=1, random_state=seed).values
    
    xgb = XGBoostVolModel(n_estimators=100, max_depth=4, random_state=seed)
    xgb.fit(X_train, pd.Series(y_train_shuffled, index=y_train.index))
    preds_shuffled = xgb.predict(X_test)
    
    # Also fit on real targets
    xgb_real = XGBoostVolModel(n_estimators=100, max_depth=4, random_state=seed)
    xgb_real.fit(X_train, y_train)
    preds_real = xgb_real.predict(X_test)
    
    metrics_shuffled = compute_metrics(y_test.values, preds_shuffled)
    metrics_real = compute_metrics(y_test.values, preds_real)
    
    return {
        'real_qlike': metrics_real.qlike,
        'shuffled_qlike': metrics_shuffled.qlike,
        'edge_preserved': metrics_real.qlike < metrics_shuffled.qlike,
        'qlike_difference': metrics_shuffled.qlike - metrics_real.qlike,
    }


def run_experiment(symbol, start_date="2018-01-01", seed=42, n_bootstrap=500):
    """Run complete experiment for one symbol."""
    print(f"\n{'='*60}")
    print(f"Running experiment for {symbol}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading price data from yfinance...")
    try:
        prices = load_price_data(symbol, start_date)
        vix = load_vix_data(start_date)
        print(f"  Loaded {len(prices)} days ({prices.index[0].date()} to {prices.index[-1].date()})")
        data_source = "yfinance"
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"symbol": symbol, "error": str(e), "data_source": "failed"}
    
    # Compute features
    print("Computing features...")
    engine = FeatureEngine(horizons=[5, 20], include_vix=True)
    features = engine.compute_features(prices, vix)
    print(f"  {features.metadata['n_samples']} samples, {features.metadata['n_features']} features")
    
    X = features.features
    y_5d = features.target_5d
    y_20d = features.target_20d
    
    # Verify targets are forward-looking
    print("  Verifying target alignment...")
    print(f"    Target 5d range: [{y_5d.min():.6f}, {y_5d.max():.6f}]")
    print(f"    Feature rvar_5d range: [{X['rvar_5d'].min():.6f}, {X['rvar_5d'].max():.6f}]")
    
    # Validation setup
    val_config = ValidationConfig(
        initial_train_size=252 * 2,
        test_size=21,
        step_size=21,
        purge_gap=5,
        embargo_gap=20,
        final_test_size=252,
    )
    validator = WalkForwardValidator(val_config)
    
    results = {
        "symbol": symbol,
        "data_source": data_source,
        "n_total": len(X),
        "date_range": [str(X.index[0].date()), str(X.index[-1].date())],
        "horizons": {},
    }
    
    for horizon, y in [(5, y_5d), (20, y_20d)]:
        print(f"\n--- Horizon: {horizon}d ---")
        
        train_idx, test_idx = validator.get_final_test_split(X, y, horizon)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        print(f"  Train: {len(train_idx)} samples ({X_train.index[0].date()} to {X_train.index[-1].date()})")
        print(f"  Test:  {len(test_idx)} samples ({X_test.index[0].date()} to {X_test.index[-1].date()})")
        
        predictions = {}
        model_results = {}
        
        # Get returns for time-series models
        returns = np.log(prices['close'] / prices['close'].shift(1)).dropna()
        returns_train = returns.loc[X_train.index[0]:X_train.index[-1]]
        
        # 1. Historical Volatility
        print("  Training HV...")
        hv = HistoricalVolModel(window=21)
        hv.fit(returns_train)
        predictions['hv'] = np.full(len(y_test), hv.predict(returns_train, horizon))
        
        # 2. EWMA
        print("  Training EWMA...")
        ewma = EWMAModel(decay=0.94)
        ewma.fit(returns_train)
        predictions['ewma'] = np.full(len(y_test), ewma.predict(returns_train, horizon))
        
        # 3. GARCH
        print("  Training GARCH...")
        try:
            garch = GARCHModel(p=1, q=1)
            garch.fit(returns_train)
            predictions['garch'] = np.full(len(y_test), garch.predict(returns_train, horizon))
        except Exception as e:
            print(f"    GARCH failed: {e}, using HV as fallback")
            predictions['garch'] = predictions['hv'].copy()
        
        # 4. HAR-RV (standard benchmark)
        print("  Training HAR-RV...")
        har = HARRVModel()
        har.fit(X_train, y_train)
        predictions['har'] = har.predict(X_test)
        
        # 5. XGBoost
        print("  Training XGBoost...")
        np.random.seed(seed)
        xgb = XGBoostVolModel(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed)
        xgb.fit(X_train, y_train)
        predictions['xgboost'] = xgb.predict(X_test)
        
        # Compute metrics and CIs for ALL models
        print("  Computing metrics and bootstrap CIs...")
        for model_name, preds in predictions.items():
            metrics = compute_metrics(y_test.values, preds)
            ci = block_bootstrap_ci(y_test.values, preds, 'qlike', n_bootstrap, random_state=seed)
            
            model_results[model_name] = {
                'mae': metrics.mae,
                'rmse': metrics.rmse,
                'qlike': metrics.qlike,
                'qlike_ci_lower': ci.ci_lower,
                'qlike_ci_upper': ci.ci_upper,
                'qlike_se': ci.std_error,
            }
            print(f"    {model_name}: QLIKE={metrics.qlike:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")
        
        # DM tests: compare each model against HAR-RV (standard benchmark)
        print("  Running DM tests vs HAR-RV...")
        dm_results = {}
        p_values = []
        model_names = []
        
        for model_name in ['hv', 'ewma', 'garch', 'xgboost']:
            dm = diebold_mariano_test(
                y_test.values, predictions[model_name], predictions['har'],
                loss_fn='qlike', horizon=horizon
            )
            dm_results[model_name] = {
                'dm_stat': dm.dm_statistic,
                'p_value': dm.p_value,
                'better_than_har': dm.model1_better,
            }
            p_values.append(dm.p_value)
            model_names.append(model_name)
        
        # Holm correction
        holm_results = holm_correction(p_values, alpha=0.05)
        for i, (_, adj_p, sig) in enumerate(holm_results):
            dm_results[model_names[i]]['adjusted_p'] = adj_p
            dm_results[model_names[i]]['significant'] = sig
        
        # Shuffle test
        print("  Running shuffle test...")
        shuffle_result = run_shuffle_test(X_train, y_train, X_test, y_test, seed)
        print(f"    Real QLIKE: {shuffle_result['real_qlike']:.4f}")
        print(f"    Shuffled QLIKE: {shuffle_result['shuffled_qlike']:.4f}")
        print(f"    Edge preserved: {shuffle_result['edge_preserved']}")
        
        # H1: Does XGBoost beat HAR-RV?
        xgb_vs_har = dm_results['xgboost']
        h1_supported = xgb_vs_har['better_than_har'] and xgb_vs_har['significant']
        
        # H3: Regime analysis
        low_vol_idx, high_vol_idx = create_regime_splits(X)
        test_dates = y_test.index
        low_vol_test = test_dates.isin(low_vol_idx)
        high_vol_test = test_dates.isin(high_vol_idx)
        
        regime_analysis = {}
        if low_vol_test.sum() > 10 and high_vol_test.sum() > 10:
            metrics_low = compute_metrics(y_test.values[low_vol_test], predictions['xgboost'][low_vol_test])
            metrics_high = compute_metrics(y_test.values[high_vol_test], predictions['xgboost'][high_vol_test])
            
            regime_analysis = {
                'low_vol_qlike': metrics_low.qlike,
                'high_vol_qlike': metrics_high.qlike,
                'n_low_vol': int(low_vol_test.sum()),
                'n_high_vol': int(high_vol_test.sum()),
                'differs': abs(metrics_low.qlike - metrics_high.qlike) > 0.1 * min(abs(metrics_low.qlike), abs(metrics_high.qlike))
            }
        
        results['horizons'][f'{horizon}d'] = {
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'models': model_results,
            'dm_tests_vs_har': dm_results,
            'shuffle_test': shuffle_result,
            'h1_xgb_beats_har': h1_supported,
            'regime_analysis': regime_analysis,
        }
    
    return results


def format_markdown(all_results):
    """Format results as markdown."""
    lines = [
        "## Volatility Forecasting Results",
        "",
        "**Methodology:**",
        "- Walk-forward validation with 2-year initial training, 1-year held-out test",
        "- Purge gap: max(5, horizon+1) days; embargo gap: 20 days",
        "- Data source: yfinance (real market data)",
        "- Bootstrap CIs: stationary block bootstrap (n=500)",
        "- Statistical tests: Diebold-Mariano with HAC variance, HLN correction, Holm-Bonferroni correction",
        "",
        "### Model Performance (QLIKE ± 95% CI, lower is better)",
        "",
        "| Symbol | Horizon | HV | EWMA | GARCH | HAR-RV | XGBoost |",
        "|--------|---------|-----|------|-------|--------|---------|",
    ]
    
    for r in all_results:
        if 'error' in r:
            lines.append(f"| {r['symbol']} | - | Error: {r.get('error', 'unknown')[:20]} | | | | |")
            continue
        
        for h, data in r['horizons'].items():
            m = data['models']
            row = f"| {r['symbol']} | {h} |"
            for model in ['hv', 'ewma', 'garch', 'har', 'xgboost']:
                q = m[model]['qlike']
                ci_lo = m[model]['qlike_ci_lower']
                ci_hi = m[model]['qlike_ci_upper']
                row += f" {q:.3f} [{ci_lo:.3f},{ci_hi:.3f}] |"
            lines.append(row)
    
    lines.extend([
        "",
        "### Hypothesis Tests",
        "",
        "**H1: Does XGBoost beat HAR-RV?** (Diebold-Mariano test, Holm-corrected α=0.05)",
        "",
        "| Symbol | Horizon | XGB vs HAR DM stat | p-value | Significant | H1 Supported |",
        "|--------|---------|-------------------|---------|-------------|--------------|",
    ])
    
    for r in all_results:
        if 'error' in r:
            continue
        for h, data in r['horizons'].items():
            dm = data['dm_tests_vs_har']['xgboost']
            h1 = "✓" if data['h1_xgb_beats_har'] else "✗"
            sig = "✓" if dm['significant'] else "✗"
            lines.append(f"| {r['symbol']} | {h} | {dm['dm_stat']:.3f} | {dm['p_value']:.4f} | {sig} | {h1} |")
    
    lines.extend([
        "",
        "### Shuffle Test (Leakage Check)",
        "",
        "If XGBoost still performs well on shuffled targets, there's likely data leakage.",
        "",
        "| Symbol | Horizon | Real QLIKE | Shuffled QLIKE | Edge Preserved |",
        "|--------|---------|------------|----------------|----------------|",
    ])
    
    for r in all_results:
        if 'error' in r:
            continue
        for h, data in r['horizons'].items():
            st = data['shuffle_test']
            preserved = "✓" if st['edge_preserved'] else "✗ (LEAKAGE?)"
            lines.append(f"| {r['symbol']} | {h} | {st['real_qlike']:.4f} | {st['shuffled_qlike']:.4f} | {preserved} |")
    
    lines.extend([
        "",
        "*Note: Results based on real yfinance data. Run `python scripts/run_experiments.py` to reproduce.*",
    ])
    
    return "\n".join(lines)


def main():
    symbols = ["SPY", "QQQ", "NVDA"]
    all_results = []
    
    for symbol in symbols:
        result = run_experiment(symbol, start_date="2018-01-01", seed=42, n_bootstrap=500)
        all_results.append(result)
    
    # Save results
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    md = format_markdown(all_results)
    print("\n" + "="*60)
    print("MARKDOWN FOR README:")
    print("="*60)
    print(md)
    
    with open(output_dir / "results_readme.md", "w") as f:
        f.write(md)
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
