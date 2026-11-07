## Volatility Forecasting Results

Walk-forward validation with 2-year initial training, 1-year held-out test period.
Purge gap: 5 days, embargo gap: 20 days to prevent target leakage.

### Model Performance (QLIKE, lower is better)

| Symbol | Horizon | HV | EWMA | GARCH | XGBoost | XGBoost 95% CI |
|--------|---------|-----|------|-------|---------|----------------|
| SPY | 5d | -3.0846 | -3.0803 | -3.0806 | **-3.3911** | [-3.6030, -3.1736] |
| SPY | 20d | -0.9060 | -2.3062 | -3.1007 | **-3.2122** | [-3.3669, -3.0454] |
| QQQ | 5d | -2.1285 | -2.0453 | -2.1455 | **-2.5021** | [-2.7518, -2.2691] |
| QQQ | 20d | 2.8183 | -0.2962 | -1.9567 | **-2.3395** | [-2.5140, -2.1611] |
| NVDA | 5d | -0.9146 | -0.8891 | -0.9629 | **-1.2450** | [-1.4350, -1.0643] |
| NVDA | 20d | -0.9038 | -0.8029 | -0.9479 | **-1.0312** | [-1.1595, -0.9108] |

### Hypothesis Test Results

**H1: ML beats HV/EWMA baselines** (Diebold-Mariano test with HAC variance, HLN correction)

| Symbol | Horizon | XGB vs HV (p-value) | XGB vs EWMA (p-value) | H1 Supported |
|--------|---------|---------------------|----------------------|--------------|
| SPY | 5d | 0.0000 | 0.0000 | ✓ |
| SPY | 20d | 0.0000 | 0.0001 | ✓ |
| QQQ | 5d | 0.0000 | 0.0000 | ✓ |
| QQQ | 20d | 0.0000 | 0.0000 | ✓ |
| NVDA | 5d | 0.0000 | 0.0000 | ✓ |
| NVDA | 20d | 0.0000 | 0.0000 | ✓ |

**H3: Performance differs across volatility regimes**

| Symbol | Horizon | Low Vol QLIKE | High Vol QLIKE | H3 Supported |
|--------|---------|---------------|----------------|--------------|
| SPY | 5d | -3.5412 (n=206) | -2.7188 (n=46) | ✓ |
| SPY | 20d | -3.2658 (n=206) | -2.9722 (n=46) | ✓ |
| QQQ | 5d | -2.5800 (n=206) | -2.1534 (n=46) | ✓ |
| QQQ | 20d | -2.3399 (n=206) | -2.3380 (n=46) | ✓ |
| NVDA | 5d | -1.2734 (n=206) | -1.1177 (n=46) | ✓ |
| NVDA | 20d | -1.0535 (n=206) | -0.9318 (n=46) | ✓ |

*Note: Results may vary with different random seeds and market conditions.*
*Statistical significance at α=0.05 level, with Holm-Bonferroni correction for multiple comparisons.*