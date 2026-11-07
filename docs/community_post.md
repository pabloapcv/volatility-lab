Hey everyone — I put together VolatilityLab for the data-science cohort: a small research app you can clone and click through, not a slide deck of formulas.

Repo: https://github.com/pabloapcv/volatility-lab

It has three screens. The pricing workbench compares Black-Scholes, a CRR binomial tree, and Monte Carlo (with Greeks in the same per-1% units). The surface lab plots a real yfinance options chain from 7 Nov 2026 for SPY/QQQ/NVDA. The ML studio comes preloaded with a walk-forward realized-vol forecast (HV, EWMA, GARCH, HAR-RV, XGBoost) plus bootstrap CIs and Diebold-Mariano tests.

The honest finding: after I fixed a bug in HAR-RV (it was feeding the same 5-day variance in twice), XGBoost no longer beats the baseline. H1 is not supported. GARCH wins on SPY 20-day and both NVDA horizons. One shuffle test (NVDA 20d) even failed, which is the kind of result I would rather show than hide.

`make demo` then open http://127.0.0.1:5175. If something looks off, or you disagree with the stats, I would like the feedback.
