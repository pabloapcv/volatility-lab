# VolatilityLab

**Quantitative Options Pricing, Volatility Forecasting & Risk Analytics**

## Live demo

No install needed:

- **App:** [https://volatilitylab-hd1l.onrender.com](https://volatilitylab-hd1l.onrender.com)
- **API:** [https://volatilitylab-api.onrender.com](https://volatilitylab-api.onrender.com) ([OpenAPI docs](https://volatilitylab-api.onrender.com/docs))

The free Render host sleeps when idle. The first load after a pause can take **~30–60 seconds**; later clicks are fast (implied-vol surfaces are cached in memory on the API).

---

An open-source research platform combining numerical derivatives pricing, stochastic simulation, machine learning, and empirical hedging analysis. The UI is a light-themed three-tab app; each tab has a collapsible **How to use this page** guide (open on first visit).

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green)
![React](https://img.shields.io/badge/React-18.2-61DAFB)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243)
![Tests](https://img.shields.io/badge/Tests-85%25+-success)

---

## How to use it

Open the live app (or a local `make demo` session) and pick a tab. Expand **How to use this page** if it is collapsed.

1. **Options Pricing Workbench** — Set spot, strike, expiry in **years**, volatility, rate, dividend yield, and call/put. **Calculate All Models** prices with Black-Scholes, CRR binomial, and Monte Carlo. Vega and Rho are per **1%** change. Raise **Binomial Steps** to watch the tree converge to Black-Scholes; the Monte Carlo card shows a 95% CI.

2. **Volatility Surface Lab** — Pick SPY, QQQ, or NVDA. The default data is a committed **yfinance** chain observed **2026-11-07**. Switch 3D Surface / Smile / Term Structure. Drag to rotate the 3D plot, scroll to zoom. Axes are log moneyness \(\ln(K/F)\), days to expiry, and implied vol. Very short-dated quotes from a weekend snapshot can show inflated IV.

3. **ML Research Studio** — The preloaded **FR-07** table is the committed result (walk-forward QLIKE ± 95% CI). Lower QLIKE is better; bold blue is best on that row. **DM** is Diebold-Mariano of XGBoost vs HAR-RV; **Holm** is the multiple-testing correction. The shuffle test flags leakage (NVDA 20d).

**Honest H1 result:** *“XGBoost significantly beats HAR-RV”* is **NOT SUPPORTED** after Holm correction. GARCH(1,1) is best on several rows. See the tables below for the numbers.

---

## Screenshots

| Pricing Workbench | Volatility Surface | ML Research Studio |
|-------------------|--------------------|--------------------|
| ![Pricing Workbench](docs/screenshots/pricing_workbench.png) | ![Volatility Surface](docs/screenshots/volatility_surface_spy.png) | ![ML Research Studio](docs/screenshots/ml_research_studio.png) |

---

## Research Questions

This platform enables investigation of questions like:

1. **Model Convergence**: How quickly does the binomial model converge to Black-Scholes? What's the practical step count needed for pricing accuracy?

2. **Monte Carlo Efficiency**: How do antithetic and control variates affect variance? What path count achieves 1bp precision?

3. **Implied Volatility Surface**: What drives the volatility smile/skew structure? How does it evolve across expirations?

4. **American Option Premium**: When is early exercise valuable? How does dividend yield affect the American-European spread?

---

## Volatility Forecasting Results (FR-07)

**Data Source:**
- **Price data:** yfinance (SPY, QQQ, NVDA), 2018-01-02 to 2026-11-06
- **Test period:** 252 trading days (Oct 2025 – Oct 2026)
- **Training:** Expanding window, 1884 samples at test time

*Option-chain dashboard data:* compact **yfinance** snapshots committed under `data/snapshots/*_yfinance_sample_*` (SPY, QQQ, NVDA, observed 2026-11-07; nearby expiries and near-ATM strikes). Synthetic `*_sample_offline` chains remain as an offline fallback if a real file is missing.

*Forecasting results below:* real yfinance **daily prices** (not the option snapshots), 2018-01-02 to 2026-11-06, 252 test observations.

**Methodology:**
- Walk-forward validation: 2-year initial training, 1-year held-out final test
- Purge gap: max(5, horizon+1) days; embargo: 20 days
- Bootstrap CIs: stationary block bootstrap (n=500)
- Statistical tests: Diebold-Mariano with HAC variance, HLN correction, Holm-Bonferroni

### Model Performance (QLIKE ± 95% CI, lower is better)

| Symbol | Horizon | HV | EWMA | GARCH | HAR-RV | XGBoost |
|--------|---------|-----|------|-------|--------|---------|
| SPY | 5d | -2.10 [-2.72,-1.21] | -2.57 [-3.05,-1.89] | -3.00 [-3.31,-2.57] | -3.04 [-3.19,-2.84] | **-3.13** [-3.34,-2.90] |
| SPY | 20d | -2.81 [-3.12,-2.44] | -2.93 [-3.20,-2.61] | **-3.11** [-3.26,-2.92] | -2.93 [-3.02,-2.83] | -3.05 [-3.22,-2.87] |
| QQQ | 5d | 0.83 [-0.37,2.36] | -0.70 [-1.51,0.32] | -1.59 [-2.13,-0.91] | -2.24 [-2.44,-2.01] | **-2.28** [-2.50,-2.04] |
| QQQ | 20d | -1.38 [-1.87,-0.79] | -1.51 [-1.97,-0.97] | -2.10 [-2.36,-1.78] | -2.18 [-2.32,-2.02] | **-2.19** [-2.38,-1.97] |
| NVDA | 5d | -0.93 [-1.13,-0.71] | -0.90 [-1.12,-0.66] | **-0.96** [-1.10,-0.82] | -0.88 [-0.99,-0.76] | -0.82 [-0.98,-0.66] |
| NVDA | 20d | -0.41 [-0.67,-0.15] | -0.63 [-0.85,-0.41] | **-0.94** [-1.03,-0.86] | -0.85 [-0.92,-0.79] | -0.80 [-0.90,-0.71] |

### Hypothesis Test Results

**H1: Does XGBoost beat HAR-RV baseline?** (Diebold-Mariano, Holm-corrected α=0.05)

| Symbol | Horizon | DM stat | p-value | Significant |
|--------|---------|---------|---------|-------------|
| SPY | 5d | -2.16 | 0.031 | ✗ |
| SPY | 20d | -2.15 | 0.032 | ✗ |
| QQQ | 5d | -0.84 | 0.40 | ✗ |
| QQQ | 20d | -0.10 | 0.92 | ✗ |
| NVDA | 5d | 1.87 | 0.062 | ✗ |
| NVDA | 20d | 1.66 | 0.097 | ✗ |

**H1 Summary:** XGBoost does **not** significantly outperform HAR-RV after Holm correction. 
GARCH(1,1) is the best model for SPY 20d, NVDA 5d, and NVDA 20d. **H1 not supported.**

**Leakage Check (Shuffle Test):**

| Symbol | Horizon | Real QLIKE | Shuffled QLIKE | Edge Preserved |
|--------|---------|------------|----------------|----------------|
| SPY | 5d | -3.13 | -2.83 | ✓ |
| SPY | 20d | -3.05 | -2.84 | ✓ |
| QQQ | 5d | -2.28 | -2.12 | ✓ |
| QQQ | 20d | -2.19 | -2.14 | ✓ |
| NVDA | 5d | -0.82 | -0.79 | ✓ |
| NVDA | 20d | -0.80 | -0.82 | ✗ |

NVDA 20d fails the shuffle test (edge not preserved), indicating XGBoost's apparent performance on this case is likely noise. Across 10 random seeds, edge is preserved 7/10 times—the signal is weak and not robust.

*Models: HV (21d), EWMA (λ=0.94), GARCH(1,1), HAR-RV (daily/weekly/monthly RV), XGBoost (100 trees, depth 4).*
*Reproduce: `python scripts/run_experiments.py`*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React + TypeScript Frontend              │
│              Plotly · Tailwind CSS · TanStack Query         │
└─────────────────────┬───────────────────────────────────────┘
                      │ REST API
┌─────────────────────▼───────────────────────────────────────┐
│                   FastAPI Application Layer                 │
│                   Pydantic · Async · OpenAPI                │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│  Pricing  │  │ Volatility│  │   Data    │
│  Engine   │  │  Analysis │  │ Ingestion │
│───────────│  │───────────│  │───────────│
│ BS-Merton │  │ IV Solver │  │ yfinance  │
│ Binomial  │  │ Surface   │  │ Parquet   │
│ Monte     │  │ Smile     │  │ Snapshots │
│ Carlo     │  │ Term Str. │  │           │
└───────────┘  └───────────┘  └───────────┘
```

---

## Mathematical Models

### Black-Scholes-Merton (FR-02)

European option pricing with continuous dividend yield:

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

Features:
- Analytical Greeks: Delta, Gamma, Vega, Theta, Rho
- Vectorized NumPy implementation
- Edge case handling: expiry, near-zero vol, deep ITM/OTM
- Put-call parity validation

### CRR Binomial (FR-03)

Cox-Ross-Rubinstein recombining tree:

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

Features:
- European and American exercise
- Convergence analysis (O(1/N) to BS)
- Numerical Greeks via finite differences
- Early exercise node tracking

### Monte Carlo GBM (FR-04)

Risk-neutral simulation with variance reduction:

$$S_{t+\Delta t} = S_t \exp\left[(r - q - \sigma^2/2)\Delta t + \sigma\sqrt{\Delta t} Z\right]$$

Features:
- Configurable paths, steps, and seed
- Standard errors and confidence intervals
- Antithetic variates (30-50% variance reduction)
- Control variates with optimal beta
- Convergence diagnostics

### Implied Volatility (FR-05)

Brent's method root-finding with no-arbitrage validation:

Features:
- No-arbitrage bound rejection
- Bid/Ask/Mid IV calculation
- Round-trip accuracy < 10⁻⁶
- Smile, skew, and term structure views
- Quality flags (observed vs interpolated)

See [docs/mathematical_models.md](docs/mathematical_models.md) for full derivations and tolerances.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/pricing/black-scholes` | POST | BS price and Greeks |
| `/api/v1/pricing/binomial` | POST | Binomial tree pricing |
| `/api/v1/pricing/monte-carlo` | POST | Monte Carlo simulation |
| `/api/v1/pricing/compare` | POST | Compare all models |
| `/api/v1/volatility/implied` | POST | Solve for IV |
| `/api/v1/volatility/surface/{symbol}` | GET | Full IV surface |
| `/api/v1/volatility/smile/{symbol}/{days}` | GET | Smile at expiry |
| `/api/v1/data/chain` | POST | Load options chain |
| `/api/v1/data/snapshots` | GET | List saved snapshots |
| `/api/v1/experiments` | POST | Start ML experiment |
| `/api/v1/experiments` | GET | List experiments |
| `/api/v1/experiments/{id}` | GET | Get experiment results |

Interactive docs at http://127.0.0.1:8765/docs (or `/docs` on the hosted API).

---

## Running locally

Requires **Python 3.12+** and **Node 18+**.

### One command (recommended)

```bash
git clone https://github.com/pabloapcv/volatility-lab.git
cd volatility-lab
make demo
```

That installs Python + frontend deps and starts both servers.

API: http://127.0.0.1:8765 · UI: http://127.0.0.1:5175

### Docker

```bash
docker compose up --build
```

Same ports as `make demo`.

### Manual (two terminals)

```bash
pip install -e ".[dev,ml]"
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8765

# other terminal
cd frontend && npm ci && npm run dev
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/volatilitylab --cov-report=term-missing

# Property-based tests only
pytest tests/ -k "PropertyBased"

# Numerical accuracy tests
pytest tests/numerical/ -v
```

---

## Test Coverage

| Module | Coverage | Key Tests |
|--------|----------|-----------|
| Black-Scholes | 92% | Put-call parity, Greek accuracy, edge cases |
| Binomial | 88% | BS convergence, American premium, early exercise |
| Monte Carlo | 86% | CI coverage, variance reduction, reproducibility |
| Implied Vol | 91% | Round-trip, arbitrage bounds, bid-ask |

Property-based tests (Hypothesis) verify mathematical invariants across random valid parameters.

---

## Project Structure

```
volatilitylab/
├── src/volatilitylab/
│   ├── pricing/          # BS, Binomial, MC engines
│   ├── volatility/       # IV solver, surface
│   ├── data/             # Chain loading, snapshots
│   └── ml/               # Volatility forecasting
│       ├── features.py   # Point-in-time feature engineering
│       ├── models.py     # HV, EWMA, GARCH, XGBoost
│       ├── validation.py # Walk-forward with purge/embargo
│       ├── evaluation.py # Metrics, bootstrap, DM tests
│       └── experiments.py # Experiment tracking
├── backend/app/
│   ├── api/              # FastAPI routes
│   └── schemas/          # Pydantic models
├── frontend/src/
│   └── components/       # React + Plotly UI
├── tests/
│   ├── unit/             # Model unit tests
│   ├── numerical/        # Accuracy tests
│   └── integration/      # API tests
├── docs/
│   └── mathematical_models.md
├── data/snapshots/       # Saved market data
├── experiments/          # Saved experiment results
└── scripts/
    ├── create_sample_snapshot.py
    └── run_experiments.py
```

---

## Limitations & Future Work

### Current Limitations

- **European Monte Carlo only**: American options require LSM or similar
- **No stochastic vol**: Heston model is out of scope for MVP
- **ML forecasting gains are modest**: XGBoost does not significantly beat HAR-RV or GARCH in rigorous testing
- **Dashboard chains are a compact sample**: one yfinance day, nearby expiries / near-ATM strikes. Synthetic snapshots are the offline fallback only.

### Roadmap

1. **Heston Model**: Stochastic volatility calibration and pricing
2. **Delta-Hedging Backtest**: Empirical hedge performance analysis
3. **Risk Analytics**: VaR/ES with scenario analysis
4. **LSTM Volatility**: Deep learning volatility forecasting comparison

---

## What I learned

The first HAR-RV implementation reused `rvar_5d` as both the daily and weekly component, so the “standard benchmark” was misspecified. After the fix (daily \(r_t^2 \times 252\), weekly `rvar_5d`, monthly `rvar_21d`, fit on the train fold only), HAR-RV jumped from a weak extra to a competitive OLS baseline — and the headline “XGBoost beats HAR-RV” result disappeared under Holm-corrected Diebold-Mariano tests.

GARCH(1,1) holds up because a simple conditional-variance recursion is a strong default for equity index and single-name RV at these horizons. It wins SPY 20d and both NVDA horizons. Tree models with point-in-time features can look good in a table and still fail a shuffle check (NVDA 20d) once the baseline is honest.

---

## Contributing

Contributions welcome! Please:

1. Run `pytest` and ensure tests pass
2. Run `ruff check src/ tests/` for linting
3. Add tests for new functionality
4. Update documentation as needed

---

## License

MIT License - see [LICENSE](LICENSE)

---

## References

- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). *Option pricing: A simplified approach*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*
- Hull, J. C. (2022). *Options, Futures, and Other Derivatives* (11th ed.)
