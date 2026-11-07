# VolatilityLab

**Quantitative Options Pricing, Volatility Forecasting & Risk Analytics**

An open-source research platform combining numerical derivatives pricing, stochastic simulation, machine learning, and empirical hedging analysis.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green)
![React](https://img.shields.io/badge/React-18.2-61DAFB)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243)
![Tests](https://img.shields.io/badge/Tests-85%25+-success)

---

## Quick Demo

```bash
# Clone and set up
git clone https://github.com/your-repo/volatilitylab.git
cd volatilitylab

# Install Python dependencies
pip install -e ".[dev]"

# Generate sample data for offline use
python scripts/create_sample_snapshot.py

# Run tests
pytest tests/ -v

# Start the API server
uvicorn backend.app.main:app --reload

# In another terminal, start frontend
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 to explore:
- **Options Pricing Workbench**: Compare Black-Scholes, Binomial, and Monte Carlo prices
- **Volatility Surface Lab**: 3D volatility surfaces, smiles, and term structure
- **ML Research Studio**: Run and analyze volatility forecasting experiments

---

## Research Questions

This platform enables investigation of questions like:

1. **Model Convergence**: How quickly does the binomial model converge to Black-Scholes? What's the practical step count needed for pricing accuracy?

2. **Monte Carlo Efficiency**: How do antithetic and control variates affect variance? What path count achieves 1bp precision?

3. **Implied Volatility Surface**: What drives the volatility smile/skew structure? How does it evolve across expirations?

4. **American Option Premium**: When is early exercise valuable? How does dividend yield affect the American-European spread?

---

## Volatility Forecasting Results (FR-07)

**Methodology:**
- Walk-forward validation: 2-year initial training, 1-year held-out test
- Purge gap: max(5, horizon+1) days; embargo: 20 days
- Data source: yfinance (real market data, 2018-2026)
- Bootstrap CIs: stationary block bootstrap (n=500)
- Statistical tests: Diebold-Mariano with HAC variance, HLN correction, Holm-Bonferroni

### Model Performance (QLIKE ± 95% CI, lower is better)

| Symbol | Horizon | HV | EWMA | GARCH | HAR-RV | XGBoost |
|--------|---------|-----|------|-------|--------|---------|
| SPY | 5d | -2.10 [-2.72,-1.21] | -2.57 [-3.05,-1.89] | -3.00 [-3.31,-2.57] | -2.38 [-2.45,-2.28] | **-3.13** [-3.34,-2.90] |
| SPY | 20d | -2.81 [-3.12,-2.44] | -2.93 [-3.20,-2.61] | **-3.11** [-3.26,-2.92] | -2.54 [-2.60,-2.47] | -3.05 [-3.22,-2.87] |
| QQQ | 5d | 0.83 [-0.37,2.36] | -0.70 [-1.51,0.32] | -1.59 [-2.13,-0.91] | -1.95 [-2.07,-1.81] | **-2.28** [-2.50,-2.04] |
| QQQ | 20d | -1.38 [-1.87,-0.79] | -1.51 [-1.97,-0.97] | -2.10 [-2.36,-1.78] | -2.02 [-2.12,-1.90] | **-2.19** [-2.38,-1.98] |
| NVDA | 5d | -0.93 [-1.13,-0.71] | -0.90 [-1.12,-0.66] | **-0.96** [-1.10,-0.82] | -0.82 [-0.93,-0.71] | -0.82 [-0.98,-0.66] |
| NVDA | 20d | -0.41 [-0.67,-0.15] | -0.63 [-0.85,-0.41] | **-0.94** [-1.03,-0.86] | -0.82 [-0.88,-0.76] | -0.80 [-0.89,-0.71] |

### Hypothesis Test Results

**H1: Does XGBoost beat HAR-RV baseline?** (Diebold-Mariano, Holm-corrected α=0.05)

| Symbol | Horizon | DM stat | p-value | Significant |
|--------|---------|---------|---------|-------------|
| SPY | 5d | -9.70 | <0.0001 | ✓ |
| SPY | 20d | -6.38 | <0.0001 | ✓ |
| QQQ | 5d | -5.70 | <0.0001 | ✓ |
| QQQ | 20d | -2.23 | 0.026 | ✗ (not sig. after Holm) |
| NVDA | 5d | 0.17 | 0.87 | ✗ |
| NVDA | 20d | 0.54 | 0.59 | ✗ |

**H1 Summary:** XGBoost significantly beats HAR-RV for SPY (both horizons) and QQQ 5d.
No significant improvement for QQQ 20d or NVDA. **Partially supported.**

**Leakage Check (Shuffle Test):** Edge disappears on shuffled targets (5/6 cases),
confirming results are not due to data leakage.

*Models: HV (21d), EWMA (λ=0.94), GARCH(1,1), HAR-RV, XGBoost (100 trees, depth 4).*
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

Interactive docs at http://localhost:8000/docs

---

## Running Locally

### Option 1: Python + npm (Development)

```bash
# Backend
pip install -e ".[dev]"
python scripts/create_sample_snapshot.py
uvicorn backend.app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Option 2: Docker Compose

```bash
docker compose up dev
```

Opens backend on :8000 and frontend on :5173.

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
- **Synthetic sample data**: Real data requires API access or yfinance

### Roadmap

1. **Heston Model**: Stochastic volatility calibration and pricing
2. **Delta-Hedging Backtest**: Empirical hedge performance analysis
3. **Risk Analytics**: VaR/ES with scenario analysis
4. **LSTM Volatility**: Deep learning volatility forecasting comparison

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
