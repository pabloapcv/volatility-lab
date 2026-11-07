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

---

## Research Questions

This platform enables investigation of questions like:

1. **Model Convergence**: How quickly does the binomial model converge to Black-Scholes? What's the practical step count needed for pricing accuracy?

2. **Monte Carlo Efficiency**: How do antithetic and control variates affect variance? What path count achieves 1bp precision?

3. **Implied Volatility Surface**: What drives the volatility smile/skew structure? How does it evolve across expirations?

4. **American Option Premium**: When is early exercise valuable? How does dividend yield affect the American-European spread?

*Future research (data collection in progress): ML volatility forecasting and delta-hedging backtest studies.*

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
│   └── data/             # Chain loading, snapshots
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
└── scripts/
```

---

## Limitations & Future Work

### Current Limitations

- **European Monte Carlo only**: American options require LSM or similar
- **No stochastic vol**: Heston model is out of scope for MVP
- **Static vol surface**: No dynamics or forecasting yet
- **Synthetic sample data**: Real data requires API access

### Roadmap (Data Collection in Progress)

Historical option chain collection is running daily. Planned additions:

1. **Heston Model**: Stochastic volatility calibration and pricing
2. **ML Volatility Forecasting**: GARCH, XGBoost, LSTM comparison
3. **Delta-Hedging Backtest**: Empirical hedge performance analysis
4. **Risk Analytics**: VaR/ES with scenario analysis

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
