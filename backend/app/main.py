"""
VolatilityLab API Server

FastAPI application providing endpoints for option pricing, volatility analysis,
and market data management.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.app.api.data import router as data_router
from backend.app.api.pricing import router as pricing_router
from backend.app.api.volatility import router as volatility_router
from backend.app.api.experiments import router as experiments_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    data_dir = Path("./data/snapshots")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"VolatilityLab API starting. Data directory: {data_dir.absolute()}")
    yield
    # Shutdown
    print("VolatilityLab API shutting down.")


app = FastAPI(
    title="VolatilityLab API",
    description="""
Quantitative Options Pricing, Volatility Forecasting & Risk Analytics API.

## Features

- **Pricing Models**: Black-Scholes-Merton, CRR Binomial, Monte Carlo
- **Greeks**: Analytical and numerical sensitivities
- **Implied Volatility**: Brent's method solver with arbitrage validation
- **Volatility Surface**: Smile, skew, term structure, and 3D visualization
- **Market Data**: Options chain loading with quality filters and snapshots

## Models

| Model | European | American | Greeks |
|-------|----------|----------|--------|
| Black-Scholes | ✓ | - | Analytical |
| Binomial | ✓ | ✓ | Numerical |
| Monte Carlo | ✓ | - | - |
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "*",  # Allow all in development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pricing_router, prefix="/api/v1")
app.include_router(volatility_router, prefix="/api/v1")
app.include_router(data_router, prefix="/api/v1")
app.include_router(experiments_router, prefix="/api/v1")


@app.get("/")
async def root():
    """API root - health check and info."""
    return {
        "name": "VolatilityLab API",
        "version": "0.1.0",
        "status": "healthy",
        "docs": "/docs",
        "endpoints": {
            "pricing": "/api/v1/pricing",
            "volatility": "/api/v1/volatility",
            "data": "/api/v1/data",
            "experiments": "/api/v1/experiments",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__,
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("API_PORT", 8000))
    host = os.environ.get("API_HOST", "0.0.0.0")
    
    uvicorn.run(
        "backend.app.main:app",
        host=host,
        port=port,
        reload=True,
    )
