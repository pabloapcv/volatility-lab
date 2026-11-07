"""
Integration tests for FastAPI endpoints.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health and info endpoints."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "VolatilityLab API"
        assert "version" in data
        assert data["status"] == "healthy"

    def test_health(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_cors_localhost_origin(self, client):
        """Local Vite origins remain allowed."""
        response = client.get("/health", headers={"Origin": "http://localhost:5175"})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5175"

    def test_cors_onrender_origin(self, client):
        """Render static-site origins match allow_origin_regex."""
        origin = "https://volatilitylab-ui.onrender.com"
        response = client.get("/health", headers={"Origin": origin})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin


class TestPricingEndpoints:
    """Test pricing API endpoints."""

    def test_black_scholes(self, client):
        """Test Black-Scholes pricing endpoint."""
        response = client.post(
            "/api/v1/pricing/black-scholes",
            json={
                "spot": 100,
                "strike": 100,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "volatility": 0.20,
                "option_type": "call",
                "dividend_yield": 0.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "price" in data
        assert "greeks" in data
        assert data["model_id"] == "black-scholes-merton"
        assert data["price"] > 0

    def test_black_scholes_put(self, client):
        """Test Black-Scholes put pricing."""
        response = client.post(
            "/api/v1/pricing/black-scholes",
            json={
                "spot": 100,
                "strike": 100,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "volatility": 0.20,
                "option_type": "put",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["greeks"]["delta"] < 0  # Put delta is negative

    def test_black_scholes_invalid_input(self, client):
        """Test Black-Scholes rejects invalid input."""
        response = client.post(
            "/api/v1/pricing/black-scholes",
            json={
                "spot": -100,  # Invalid
                "strike": 100,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "volatility": 0.20,
            },
        )

        assert response.status_code == 422  # Validation error

    def test_binomial(self, client):
        """Test binomial pricing endpoint."""
        response = client.post(
            "/api/v1/pricing/binomial",
            json={
                "spot": 100,
                "strike": 100,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "volatility": 0.20,
                "option_type": "call",
                "steps": 100,
                "exercise_style": "european",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "price" in data
        assert data["model_id"] == "crr-binomial"
        assert data["steps"] == 100

    def test_binomial_american(self, client):
        """Test binomial American option pricing."""
        response = client.post(
            "/api/v1/pricing/binomial",
            json={
                "spot": 100,
                "strike": 100,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "volatility": 0.20,
                "option_type": "put",
                "steps": 100,
                "exercise_style": "american",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["exercise_style"] == "american"

    def test_monte_carlo(self, client):
        """Test Monte Carlo pricing endpoint."""
        response = client.post(
            "/api/v1/pricing/monte-carlo",
            json={
                "spot": 100,
                "strike": 100,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "volatility": 0.20,
                "option_type": "call",
                "n_paths": 10000,
                "seed": 42,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "price" in data
        assert "std_error" in data
        assert "confidence_interval" in data
        assert data["model_id"] == "monte-carlo-gbm"

    def test_compare_models(self, client):
        """Test model comparison endpoint."""
        response = client.post(
            "/api/v1/pricing/compare",
            json={
                "spot": 100,
                "strike": 100,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "volatility": 0.20,
                "option_type": "call",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "black_scholes" in data
        assert "binomial_100" in data
        assert "binomial_500" in data
        assert "monte_carlo" in data
        assert "max_discrepancy" in data


class TestVolatilityEndpoints:
    """Test volatility API endpoints."""

    def test_implied_vol(self, client):
        """Test implied volatility endpoint."""
        response = client.post(
            "/api/v1/volatility/implied",
            json={
                "price": 10,
                "spot": 100,
                "strike": 100,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "option_type": "call",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "implied_volatility" in data
        assert "status" in data
        assert "bounds" in data

    def test_implied_vol_arbitrage(self, client):
        """Test implied vol rejects arbitrage-violating price."""
        response = client.post(
            "/api/v1/volatility/implied",
            json={
                "price": 5,  # Below intrinsic for deep ITM
                "spot": 120,
                "strike": 100,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "option_type": "call",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "arbitrage_violation"

    def test_bid_ask_iv(self, client):
        """Test bid/ask implied volatility endpoint."""
        response = client.post(
            "/api/v1/volatility/implied/bid-ask",
            json={
                "bid": 9.5,
                "ask": 10.5,
                "spot": 100,
                "strike": 100,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "option_type": "call",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "bid_iv" in data
        assert "ask_iv" in data
        assert "mid_iv" in data


class TestDataEndpoints:
    """Test data API endpoints."""

    def test_list_underlyings(self, client):
        """Test listing supported underlyings."""
        response = client.get("/api/v1/data/underlyings")

        assert response.status_code == 200
        data = response.json()
        assert "underlyings" in data
        symbols = [u["symbol"] for u in data["underlyings"]]
        assert "SPY" in symbols
        assert "QQQ" in symbols

    def test_list_snapshots(self, client):
        """Test listing snapshots."""
        response = client.get("/api/v1/data/snapshots")

        assert response.status_code == 200
        data = response.json()
        assert "snapshots" in data
        assert "total" in data

    def test_spy_surface_uses_real_snapshot_and_day_axis(self, client):
        """Default SPY surface should load the committed yfinance sample with day expiries."""
        response = client.get("/api/v1/volatility/surface/SPY")
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"].get("is_synthetic") is False
        assert data["metadata"].get("provider") == "yfinance"
        assert data["metadata"]["n_observed_points"] > 10
        expiries = [p["expiry"] for p in data["points"]]
        # Days, not year fractions (those would be ~0.02–0.2).
        assert max(expiries) > 5
        assert min(expiries) >= 1.0
        if data.get("grid"):
            ys = [row[0] for row in data["grid"]["expiry"]]
            assert max(ys) > 5


class TestExperimentsEndpoints:
    """Published FR-07 results must load without running an experiment."""

    def test_published_results(self, client):
        response = client.get("/api/v1/experiments/published")
        assert response.status_code == 200
        data = response.json()
        assert data["h1_verdict"] == "NOT SUPPORTED"
        assert len(data["rows"]) == 6
        assert len(data["dm_tests"]) == 6
        assert any(not s["edge_preserved"] for s in data["shuffle_tests"])


class TestGreeksUnits:
    """Black-Scholes and binomial Vega/Rho must share per-1% units."""

    def test_workbench_greeks_agree(self, client):
        payload = {
            "spot": 500,
            "strike": 510,
            "time_to_expiry": 0.25,
            "risk_free_rate": 0.045,
            "volatility": 0.22,
            "option_type": "call",
            "dividend_yield": 0.01,
            "steps": 200,
        }
        bs = client.post("/api/v1/pricing/black-scholes", json=payload).json()
        bn = client.post("/api/v1/pricing/binomial", json=payload).json()
        assert abs(bn["greeks"]["vega"] - bs["greeks"]["vega"]) / abs(bs["greeks"]["vega"]) < 0.05
        assert abs(bn["greeks"]["rho"] - bs["greeks"]["rho"]) / abs(bs["greeks"]["rho"]) < 0.05
        assert bn["greeks"]["vega"] > 0.5
