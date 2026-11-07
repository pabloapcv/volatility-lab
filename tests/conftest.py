"""
Pytest configuration and fixtures.
"""

import sys
from pathlib import Path

import pytest

# Add src and backend to path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))


@pytest.fixture
def sample_option_params():
    """Standard option parameters for testing."""
    return {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 0.5,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "dividend_yield": 0.0,
    }


@pytest.fixture
def itm_call_params():
    """ITM call option parameters."""
    return {
        "spot": 110.0,
        "strike": 100.0,
        "time_to_expiry": 0.5,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "dividend_yield": 0.0,
    }


@pytest.fixture
def otm_call_params():
    """OTM call option parameters."""
    return {
        "spot": 90.0,
        "strike": 100.0,
        "time_to_expiry": 0.5,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "dividend_yield": 0.0,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "numerical: marks tests that verify numerical accuracy"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
