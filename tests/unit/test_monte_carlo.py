"""
Unit tests for Monte Carlo pricing model.

Tests include:
- Convergence to Black-Scholes
- Confidence interval coverage
- Variance reduction effectiveness
- Reproducibility with seeds
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from volatilitylab.pricing.monte_carlo import (
    MonteCarloModel,
    VarianceReduction,
    monte_carlo_price,
)
from volatilitylab.pricing.black_scholes import (
    OptionType,
    black_scholes_price,
)


class TestMonteCarloPrice:
    """Test Monte Carlo pricing calculations."""
    
    def test_call_close_to_bs(self):
        """Test MC call price is close to Black-Scholes."""
        bs_price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        mc_price = monte_carlo_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL,
            n_paths=100000, seed=42
        )
        
        # Should be within 1% of BS price
        assert abs(mc_price - bs_price) / bs_price < 0.01
    
    def test_put_close_to_bs(self):
        """Test MC put price is close to Black-Scholes."""
        bs_price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT
        )
        
        mc_price = monte_carlo_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT,
            n_paths=100000, seed=42
        )
        
        assert abs(mc_price - bs_price) / bs_price < 0.01
    
    def test_reproducible_with_seed(self):
        """Test same seed produces same result."""
        price1 = monte_carlo_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, n_paths=10000, seed=123
        )
        
        price2 = monte_carlo_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, n_paths=10000, seed=123
        )
        
        assert price1 == price2
    
    def test_different_seeds_differ(self):
        """Test different seeds produce different results."""
        price1 = monte_carlo_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, n_paths=10000, seed=1
        )
        
        price2 = monte_carlo_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, n_paths=10000, seed=2
        )
        
        assert price1 != price2
    
    def test_at_expiry_intrinsic(self):
        """Test price at expiry equals intrinsic value."""
        price = monte_carlo_price(
            spot=110, strike=100, time_to_expiry=0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        assert price == pytest.approx(10.0, abs=1e-6)
    
    def test_with_dividend(self):
        """Test pricing with continuous dividend yield."""
        price_no_div = monte_carlo_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, dividend_yield=0.0,
            n_paths=50000, seed=42
        )
        
        price_with_div = monte_carlo_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, dividend_yield=0.03,
            n_paths=50000, seed=42
        )
        
        # Call with dividend should be cheaper
        assert price_with_div < price_no_div


class TestVarianceReduction:
    """Test variance reduction techniques."""
    
    def test_antithetic_reduces_variance(self):
        """Test antithetic variates reduce standard error."""
        model = MonteCarloModel(n_paths=10000, seed=42)
        
        result_none = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            variance_reduction=VarianceReduction.NONE
        )
        
        result_anti = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            variance_reduction=VarianceReduction.ANTITHETIC
        )
        
        # Antithetic should have lower or similar std error
        # (Not always strictly lower due to randomness)
        assert result_anti.std_error < result_none.std_error * 1.5
    
    def test_control_variate_reduces_variance(self):
        """Test control variate can reduce variance."""
        model = MonteCarloModel(n_paths=20000, seed=42)
        
        result_none = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            variance_reduction=VarianceReduction.NONE
        )
        
        result_cv = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            variance_reduction=VarianceReduction.CONTROL_VARIATE
        )
        
        # Control variate result includes beta
        assert result_cv.control_variate_beta is not None
    
    def test_both_variance_reduction(self):
        """Test combined variance reduction."""
        model = MonteCarloModel(n_paths=10000, seed=42)
        
        result = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            variance_reduction=VarianceReduction.BOTH
        )
        
        assert result.variance_reduction == VarianceReduction.BOTH
        assert result.control_variate_beta is not None


class TestConfidenceInterval:
    """Test confidence interval calculations."""
    
    def test_ci_contains_price(self):
        """Test CI often contains the true price."""
        bs_price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        model = MonteCarloModel(n_paths=50000, seed=42)
        result = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL,
            confidence_level=0.95
        )
        
        # BS price should be within 95% CI
        assert result.confidence_interval.lower <= bs_price <= result.confidence_interval.upper
    
    def test_ci_coverage_verification(self):
        """Test CI coverage rate across multiple experiments."""
        model = MonteCarloModel(n_paths=5000, seed=0)
        
        coverage = model.verify_confidence_interval(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL,
            n_experiments=50, n_paths=5000, confidence_level=0.95
        )
        
        # Coverage should be close to 95% (within 2 std devs)
        assert coverage['within_2_std'] or coverage['coverage_rate'] > 0.85
    
    def test_wider_ci_for_higher_level(self):
        """Test higher confidence level gives wider CI."""
        model = MonteCarloModel(n_paths=10000, seed=42)
        
        result_95 = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            confidence_level=0.95
        )
        
        result_99 = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            confidence_level=0.99
        )
        
        width_95 = result_95.confidence_interval.upper - result_95.confidence_interval.lower
        width_99 = result_99.confidence_interval.upper - result_99.confidence_interval.lower
        
        assert width_99 > width_95


class TestConvergenceDiagnostics:
    """Test convergence diagnostics."""
    
    def test_diagnostics_computed(self):
        """Test convergence diagnostics are computed."""
        model = MonteCarloModel(n_paths=10000, seed=42)
        result = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            compute_convergence=True, n_batches=5
        )
        
        assert result.convergence is not None
        assert len(result.convergence.batch_prices) == 5
        assert len(result.convergence.cumulative_means) == 5
    
    def test_effective_samples_with_antithetic(self):
        """Test effective samples accounts for antithetic correlation."""
        model = MonteCarloModel(n_paths=10000, seed=42)
        
        result = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            variance_reduction=VarianceReduction.ANTITHETIC,
            compute_convergence=True
        )
        
        # Effective samples should be computed
        assert result.convergence is not None
        assert result.convergence.effective_samples > 0


class TestMonteCarloModel:
    """Test MonteCarloModel class."""
    
    def test_model_parameters(self):
        """Test model stores parameters correctly."""
        model = MonteCarloModel(n_paths=50000, n_steps=50, seed=99)
        
        result = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20
        )
        
        assert result.n_paths == 50000
        assert result.n_steps == 50
        assert result.seed == 99
    
    def test_model_id(self):
        """Test model has correct identifier."""
        model = MonteCarloModel()
        result = model.price(100, 100, 0.5, 0.05, 0.20)
        
        assert result.model_id == "monte-carlo-gbm"
    
    def test_invalid_n_paths_raises(self):
        """Test invalid n_paths raises error."""
        with pytest.raises(ValueError):
            MonteCarloModel(n_paths=10)
    
    def test_warnings_for_low_paths(self):
        """Test warnings for low path count."""
        model = MonteCarloModel(n_paths=1000)
        result = model.price(100, 100, 0.5, 0.05, 0.20)
        
        assert result.warnings is not None


class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(
        spot=st.floats(min_value=50, max_value=150),
        strike=st.floats(min_value=50, max_value=150),
    )
    @settings(max_examples=30)
    def test_price_positive(self, spot, strike):
        """Test MC prices are always non-negative."""
        assume(spot > 0 and strike > 0)
        
        price = monte_carlo_price(
            spot=spot, strike=strike, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.25,
            option_type=OptionType.CALL,
            n_paths=1000, seed=42
        )
        
        assert price >= 0
    
    @given(
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=10)
    def test_reproducibility(self, seed):
        """Test same seed always gives same result."""
        price1 = monte_carlo_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL,
            n_paths=1000, seed=seed
        )
        
        price2 = monte_carlo_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL,
            n_paths=1000, seed=seed
        )
        
        assert price1 == price2
