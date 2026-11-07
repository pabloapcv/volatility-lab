"""
Numerical accuracy tests for pricing models.

Tests against known reference values and mathematical properties.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from volatilitylab.pricing.black_scholes import (
    OptionType,
    black_scholes_price,
    black_scholes_greeks,
)
from volatilitylab.pricing.binomial import (
    BinomialModel,
    ExerciseStyle,
    binomial_price,
)
from volatilitylab.pricing.monte_carlo import (
    MonteCarloModel,
    VarianceReduction,
)


class TestReferenceValues:
    """Test against known reference values from textbooks."""
    
    def test_hull_example_european_call(self):
        """Test against Hull's Options, Futures textbook example."""
        # S=42, K=40, r=10%, T=0.5, sigma=20%
        price = black_scholes_price(
            spot=42, strike=40, time_to_expiry=0.5,
            risk_free_rate=0.10, volatility=0.20,
            option_type=OptionType.CALL
        )
        # Expected: approximately 4.76
        assert abs(price - 4.76) < 0.01
    
    def test_hull_example_european_put(self):
        """Test put price using put-call parity."""
        S, K, r, T, sigma = 42, 40, 0.10, 0.5, 0.20
        
        call = black_scholes_price(S, K, T, r, sigma, OptionType.CALL)
        put = black_scholes_price(S, K, T, r, sigma, OptionType.PUT)
        
        # Put-call parity: C - P = S - K*e^(-rT)
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 1e-6
    
    def test_wilmott_greeks(self):
        """Test Greeks against Wilmott reference values."""
        # S=100, K=100, r=5%, T=1, sigma=25%
        greeks = black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.25,
            option_type=OptionType.CALL
        )
        
        # Approximate expected values
        assert 0.5 < greeks.delta < 0.7  # ATM call delta ~0.5-0.6
        assert greeks.gamma > 0
        assert greeks.vega > 0
        assert greeks.theta < 0
        assert greeks.rho > 0


class TestMonotonicity:
    """Test monotonicity properties of option prices."""
    
    def test_call_decreasing_in_strike(self):
        """Test call price decreases as strike increases."""
        strikes = np.array([90, 95, 100, 105, 110])
        prices = black_scholes_price(
            spot=100, strike=strikes, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        # Each price should be less than the previous
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))
    
    def test_put_increasing_in_strike(self):
        """Test put price increases as strike increases."""
        strikes = np.array([90, 95, 100, 105, 110])
        prices = black_scholes_price(
            spot=100, strike=strikes, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT
        )
        
        # Each price should be greater than the previous
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))
    
    def test_call_increasing_in_spot(self):
        """Test call price increases as spot increases."""
        spots = np.array([90, 95, 100, 105, 110])
        prices = black_scholes_price(
            spot=spots, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))
    
    def test_option_increasing_in_volatility(self):
        """Test option price increases with volatility."""
        vols = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        prices = black_scholes_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=vols,
            option_type=OptionType.CALL
        )
        
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))
    
    def test_option_increasing_in_time(self):
        """Test option price increases with time to expiry (usually)."""
        times = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
        prices = black_scholes_price(
            spot=100, strike=100, time_to_expiry=times,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        # For ATM options, price generally increases with time
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))


class TestNoArbitrageBounds:
    """Test no-arbitrage bounds are satisfied."""
    
    @given(
        spot=st.floats(min_value=50, max_value=150),
        strike=st.floats(min_value=50, max_value=150),
        tte=st.floats(min_value=0.05, max_value=2.0),
        vol=st.floats(min_value=0.05, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_call_upper_bound(self, spot, strike, tte, vol):
        """Test call price <= spot."""
        assume(spot > 0 and strike > 0)
        
        price = black_scholes_price(
            spot=spot, strike=strike, time_to_expiry=tte,
            risk_free_rate=0.05, volatility=vol,
            option_type=OptionType.CALL
        )
        
        assert price <= spot + 1e-6
    
    @given(
        spot=st.floats(min_value=50, max_value=150),
        strike=st.floats(min_value=50, max_value=150),
        tte=st.floats(min_value=0.05, max_value=2.0),
        vol=st.floats(min_value=0.05, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_put_upper_bound(self, spot, strike, tte, vol):
        """Test put price <= strike * e^(-rT)."""
        assume(spot > 0 and strike > 0)
        
        price = black_scholes_price(
            spot=spot, strike=strike, time_to_expiry=tte,
            risk_free_rate=0.05, volatility=vol,
            option_type=OptionType.PUT
        )
        
        upper = strike * np.exp(-0.05 * tte)
        assert price <= upper + 1e-6
    
    @given(
        spot=st.floats(min_value=80, max_value=120),
        strike=st.floats(min_value=80, max_value=120),
    )
    @settings(max_examples=50)
    def test_call_lower_bound(self, spot, strike):
        """Test call price >= max(0, S - K*e^(-rT))."""
        assume(spot > 0 and strike > 0)
        
        price = black_scholes_price(
            spot=spot, strike=strike, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.25,
            option_type=OptionType.CALL
        )
        
        lower = max(0, spot - strike * np.exp(-0.05 * 0.5))
        assert price >= lower - 1e-6


class TestBinomialConvergence:
    """Test binomial convergence to Black-Scholes."""
    
    def test_european_call_convergence(self):
        """Test European call converges to BS."""
        bs_price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.25,
            option_type=OptionType.CALL
        )
        
        errors = []
        for steps in [100, 200, 400, 800]:
            bin_price = binomial_price(
                spot=100, strike=100, time_to_expiry=0.5,
                risk_free_rate=0.05, volatility=0.25,
                option_type=OptionType.CALL, steps=steps,
                exercise_style=ExerciseStyle.EUROPEAN
            )
            errors.append(abs(bin_price - bs_price))
        
        # Error should decrease with steps
        assert errors[3] < errors[0]
        # Final error should be small
        assert errors[3] < 0.01
    
    def test_convergence_rate_o_1_n(self):
        """Test convergence rate is O(1/N)."""
        bs_price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        steps = np.array([100, 200, 400])
        errors = []
        
        for n in steps:
            bin_price = binomial_price(
                100, 100, 0.5, 0.05, 0.20, OptionType.CALL,
                steps=int(n), exercise_style=ExerciseStyle.EUROPEAN
            )
            errors.append(abs(bin_price - bs_price))
        
        errors = np.array(errors)
        
        # For O(1/N) convergence, error * N should be roughly constant
        error_times_n = errors * steps
        # Check variance is not too high
        assert np.std(error_times_n) / np.mean(error_times_n) < 0.5


class TestMonteCarloCoverage:
    """Test Monte Carlo confidence interval coverage."""
    
    @pytest.mark.slow
    def test_95_ci_coverage(self):
        """Test 95% CI covers true value ~95% of time."""
        bs_price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        model = MonteCarloModel(n_paths=10000)
        covered = 0
        n_experiments = 100
        
        for seed in range(n_experiments):
            result = model.price(
                spot=100, strike=100, time_to_expiry=0.5,
                risk_free_rate=0.05, volatility=0.20,
                option_type=OptionType.CALL,
                seed=seed, confidence_level=0.95,
                compute_convergence=False
            )
            
            if result.confidence_interval.lower <= bs_price <= result.confidence_interval.upper:
                covered += 1
        
        coverage = covered / n_experiments
        # Should be close to 95%, allow for sampling variation
        assert 0.85 <= coverage <= 1.0


class TestGreekAccuracy:
    """Test Greek calculation accuracy."""
    
    def test_delta_finite_difference(self):
        """Test delta matches finite difference approximation."""
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        h = 0.01
        
        price_up = black_scholes_price(S + h, K, T, r, sigma, OptionType.CALL)
        price_down = black_scholes_price(S - h, K, T, r, sigma, OptionType.CALL)
        numerical_delta = (price_up - price_down) / (2 * h)
        
        analytical_delta = black_scholes_greeks(S, K, T, r, sigma, OptionType.CALL).delta
        
        assert abs(numerical_delta - analytical_delta) < 0.001
    
    def test_gamma_finite_difference(self):
        """Test gamma matches finite difference approximation."""
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        h = 0.01
        
        price_up = black_scholes_price(S + h, K, T, r, sigma, OptionType.CALL)
        price_mid = black_scholes_price(S, K, T, r, sigma, OptionType.CALL)
        price_down = black_scholes_price(S - h, K, T, r, sigma, OptionType.CALL)
        numerical_gamma = (price_up - 2*price_mid + price_down) / (h**2)
        
        analytical_gamma = black_scholes_greeks(S, K, T, r, sigma, OptionType.CALL).gamma
        
        assert abs(numerical_gamma - analytical_gamma) < 0.01
    
    def test_vega_finite_difference(self):
        """Test vega matches finite difference approximation."""
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        h = 0.001
        
        price_up = black_scholes_price(S, K, T, r, sigma + h, OptionType.CALL)
        price_down = black_scholes_price(S, K, T, r, sigma - h, OptionType.CALL)
        numerical_vega = (price_up - price_down) / (2 * h) * 0.01  # Per 1% move
        
        analytical_vega = black_scholes_greeks(S, K, T, r, sigma, OptionType.CALL).vega
        
        assert abs(numerical_vega - analytical_vega) < 0.01
