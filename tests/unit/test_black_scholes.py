"""
Unit tests for Black-Scholes-Merton pricing model.

Tests include:
- Known reference values
- Put-call parity
- Edge cases (expiry, zero vol, deep ITM/OTM)
- Greek calculations
- Vectorized operations
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from volatilitylab.pricing.black_scholes import (
    BlackScholesModel,
    OptionType,
    black_scholes_price,
    black_scholes_greeks,
)


class TestBlackScholesPrice:
    """Test Black-Scholes price calculations."""
    
    def test_atm_call_reference(self):
        """Test ATM call against known reference value."""
        # Reference: S=100, K=100, T=1, r=5%, sigma=20%, q=0
        # Expected price ≈ 10.4506
        price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, dividend_yield=0.0
        )
        assert abs(price - 10.4506) < 0.01
    
    def test_atm_put_reference(self):
        """Test ATM put against known reference value."""
        # Same parameters as call
        price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT, dividend_yield=0.0
        )
        # Expected ≈ 5.5735
        assert abs(price - 5.5735) < 0.01
    
    def test_itm_call(self):
        """Test deep ITM call approaches intrinsic value."""
        price = black_scholes_price(
            spot=150, strike=100, time_to_expiry=0.001,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        intrinsic = 150 - 100
        assert abs(price - intrinsic) < 0.1
    
    def test_otm_call_low_value(self):
        """Test deep OTM call has low value."""
        price = black_scholes_price(
            spot=50, strike=100, time_to_expiry=0.1,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        assert price < 0.01
    
    def test_at_expiry_call(self):
        """Test call at expiry equals intrinsic value."""
        price = black_scholes_price(
            spot=110, strike=100, time_to_expiry=0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        assert price == pytest.approx(10.0, abs=1e-6)
    
    def test_at_expiry_put(self):
        """Test put at expiry equals intrinsic value."""
        price = black_scholes_price(
            spot=90, strike=100, time_to_expiry=0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT
        )
        assert price == pytest.approx(10.0, abs=1e-6)
    
    def test_zero_vol_itm_call(self):
        """Test zero vol ITM call equals discounted intrinsic."""
        price = black_scholes_price(
            spot=120, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.0001,
            option_type=OptionType.CALL
        )
        # Should be close to S - K*e^(-rT)
        expected = 120 - 100 * np.exp(-0.05)
        assert abs(price - expected) < 0.1
    
    def test_vectorized_calculation(self):
        """Test vectorized calculation with multiple strikes."""
        strikes = np.array([90, 100, 110, 120])
        prices = black_scholes_price(
            spot=100, strike=strikes, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        assert prices.shape == (4,)
        # ITM should have higher price
        assert prices[0] > prices[1] > prices[2] > prices[3]
    
    def test_with_dividend(self):
        """Test pricing with continuous dividend yield."""
        price_no_div = black_scholes_price(
            spot=100, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, dividend_yield=0.0
        )
        price_with_div = black_scholes_price(
            spot=100, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, dividend_yield=0.02
        )
        # Call should be cheaper with dividend
        assert price_with_div < price_no_div


class TestPutCallParity:
    """Test put-call parity: C - P = S*e^(-qT) - K*e^(-rT)."""
    
    def test_parity_atm(self):
        """Test put-call parity for ATM options."""
        S, K, T, r, q, sigma = 100, 100, 0.5, 0.05, 0.01, 0.25
        
        call = black_scholes_price(S, K, T, r, sigma, OptionType.CALL, q)
        put = black_scholes_price(S, K, T, r, sigma, OptionType.PUT, q)
        
        lhs = call - put
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        
        assert abs(lhs - rhs) < 1e-8
    
    def test_parity_itm(self):
        """Test put-call parity for ITM call."""
        S, K, T, r, q, sigma = 120, 100, 0.25, 0.04, 0.0, 0.30
        
        call = black_scholes_price(S, K, T, r, sigma, OptionType.CALL, q)
        put = black_scholes_price(S, K, T, r, sigma, OptionType.PUT, q)
        
        lhs = call - put
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        
        assert abs(lhs - rhs) < 1e-8
    
    def test_parity_model_method(self):
        """Test put-call parity using model method."""
        model = BlackScholesModel()
        
        result_call = model.price(100, 105, 0.3, 0.05, 0.22, OptionType.CALL, 0.01)
        result_put = model.price(100, 105, 0.3, 0.05, 0.22, OptionType.PUT, 0.01)
        
        passes, error = model.put_call_parity_check(
            result_call.price, result_put.price,
            100, 105, 0.3, 0.05, 0.01
        )
        
        assert passes
        assert error < 1e-6


class TestGreeks:
    """Test Greek calculations."""
    
    def test_call_delta_range(self):
        """Test call delta is between 0 and 1."""
        greeks = black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        assert 0 < greeks.delta < 1
    
    def test_put_delta_range(self):
        """Test put delta is between -1 and 0."""
        greeks = black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT
        )
        assert -1 < greeks.delta < 0
    
    def test_gamma_positive(self):
        """Test gamma is always positive."""
        greeks = black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        assert greeks.gamma > 0
    
    def test_vega_positive(self):
        """Test vega is always positive."""
        greeks = black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        assert greeks.vega > 0
    
    def test_theta_negative_call(self):
        """Test theta is typically negative for calls."""
        greeks = black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        assert greeks.theta < 0
    
    def test_call_rho_positive(self):
        """Test call rho is positive (higher rates → higher call)."""
        greeks = black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        assert greeks.rho > 0
    
    def test_put_rho_negative(self):
        """Test put rho is negative (higher rates → lower put)."""
        greeks = black_scholes_greeks(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT
        )
        assert greeks.rho < 0
    
    def test_gamma_same_for_call_put(self):
        """Test gamma is same for call and put at same strike."""
        gamma_call = black_scholes_greeks(
            100, 100, 0.5, 0.05, 0.20, OptionType.CALL
        ).gamma
        gamma_put = black_scholes_greeks(
            100, 100, 0.5, 0.05, 0.20, OptionType.PUT
        ).gamma
        
        assert abs(gamma_call - gamma_put) < 1e-10
    
    def test_vega_same_for_call_put(self):
        """Test vega is same for call and put at same strike."""
        vega_call = black_scholes_greeks(
            100, 100, 0.5, 0.05, 0.20, OptionType.CALL
        ).vega
        vega_put = black_scholes_greeks(
            100, 100, 0.5, 0.05, 0.20, OptionType.PUT
        ).vega
        
        assert abs(vega_call - vega_put) < 1e-10
    
    def test_delta_sensitivity(self):
        """Test delta increases with spot price for calls."""
        delta_low = black_scholes_greeks(90, 100, 0.5, 0.05, 0.20, OptionType.CALL).delta
        delta_mid = black_scholes_greeks(100, 100, 0.5, 0.05, 0.20, OptionType.CALL).delta
        delta_high = black_scholes_greeks(110, 100, 0.5, 0.05, 0.20, OptionType.CALL).delta
        
        assert delta_low < delta_mid < delta_high


class TestBlackScholesModel:
    """Test BlackScholesModel class."""
    
    def test_price_returns_result(self):
        """Test model returns full result object."""
        model = BlackScholesModel()
        result = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20
        )
        
        assert result.model_id == "black-scholes-merton"
        assert isinstance(result.price, float)
        assert hasattr(result.greeks, 'delta')
        assert isinstance(result.d1, float)
        assert isinstance(result.d2, float)
    
    def test_invalid_inputs_raise(self):
        """Test invalid inputs raise ValueError."""
        model = BlackScholesModel()
        
        with pytest.raises(ValueError):
            model.price(spot=-100, strike=100, time_to_expiry=0.5,
                       risk_free_rate=0.05, volatility=0.20)
        
        with pytest.raises(ValueError):
            model.price(spot=100, strike=100, time_to_expiry=-0.5,
                       risk_free_rate=0.05, volatility=0.20)
    
    def test_warnings_for_edge_cases(self):
        """Test warnings are generated for edge cases."""
        model = BlackScholesModel()
        
        # Very near expiry (below threshold)
        result = model.price(
            spot=100, strike=100, time_to_expiry=1e-11,
            risk_free_rate=0.05, volatility=0.20
        )
        assert result.warnings is not None
        assert any('expiry' in w.lower() for w in result.warnings)


class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(
        spot=st.floats(min_value=10, max_value=1000),
        strike=st.floats(min_value=10, max_value=1000),
        time_to_expiry=st.floats(min_value=0.01, max_value=5),
        rate=st.floats(min_value=-0.1, max_value=0.2),
        vol=st.floats(min_value=0.05, max_value=2.0),
        div=st.floats(min_value=0, max_value=0.1),
    )
    @settings(max_examples=100)
    def test_put_call_parity_property(self, spot, strike, time_to_expiry, rate, vol, div):
        """Test put-call parity holds for random valid parameters."""
        assume(spot > 0 and strike > 0)
        
        call = black_scholes_price(spot, strike, time_to_expiry, rate, vol, OptionType.CALL, div)
        put = black_scholes_price(spot, strike, time_to_expiry, rate, vol, OptionType.PUT, div)
        
        lhs = call - put
        rhs = spot * np.exp(-div * time_to_expiry) - strike * np.exp(-rate * time_to_expiry)
        
        assert abs(lhs - rhs) < 1e-6
    
    @given(
        spot=st.floats(min_value=50, max_value=150),
        strike=st.floats(min_value=50, max_value=150),
        vol=st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_price_positive(self, spot, strike, vol):
        """Test prices are always non-negative."""
        assume(spot > 0 and strike > 0)
        
        call = black_scholes_price(spot, strike, 0.5, 0.05, vol, OptionType.CALL)
        put = black_scholes_price(spot, strike, 0.5, 0.05, vol, OptionType.PUT)
        
        assert call >= 0
        assert put >= 0
    
    @given(
        spot=st.floats(min_value=80, max_value=120),
        strike=st.floats(min_value=80, max_value=120),
    )
    @settings(max_examples=50)
    def test_delta_bounds(self, spot, strike):
        """Test delta stays within valid bounds."""
        assume(spot > 0 and strike > 0)
        
        greeks_call = black_scholes_greeks(spot, strike, 0.5, 0.05, 0.25, OptionType.CALL)
        greeks_put = black_scholes_greeks(spot, strike, 0.5, 0.05, 0.25, OptionType.PUT)
        
        assert 0 <= greeks_call.delta <= 1
        assert -1 <= greeks_put.delta <= 0
