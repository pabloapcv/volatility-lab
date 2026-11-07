"""
Unit tests for CRR Binomial pricing model.

Tests include:
- Convergence to Black-Scholes for European options
- Early exercise for American options
- Numerical Greeks accuracy
- Edge cases
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume

from volatilitylab.pricing.binomial import (
    BinomialModel,
    ExerciseStyle,
    binomial_price,
)
from volatilitylab.pricing.black_scholes import (
    OptionType,
    black_scholes_greeks,
    black_scholes_price,
)


class TestBinomialPrice:
    """Test binomial pricing calculations."""
    
    def test_european_call_converges_to_bs(self):
        """Test European call converges to BS as steps increase."""
        bs_price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        bin_price = binomial_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL,
            steps=500, exercise_style=ExerciseStyle.EUROPEAN
        )
        
        assert abs(bin_price - bs_price) < 0.02
    
    def test_european_put_converges_to_bs(self):
        """Test European put converges to BS as steps increase."""
        bs_price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT
        )
        
        bin_price = binomial_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT,
            steps=500, exercise_style=ExerciseStyle.EUROPEAN
        )
        
        assert abs(bin_price - bs_price) < 0.02
    
    def test_american_call_no_dividend(self):
        """Test American call equals European without dividend."""
        euro = binomial_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, dividend_yield=0.0,
            steps=200, exercise_style=ExerciseStyle.EUROPEAN
        )
        
        amer = binomial_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, dividend_yield=0.0,
            steps=200, exercise_style=ExerciseStyle.AMERICAN
        )
        
        # American call without dividend should equal European
        assert abs(amer - euro) < 0.01
    
    def test_american_put_geq_european(self):
        """Test American put >= European put."""
        euro = binomial_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT,
            steps=200, exercise_style=ExerciseStyle.EUROPEAN
        )
        
        amer = binomial_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT,
            steps=200, exercise_style=ExerciseStyle.AMERICAN
        )
        
        assert amer >= euro - 0.001  # Allow tiny numerical error
    
    def test_itm_american_put_early_exercise_value(self):
        """Test deep ITM American put has early exercise value."""
        amer = binomial_price(
            spot=80, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT,
            steps=200, exercise_style=ExerciseStyle.AMERICAN
        )
        
        intrinsic = 100 - 80
        # American put should be worth at least intrinsic
        assert amer >= intrinsic - 0.01
    
    def test_at_expiry(self):
        """Test price at expiry equals intrinsic value."""
        call = binomial_price(
            spot=110, strike=100, time_to_expiry=0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        assert call == pytest.approx(10.0, abs=1e-6)
        
        put = binomial_price(
            spot=90, strike=100, time_to_expiry=0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT
        )
        assert put == pytest.approx(10.0, abs=1e-6)
    
    def test_invalid_volatility_raises(self):
        """Test zero/negative volatility raises error."""
        with pytest.raises(ValueError):
            binomial_price(
                spot=100, strike=100, time_to_expiry=0.5,
                risk_free_rate=0.05, volatility=-0.20,
                option_type=OptionType.CALL
            )
    
    def test_low_step_count_works(self):
        """Test pricing works with low step count."""
        price = binomial_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL, steps=10
        )
        assert price > 0


class TestConvergence:
    """Test binomial convergence properties."""
    
    def test_convergence_rate(self):
        """Test error decreases with steps."""
        bs_price = black_scholes_price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.25,
            option_type=OptionType.CALL
        )
        
        errors = []
        for steps in [50, 100, 200, 400]:
            bin_price = binomial_price(
                spot=100, strike=100, time_to_expiry=0.5,
                risk_free_rate=0.05, volatility=0.25,
                option_type=OptionType.CALL,
                steps=steps, exercise_style=ExerciseStyle.EUROPEAN
            )
            errors.append(abs(bin_price - bs_price))
        
        # Error should generally decrease (allow for oscillation)
        assert errors[-1] < errors[0]
    
    def test_model_convergence_analysis(self):
        """Test convergence analysis from model."""
        model = BinomialModel()
        analysis = model.convergence_analysis(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL,
            step_range=(50, 250, 50)
        )
        
        assert len(analysis.steps) == 4
        assert len(analysis.prices) == 4
        assert all(p > 0 for p in analysis.prices)
        # Convergence rate should be positive (errors decrease)
        assert analysis.convergence_rate > 0


class TestNumericalGreeks:
    """Test numerical Greeks from binomial model."""
    
    def test_delta_reasonable(self):
        """Test numerical delta is in reasonable range."""
        model = BinomialModel(steps=200)
        result = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        assert 0 < result.greeks.delta < 1
    
    def test_gamma_positive(self):
        """Test numerical gamma is positive."""
        model = BinomialModel(steps=200)
        result = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        assert result.greeks.gamma >= 0
    
    def test_vega_positive(self):
        """Test numerical vega is positive."""
        model = BinomialModel(steps=100)
        result = model.price(
            spot=100, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.CALL
        )
        
        assert result.greeks.vega > 0

    def test_vega_rho_units_match_black_scholes(self):
        """Binomial Vega/Rho must use the same per-1% units as Black-Scholes."""
        params = dict(
            spot=500, strike=510, time_to_expiry=0.25,
            risk_free_rate=0.045, volatility=0.22,
            option_type=OptionType.CALL,
        )
        bs = black_scholes_greeks(**params, dividend_yield=0.01)
        binom = BinomialModel(steps=200).price(**params, dividend_yield=0.01)

        # Workbench screenshot case: previously off by ~100x (0.99 vs 0.0099).
        assert abs(binom.greeks.vega - bs.vega) / abs(bs.vega) < 0.05
        assert abs(binom.greeks.rho - bs.rho) / abs(bs.rho) < 0.05
        assert binom.greeks.vega > 0.5  # not the 0.01-scaled value


class TestAmericanOptions:
    """Test American option specific behavior."""
    
    def test_early_exercise_counted(self):
        """Test early exercise nodes are counted for American puts."""
        model = BinomialModel(steps=100)
        result = model.price(
            spot=80, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.AMERICAN
        )
        
        # Deep ITM put should have early exercise nodes
        assert result.early_exercise_nodes > 0
    
    def test_no_early_exercise_for_european(self):
        """Test European options have zero early exercise nodes."""
        model = BinomialModel(steps=100)
        result = model.price(
            spot=80, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.20,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN
        )
        
        assert result.early_exercise_nodes == 0
    
    def test_american_call_with_dividend(self):
        """Test American call with dividend may differ from European."""
        model = BinomialModel(steps=200)
        
        euro = model.price(
            spot=100, strike=95, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.20, dividend_yield=0.05,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN
        )
        
        amer = model.price(
            spot=100, strike=95, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.20, dividend_yield=0.05,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN
        )
        
        # American should be >= European
        assert amer.price >= euro.price - 0.001


class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(
        spot=st.floats(min_value=50, max_value=150),
        strike=st.floats(min_value=50, max_value=150),
        vol=st.floats(min_value=0.1, max_value=0.8),
    )
    @settings(max_examples=50)
    def test_american_geq_european(self, spot, strike, vol):
        """Test American option >= European option."""
        assume(spot > 0 and strike > 0 and vol > 0)
        
        euro = binomial_price(
            spot=spot, strike=strike, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=vol,
            option_type=OptionType.PUT, steps=100,
            exercise_style=ExerciseStyle.EUROPEAN
        )
        
        amer = binomial_price(
            spot=spot, strike=strike, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=vol,
            option_type=OptionType.PUT, steps=100,
            exercise_style=ExerciseStyle.AMERICAN
        )
        
        assert amer >= euro - 0.01  # Allow numerical tolerance
    
    @given(
        spot=st.floats(min_value=80, max_value=120),
        steps=st.integers(min_value=50, max_value=300),
    )
    @settings(max_examples=30)
    def test_price_positive(self, spot, steps):
        """Test prices are always non-negative."""
        assume(spot > 0)
        
        price = binomial_price(
            spot=spot, strike=100, time_to_expiry=0.5,
            risk_free_rate=0.05, volatility=0.25,
            option_type=OptionType.CALL, steps=steps
        )
        
        assert price >= 0
