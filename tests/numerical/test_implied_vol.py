"""
Tests for implied volatility solver.

Tests include:
- Round-trip accuracy (IV → Price → IV)
- No-arbitrage bound rejection
- Edge cases
- Bid/Ask/Mid IV
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from volatilitylab.pricing.black_scholes import (
    OptionType,
    black_scholes_price,
)
from volatilitylab.volatility.implied_vol import (
    ImpliedVolSolver,
    IVStatus,
    implied_volatility,
)


class TestImpliedVolatility:
    """Test implied volatility calculations."""

    def test_round_trip_atm(self):
        """Test IV round-trip for ATM option."""
        original_iv = 0.25
        price = black_scholes_price(
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=original_iv,
            option_type=OptionType.CALL,
        )

        solved_iv = implied_volatility(
            price=price,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert solved_iv is not None
        assert abs(solved_iv - original_iv) < 1e-6

    def test_round_trip_itm(self):
        """Test IV round-trip for ITM call."""
        original_iv = 0.30
        price = black_scholes_price(
            spot=120,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=original_iv,
            option_type=OptionType.CALL,
        )

        solved_iv = implied_volatility(
            price=price,
            spot=120,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert solved_iv is not None
        assert abs(solved_iv - original_iv) < 1e-5

    def test_round_trip_otm(self):
        """Test IV round-trip for OTM call."""
        original_iv = 0.35
        price = black_scholes_price(
            spot=90,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=original_iv,
            option_type=OptionType.CALL,
        )

        solved_iv = implied_volatility(
            price=price,
            spot=90,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert solved_iv is not None
        assert abs(solved_iv - original_iv) < 1e-5

    def test_round_trip_put(self):
        """Test IV round-trip for put option."""
        original_iv = 0.22
        price = black_scholes_price(
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=original_iv,
            option_type=OptionType.PUT,
        )

        solved_iv = implied_volatility(
            price=price,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.PUT,
        )

        assert solved_iv is not None
        assert abs(solved_iv - original_iv) < 1e-6

    def test_with_dividend(self):
        """Test IV with dividend yield."""
        original_iv = 0.28
        price = black_scholes_price(
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=original_iv,
            option_type=OptionType.CALL,
            dividend_yield=0.02,
        )

        solved_iv = implied_volatility(
            price=price,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            dividend_yield=0.02,
        )

        assert solved_iv is not None
        assert abs(solved_iv - original_iv) < 1e-5


class TestArbitrageBounds:
    """Test no-arbitrage bound checking."""

    def test_price_below_intrinsic_rejected(self):
        """Test price below intrinsic value is rejected."""
        solver = ImpliedVolSolver()

        # Call with S=110, K=100 has intrinsic of 10
        # Price of 5 violates lower bound
        result = solver.solve(
            price=5,
            spot=110,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert result.status == IVStatus.ARBITRAGE_VIOLATION
        assert result.iv is None

    def test_price_above_spot_rejected(self):
        """Test call price above spot is rejected."""
        solver = ImpliedVolSolver()

        # Call cannot be worth more than the underlying
        result = solver.solve(
            price=110,
            spot=100,
            strike=90,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert result.status == IVStatus.ARBITRAGE_VIOLATION

    def test_put_above_strike_rejected(self):
        """Test put price above discounted strike is rejected."""
        solver = ImpliedVolSolver()

        # Put cannot be worth more than K*e^(-rT)
        result = solver.solve(
            price=110,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.PUT,
        )

        assert result.status == IVStatus.ARBITRAGE_VIOLATION

    def test_valid_price_accepted(self):
        """Test valid price is accepted."""
        solver = ImpliedVolSolver()

        result = solver.solve(
            price=10,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert result.status == IVStatus.SUCCESS
        assert result.iv is not None


class TestImpliedVolSolver:
    """Test ImpliedVolSolver class."""

    def test_solve_returns_result(self):
        """Test solver returns full result object."""
        solver = ImpliedVolSolver()
        result = solver.solve(
            price=10,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert hasattr(result, "iv")
        assert hasattr(result, "status")
        assert hasattr(result, "price_error")
        assert hasattr(result, "iterations")
        assert hasattr(result, "bounds")

    def test_price_error_small(self):
        """Test repricing error is small for successful solve."""
        solver = ImpliedVolSolver()

        original_price = 8.5
        result = solver.solve(
            price=original_price,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        if result.status == IVStatus.SUCCESS:
            assert result.price_error is not None
            assert result.price_error < 1e-6

    def test_iterations_reasonable(self):
        """Test solver uses reasonable number of iterations."""
        solver = ImpliedVolSolver()

        result = solver.solve(
            price=10,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        # Brent's method should converge quickly
        assert result.iterations < 50

    def test_round_trip_check(self):
        """Test round-trip verification method."""
        solver = ImpliedVolSolver()

        passes, error = solver.round_trip_check(
            iv=0.25,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert passes
        assert error < 1e-6

    def test_invalid_price_rejected(self):
        """Test negative/zero price is rejected."""
        solver = ImpliedVolSolver()

        result = solver.solve(
            price=-5,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert result.status == IVStatus.INVALID_INPUT


class TestBidAskIV:
    """Test bid/ask/mid IV calculation."""

    def test_bid_ask_mid(self):
        """Test bid/ask/mid IV calculation."""
        solver = ImpliedVolSolver()

        # Generate realistic bid/ask from known IV
        true_iv = 0.25
        mid_price = black_scholes_price(
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=true_iv,
            option_type=OptionType.CALL,
        )

        bid = mid_price * 0.98
        ask = mid_price * 1.02

        results = solver.solve_bid_ask_mid(
            bid=bid,
            ask=ask,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert results["bid_iv"].status == IVStatus.SUCCESS
        assert results["ask_iv"].status == IVStatus.SUCCESS
        assert results["mid_iv"].status == IVStatus.SUCCESS

        # Bid IV < Mid IV < Ask IV
        assert results["bid_iv"].iv < results["mid_iv"].iv < results["ask_iv"].iv


class TestEdgeCases:
    """Test edge cases."""

    def test_near_expiry(self):
        """Test very short time to expiry."""
        solver = ImpliedVolSolver()

        # Very short time to expiry
        result = solver.solve(
            price=0.5,
            spot=100,
            strike=100.5,
            time_to_expiry=0.001,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        # Should either succeed or report no solution
        assert result.status in [
            IVStatus.SUCCESS,
            IVStatus.NO_SOLUTION,
            IVStatus.ARBITRAGE_VIOLATION,
        ]

    def test_high_volatility(self):
        """Test high volatility pricing and solving."""
        high_iv = 1.5  # 150% volatility
        price = black_scholes_price(
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=high_iv,
            option_type=OptionType.CALL,
        )

        solved_iv = implied_volatility(
            price=price,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert solved_iv is not None
        assert abs(solved_iv - high_iv) < 0.01

    def test_low_volatility(self):
        """Test low volatility pricing and solving."""
        low_iv = 0.05  # 5% volatility
        price = black_scholes_price(
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            volatility=low_iv,
            option_type=OptionType.CALL,
        )

        solved_iv = implied_volatility(
            price=price,
            spot=100,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        assert solved_iv is not None
        assert abs(solved_iv - low_iv) < 0.001


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        iv=st.floats(min_value=0.15, max_value=0.8),
        spot=st.floats(min_value=95, max_value=105),
        strike=st.floats(min_value=95, max_value=105),
        tte=st.floats(min_value=0.2, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_round_trip_property(self, iv, spot, strike, tte):
        """Test IV round-trip for ATM-ish options where vega is significant."""
        assume(spot > 0 and strike > 0 and tte > 0)

        # Generate price from IV
        price = black_scholes_price(
            spot=spot,
            strike=strike,
            time_to_expiry=tte,
            risk_free_rate=0.05,
            volatility=iv,
            option_type=OptionType.CALL,
        )

        if price > 0.50:  # Skip low-value options
            solved_iv = implied_volatility(
                price=price,
                spot=spot,
                strike=strike,
                time_to_expiry=tte,
                risk_free_rate=0.05,
                option_type=OptionType.CALL,
            )

            if solved_iv is not None:
                # Round-trip should be accurate for ATM-ish options
                assert abs(solved_iv - iv) < 0.001

    @given(
        price=st.floats(min_value=-10, max_value=200),
        spot=st.floats(min_value=50, max_value=150),
    )
    @settings(max_examples=50)
    def test_arbitrage_detection(self, price, spot):
        """Test arbitrage violations are detected."""
        assume(spot > 0)

        solver = ImpliedVolSolver()
        result = solver.solve(
            price=price,
            spot=spot,
            strike=100,
            time_to_expiry=0.5,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        # Either succeeds with valid IV or reports specific status
        assert result.status in [
            IVStatus.SUCCESS,
            IVStatus.NO_SOLUTION,
            IVStatus.ARBITRAGE_VIOLATION,
            IVStatus.INVALID_INPUT,
            IVStatus.CONVERGENCE_FAILED,
        ]


class TestImpliedVolBatch:
    """Vectorized IV must match the scalar Brent solver."""

    def test_batch_matches_scalar_round_trip(self):
        from volatilitylab.volatility.implied_vol import implied_volatility_batch

        spots = np.array([100.0, 100.0, 50.0, 200.0])
        strikes = np.array([100.0, 110.0, 50.0, 180.0])
        ttes = np.array([0.5, 0.25, 1.0, 0.1])
        vols = np.array([0.20, 0.35, 0.15, 0.40])
        is_call = np.array([True, True, False, False])
        prices = np.array(
            [
                float(
                    black_scholes_price(
                        spots[i],
                        strikes[i],
                        ttes[i],
                        0.05,
                        vols[i],
                        OptionType.CALL if is_call[i] else OptionType.PUT,
                    )
                )
                for i in range(len(spots))
            ]
        )
        batch = implied_volatility_batch(prices, spots, strikes, ttes, 0.05, is_call, 0.0)
        np.testing.assert_allclose(batch, vols, atol=1e-6)

    def test_batch_matches_solver_on_spy_sample(self):
        from pathlib import Path

        from volatilitylab.data.snapshots import SnapshotManager
        from volatilitylab.volatility.implied_vol import implied_volatility_batch

        result = SnapshotManager(Path("./data/snapshots")).load_latest("SPY")
        assert result is not None
        df, meta = result
        sample = df.iloc[:: max(1, len(df) // 40)].head(40)
        spot = float(meta["spot"])
        tte = sample["time_to_expiry"].to_numpy(dtype=float)
        stored_as_days = tte > 3.0
        years = np.where(stored_as_days, tte / 365.0, tte)
        strike = sample["strike"].to_numpy(dtype=float)
        is_call = sample["option_type"].astype(str).str.lower().str.contains("call").to_numpy()
        bid = np.nan_to_num(sample["bid"].to_numpy(dtype=float), nan=0.0)
        ask = np.nan_to_num(sample["ask"].to_numpy(dtype=float), nan=0.0)
        mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, np.nan)
        keep = np.isfinite(mid) & (mid > 0) & (years >= 1.0 / 365.0)
        solver = ImpliedVolSolver()
        scalar = []
        for i, ok in enumerate(keep):
            if not ok:
                scalar.append(np.nan)
                continue
            opt = OptionType.CALL if is_call[i] else OptionType.PUT
            res = solver.solve(float(mid[i]), spot, float(strike[i]), float(years[i]), 0.045, opt)
            scalar.append(res.iv if res.status == IVStatus.SUCCESS else np.nan)
        scalar = np.array(scalar, dtype=float)
        batch = implied_volatility_batch(mid, spot, strike, years, 0.045, is_call, 0.0)
        both_nan = np.isnan(scalar) & np.isnan(batch)
        both_ok = np.isfinite(scalar) & np.isfinite(batch)
        assert np.all(both_nan | both_ok)
        if np.any(both_ok):
            np.testing.assert_allclose(batch[both_ok], scalar[both_ok], atol=1e-6)
