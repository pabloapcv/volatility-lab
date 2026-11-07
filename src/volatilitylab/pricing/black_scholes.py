"""
Black-Scholes-Merton Option Pricing Model

Implements European call and put pricing with continuous dividend yield,
analytical Greeks (Delta, Gamma, Vega, Theta, Rho), vectorized NumPy operations,
and careful handling of edge cases (expiry, near-zero volatility, deep ITM/OTM).

Mathematical formulas:
    C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
    P = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)

where:
    d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

# Numerical constants for stability
MIN_VOLATILITY = 1e-10
MIN_TIME = 1e-10
DEEP_ITM_THRESHOLD = 8.0  # Standard deviations for d1/d2
DEEP_OTM_THRESHOLD = -8.0


class OptionType(StrEnum):
    """Option type: call or put."""

    CALL = "call"
    PUT = "put"


class Greeks(NamedTuple):
    """Container for option Greeks."""

    delta: float | NDArray[np.floating]
    gamma: float | NDArray[np.floating]
    vega: float | NDArray[np.floating]
    theta: float | NDArray[np.floating]
    rho: float | NDArray[np.floating]


@dataclass(frozen=True)
class BlackScholesResult:
    """Result container for Black-Scholes pricing."""

    price: float | NDArray[np.floating]
    greeks: Greeks
    d1: float | NDArray[np.floating]
    d2: float | NDArray[np.floating]
    model_id: str = "black-scholes-merton"
    warnings: list[str] | None = None


def _compute_d1_d2(
    spot: NDArray[np.floating],
    strike: NDArray[np.floating],
    time_to_expiry: NDArray[np.floating],
    risk_free_rate: NDArray[np.floating],
    dividend_yield: NDArray[np.floating],
    volatility: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute d1 and d2 for Black-Scholes formula.

    Handles edge cases:
    - Near-zero volatility: returns ±∞ based on moneyness
    - Near-expiry: uses limiting behavior
    """
    vol_sqrt_t = volatility * np.sqrt(time_to_expiry)

    # Handle near-zero vol*sqrt(T) case
    safe_vol_sqrt_t = np.where(vol_sqrt_t < MIN_VOLATILITY, MIN_VOLATILITY, vol_sqrt_t)

    log_moneyness = np.log(spot / strike)
    drift = (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry

    d1 = (log_moneyness + drift) / safe_vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    # For near-zero vol, set d1/d2 based on whether ITM or OTM
    near_zero_vol = vol_sqrt_t < MIN_VOLATILITY
    forward = spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)
    itm = forward > strike

    d1 = np.where(near_zero_vol & itm, DEEP_ITM_THRESHOLD, d1)
    d1 = np.where(near_zero_vol & ~itm, DEEP_OTM_THRESHOLD, d1)
    d2 = np.where(near_zero_vol & itm, DEEP_ITM_THRESHOLD, d2)
    d2 = np.where(near_zero_vol & ~itm, DEEP_OTM_THRESHOLD, d2)

    return d1, d2


def _ensure_array(*args: float | NDArray) -> tuple[NDArray[np.floating], ...]:
    """Convert inputs to numpy arrays with consistent shape."""
    arrays = [np.atleast_1d(np.asarray(arg, dtype=np.float64)) for arg in args]
    # Broadcast to common shape
    shape = np.broadcast_shapes(*[a.shape for a in arrays])
    return tuple(np.broadcast_to(a, shape) for a in arrays)


def black_scholes_price(
    spot: float | NDArray,
    strike: float | NDArray,
    time_to_expiry: float | NDArray,
    risk_free_rate: float | NDArray,
    volatility: float | NDArray,
    option_type: OptionType = OptionType.CALL,
    dividend_yield: float | NDArray = 0.0,
) -> float | NDArray[np.floating]:
    """
    Calculate Black-Scholes-Merton option price.

    Parameters
    ----------
    spot : float or array
        Current underlying price
    strike : float or array
        Option strike price
    time_to_expiry : float or array
        Time to expiration in years
    risk_free_rate : float or array
        Continuously compounded risk-free rate
    volatility : float or array
        Annualized volatility (σ)
    option_type : OptionType
        CALL or PUT
    dividend_yield : float or array
        Continuous dividend yield

    Returns
    -------
    float or ndarray
        Option price(s)
    """
    S, K, T, r, sigma, q = _ensure_array(
        spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
    )

    # Handle expiry case
    at_expiry = T < MIN_TIME

    d1, d2 = _compute_d1_d2(S, K, T, r, q, sigma)

    discount = np.exp(-r * T)
    forward_discount = np.exp(-q * T)

    if option_type == OptionType.CALL:
        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)
        price = S * forward_discount * N_d1 - K * discount * N_d2
        # At expiry: max(S - K, 0)
        price = np.where(at_expiry, np.maximum(S - K, 0), price)
    else:
        N_neg_d1 = stats.norm.cdf(-d1)
        N_neg_d2 = stats.norm.cdf(-d2)
        price = K * discount * N_neg_d2 - S * forward_discount * N_neg_d1
        # At expiry: max(K - S, 0)
        price = np.where(at_expiry, np.maximum(K - S, 0), price)

    # Return scalar if input was scalar
    if price.size == 1:
        return float(price.flat[0])
    return price


def black_scholes_greeks(
    spot: float | NDArray,
    strike: float | NDArray,
    time_to_expiry: float | NDArray,
    risk_free_rate: float | NDArray,
    volatility: float | NDArray,
    option_type: OptionType = OptionType.CALL,
    dividend_yield: float | NDArray = 0.0,
) -> Greeks:
    """
    Calculate analytical Black-Scholes Greeks.

    Returns
    -------
    Greeks
        Named tuple with delta, gamma, vega, theta, rho

    Notes
    -----
    Vega is per 1% change in volatility (multiply by 0.01 for per-point).
    Theta is per calendar day (divide by 365).
    """
    S, K, T, r, sigma, q = _ensure_array(
        spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
    )

    # Handle edge cases
    at_expiry = T < MIN_TIME
    near_zero_vol = sigma < MIN_VOLATILITY
    edge_case = at_expiry | near_zero_vol

    d1, d2 = _compute_d1_d2(S, K, T, r, q, sigma)

    sqrt_T = np.sqrt(np.maximum(T, MIN_TIME))
    discount = np.exp(-r * T)
    forward_discount = np.exp(-q * T)

    # PDF and CDF values
    n_d1 = stats.norm.pdf(d1)
    N_d1 = stats.norm.cdf(d1)
    N_d2 = stats.norm.cdf(d2)
    N_neg_d1 = stats.norm.cdf(-d1)
    N_neg_d2 = stats.norm.cdf(-d2)

    # Gamma (same for calls and puts)
    gamma = forward_discount * n_d1 / (S * sigma * sqrt_T)
    gamma = np.where(edge_case, 0.0, gamma)

    # Vega (same for calls and puts, per 1% move = 0.01 vol points)
    vega = S * forward_discount * sqrt_T * n_d1 * 0.01
    vega = np.where(edge_case, 0.0, vega)

    if option_type == OptionType.CALL:
        # Delta
        delta = forward_discount * N_d1
        delta = np.where(at_expiry, np.where(S > K, 1.0, 0.0), delta)

        # Theta (per year, will convert to per day)
        theta = (
            -S * forward_discount * n_d1 * sigma / (2 * sqrt_T)
            - r * K * discount * N_d2
            + q * S * forward_discount * N_d1
        )
        theta = np.where(edge_case, 0.0, theta)
        theta = theta / 365  # Convert to per calendar day

        # Rho (per 1% change in rate)
        rho = K * T * discount * N_d2 * 0.01
        rho = np.where(at_expiry, 0.0, rho)
    else:
        # Delta (put)
        delta = -forward_discount * N_neg_d1
        delta = np.where(at_expiry, np.where(S < K, -1.0, 0.0), delta)

        # Theta (put)
        theta = (
            -S * forward_discount * n_d1 * sigma / (2 * sqrt_T)
            + r * K * discount * N_neg_d2
            - q * S * forward_discount * N_neg_d1
        )
        theta = np.where(edge_case, 0.0, theta)
        theta = theta / 365

        # Rho (put)
        rho = -K * T * discount * N_neg_d2 * 0.01
        rho = np.where(at_expiry, 0.0, rho)

    # Convert to scalars if appropriate
    def maybe_scalar(arr: NDArray) -> float | NDArray:
        if arr.size == 1:
            return float(arr.flat[0])
        return arr

    return Greeks(
        delta=maybe_scalar(delta),
        gamma=maybe_scalar(gamma),
        vega=maybe_scalar(vega),
        theta=maybe_scalar(theta),
        rho=maybe_scalar(rho),
    )


class BlackScholesModel:
    """
    Black-Scholes-Merton pricing model with Greeks.

    Provides a class-based interface for option pricing with full diagnostics.

    Example
    -------
    >>> model = BlackScholesModel()
    >>> result = model.price(
    ...     spot=100, strike=100, time_to_expiry=0.25,
    ...     risk_free_rate=0.05, volatility=0.20, option_type=OptionType.CALL
    ... )
    >>> print(f"Price: {result.price:.4f}, Delta: {result.greeks.delta:.4f}")
    """

    MODEL_ID = "black-scholes-merton"

    def price(
        self,
        spot: float | NDArray,
        strike: float | NDArray,
        time_to_expiry: float | NDArray,
        risk_free_rate: float | NDArray,
        volatility: float | NDArray,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float | NDArray = 0.0,
    ) -> BlackScholesResult:
        """
        Calculate option price and Greeks.

        Returns
        -------
        BlackScholesResult
            Contains price, greeks, d1, d2, model_id, and any warnings
        """
        warnings = []

        # Input validation and warnings
        S, K, T, r, sigma, q = _ensure_array(
            spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
        )

        if np.any(T < MIN_TIME):
            warnings.append("Near or at expiry: using intrinsic value")
        if np.any(sigma < 0.01):
            warnings.append("Very low volatility: numerical precision may be affected")
        if np.any(S <= 0) or np.any(K <= 0):
            raise ValueError("Spot and strike must be positive")
        if np.any(T < 0):
            raise ValueError("Time to expiry cannot be negative")
        if np.any(sigma < 0):
            raise ValueError("Volatility cannot be negative")

        # Check for deep ITM/OTM
        d1, d2 = _compute_d1_d2(S, K, T, r, q, sigma)
        if np.any(np.abs(d1) > 5):
            warnings.append("Deep ITM/OTM: standard warnings apply")

        price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
        )
        greeks = black_scholes_greeks(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
        )

        # Return scalar d1/d2 if appropriate
        def maybe_scalar(arr: NDArray) -> float | NDArray:
            if arr.size == 1:
                return float(arr.flat[0])
            return arr

        return BlackScholesResult(
            price=price,
            greeks=greeks,
            d1=maybe_scalar(d1),
            d2=maybe_scalar(d2),
            model_id=self.MODEL_ID,
            warnings=warnings if warnings else None,
        )

    def put_call_parity_check(
        self,
        call_price: float,
        put_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        tolerance: float = 1e-6,
    ) -> tuple[bool, float]:
        """
        Verify put-call parity: C - P = S*e^(-qT) - K*e^(-rT)

        Returns
        -------
        tuple[bool, float]
            (parity_holds, absolute_error)
        """
        discount = np.exp(-risk_free_rate * time_to_expiry)
        forward_discount = np.exp(-dividend_yield * time_to_expiry)

        lhs = call_price - put_price
        rhs = spot * forward_discount - strike * discount
        error = abs(lhs - rhs)

        return error < tolerance, error
