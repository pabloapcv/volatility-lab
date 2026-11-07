"""
Cox-Ross-Rubinstein (CRR) Binomial Option Pricing Model

Implements European and American option pricing using a recombining binomial tree.
Supports configurable time steps, early exercise, and numerical Greeks estimation.

Mathematical framework:
    u = e^(σ√Δt)           Up factor
    d = 1/u = e^(-σ√Δt)    Down factor
    p = (e^((r-q)Δt) - d) / (u - d)  Risk-neutral probability

Tree construction builds forward, valuation uses backward induction.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from volatilitylab.pricing.black_scholes import OptionType


class ExerciseStyle(str, Enum):
    """Option exercise style."""
    EUROPEAN = "european"
    AMERICAN = "american"


class NumericalGreeks(NamedTuple):
    """Container for numerically estimated Greeks."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True)
class BinomialResult:
    """Result container for binomial pricing."""
    price: float
    greeks: NumericalGreeks
    steps: int
    exercise_style: ExerciseStyle
    early_exercise_nodes: int
    convergence_estimate: float | None
    model_id: str = "crr-binomial"
    warnings: list[str] | None = None


@dataclass(frozen=True)
class ConvergenceAnalysis:
    """Convergence analysis results."""
    steps: NDArray[np.int_]
    prices: NDArray[np.floating]
    reference_price: float
    errors: NDArray[np.floating]
    convergence_rate: float


def _build_price_tree(
    spot: float,
    up: float,
    down: float,
    steps: int,
) -> NDArray[np.floating]:
    """
    Build terminal asset prices using vectorized computation.
    
    At step N, there are N+1 possible prices:
    S_j = S * u^j * d^(N-j) for j = 0, 1, ..., N
    """
    j = np.arange(steps + 1)
    # Number of up moves = j, number of down moves = steps - j
    return spot * (up ** j) * (down ** (steps - j))


def binomial_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType = OptionType.CALL,
    dividend_yield: float = 0.0,
    steps: int = 100,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
) -> float:
    """
    Calculate option price using CRR binomial tree.
    
    Parameters
    ----------
    spot : float
        Current underlying price
    strike : float
        Option strike price
    time_to_expiry : float
        Time to expiration in years
    risk_free_rate : float
        Continuously compounded risk-free rate
    volatility : float
        Annualized volatility
    option_type : OptionType
        CALL or PUT
    dividend_yield : float
        Continuous dividend yield
    steps : int
        Number of time steps in tree
    exercise_style : ExerciseStyle
        EUROPEAN or AMERICAN
        
    Returns
    -------
    float
        Option price
    """
    if time_to_expiry <= 0:
        # At expiry
        if option_type == OptionType.CALL:
            return max(spot - strike, 0)
        return max(strike - spot, 0)
    
    if volatility <= 0:
        raise ValueError("Volatility must be positive")
    if steps < 1:
        raise ValueError("Steps must be at least 1")
    
    dt = time_to_expiry / steps
    
    # CRR parameters
    u = np.exp(volatility * np.sqrt(dt))
    d = 1.0 / u
    
    # Risk-neutral probability
    drift = np.exp((risk_free_rate - dividend_yield) * dt)
    p = (drift - d) / (u - d)
    
    # Validate probability
    if not (0 < p < 1):
        raise ValueError(
            f"Invalid risk-neutral probability {p:.4f}. "
            "Check that parameters are consistent."
        )
    
    discount = np.exp(-risk_free_rate * dt)
    
    # Terminal asset prices
    asset_prices = _build_price_tree(spot, u, d, steps)
    
    # Terminal payoffs
    if option_type == OptionType.CALL:
        option_values = np.maximum(asset_prices - strike, 0)
    else:
        option_values = np.maximum(strike - asset_prices, 0)
    
    # Backward induction
    for step in range(steps - 1, -1, -1):
        # Asset prices at this step
        j = np.arange(step + 1)
        current_asset = spot * (u ** j) * (d ** (step - j))
        
        # Expected discounted value
        option_values = discount * (p * option_values[1:] + (1 - p) * option_values[:-1])
        
        # American: check early exercise
        if exercise_style == ExerciseStyle.AMERICAN:
            if option_type == OptionType.CALL:
                intrinsic = np.maximum(current_asset - strike, 0)
            else:
                intrinsic = np.maximum(strike - current_asset, 0)
            option_values = np.maximum(option_values, intrinsic)
    
    return float(option_values[0])


def _estimate_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float,
    steps: int,
    exercise_style: ExerciseStyle,
) -> NumericalGreeks:
    """
    Estimate Greeks using finite differences.
    
    Delta and Gamma are extracted from the tree structure.
    Vega, Theta, Rho use central differences.
    """
    base_price = binomial_price(
        spot, strike, time_to_expiry, risk_free_rate, volatility,
        option_type, dividend_yield, steps, exercise_style
    )
    
    # For delta and gamma, use the tree structure
    # Build a small tree to extract values at step 1 and 2
    dt = time_to_expiry / steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1.0 / u
    
    # Prices at step 1
    S_u = spot * u
    S_d = spot * d
    
    # Option values at step 1 (using full tree for each)
    V_u = binomial_price(
        S_u, strike, time_to_expiry - dt, risk_free_rate, volatility,
        option_type, dividend_yield, max(1, steps - 1), exercise_style
    )
    V_d = binomial_price(
        S_d, strike, time_to_expiry - dt, risk_free_rate, volatility,
        option_type, dividend_yield, max(1, steps - 1), exercise_style
    )
    
    # Delta from tree
    delta = (V_u - V_d) / (S_u - S_d)
    
    # For gamma, need values at step 2
    if steps >= 2 and time_to_expiry > 2 * dt:
        S_uu = spot * u * u
        S_ud = spot  # u * d = 1
        S_dd = spot * d * d
        
        V_uu = binomial_price(
            S_uu, strike, time_to_expiry - 2*dt, risk_free_rate, volatility,
            option_type, dividend_yield, max(1, steps - 2), exercise_style
        )
        V_ud = binomial_price(
            S_ud, strike, time_to_expiry - 2*dt, risk_free_rate, volatility,
            option_type, dividend_yield, max(1, steps - 2), exercise_style
        )
        V_dd = binomial_price(
            S_dd, strike, time_to_expiry - 2*dt, risk_free_rate, volatility,
            option_type, dividend_yield, max(1, steps - 2), exercise_style
        )
        
        delta_up = (V_uu - V_ud) / (S_uu - S_ud)
        delta_down = (V_ud - V_dd) / (S_ud - S_dd)
        h = 0.5 * (S_uu - S_dd)
        gamma = (delta_up - delta_down) / h
    else:
        gamma = 0.0
    
    # Vega: central difference on volatility (per 1% = 0.01)
    vol_bump = 0.01
    V_vol_up = binomial_price(
        spot, strike, time_to_expiry, risk_free_rate, volatility + vol_bump,
        option_type, dividend_yield, steps, exercise_style
    )
    V_vol_down = binomial_price(
        spot, strike, time_to_expiry, risk_free_rate, max(0.001, volatility - vol_bump),
        option_type, dividend_yield, steps, exercise_style
    )
    vega = (V_vol_up - V_vol_down) / 2 * 0.01  # Scale to per 1% move
    
    # Theta: forward difference on time (per day)
    time_bump = 1 / 365
    if time_to_expiry > time_bump:
        V_time_down = binomial_price(
            spot, strike, time_to_expiry - time_bump, risk_free_rate, volatility,
            option_type, dividend_yield, steps, exercise_style
        )
        theta = (V_time_down - base_price)  # Already per day
    else:
        theta = 0.0
    
    # Rho: central difference on rate (per 1% = 0.01)
    rate_bump = 0.01
    V_rate_up = binomial_price(
        spot, strike, time_to_expiry, risk_free_rate + rate_bump, volatility,
        option_type, dividend_yield, steps, exercise_style
    )
    V_rate_down = binomial_price(
        spot, strike, time_to_expiry, risk_free_rate - rate_bump, volatility,
        option_type, dividend_yield, steps, exercise_style
    )
    rho = (V_rate_up - V_rate_down) / 2 * 0.01  # Scale to per 1% move
    
    return NumericalGreeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
    )


def _count_early_exercise_nodes(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float,
    steps: int,
) -> int:
    """Count nodes where early exercise is optimal."""
    if time_to_expiry <= 0:
        return 0
    
    dt = time_to_expiry / steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1.0 / u
    drift = np.exp((risk_free_rate - dividend_yield) * dt)
    p = (drift - d) / (u - d)
    discount = np.exp(-risk_free_rate * dt)
    
    # Terminal payoffs
    asset_prices = _build_price_tree(spot, u, d, steps)
    if option_type == OptionType.CALL:
        option_values = np.maximum(asset_prices - strike, 0)
    else:
        option_values = np.maximum(strike - asset_prices, 0)
    
    early_exercise_count = 0
    
    for step in range(steps - 1, -1, -1):
        j = np.arange(step + 1)
        current_asset = spot * (u ** j) * (d ** (step - j))
        
        continuation = discount * (p * option_values[1:] + (1 - p) * option_values[:-1])
        
        if option_type == OptionType.CALL:
            intrinsic = np.maximum(current_asset - strike, 0)
        else:
            intrinsic = np.maximum(strike - current_asset, 0)
        
        early_exercise_count += np.sum(intrinsic > continuation + 1e-10)
        option_values = np.maximum(continuation, intrinsic)
    
    return int(early_exercise_count)


class BinomialModel:
    """
    CRR Binomial pricing model with Greeks and convergence analysis.
    
    Example
    -------
    >>> model = BinomialModel(steps=200)
    >>> result = model.price(
    ...     spot=100, strike=100, time_to_expiry=0.25,
    ...     risk_free_rate=0.05, volatility=0.20, option_type=OptionType.PUT,
    ...     exercise_style=ExerciseStyle.AMERICAN
    ... )
    >>> print(f"American Put: {result.price:.4f}")
    """
    
    MODEL_ID = "crr-binomial"
    
    def __init__(self, steps: int = 100):
        """
        Initialize binomial model.
        
        Parameters
        ----------
        steps : int
            Default number of time steps
        """
        if steps < 1:
            raise ValueError("Steps must be at least 1")
        self.steps = steps
    
    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = 0.0,
        exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
        steps: int | None = None,
        compute_greeks: bool = True,
    ) -> BinomialResult:
        """
        Calculate option price with full diagnostics.
        
        Returns
        -------
        BinomialResult
            Contains price, greeks, convergence info, and warnings
        """
        n_steps = steps if steps is not None else self.steps
        warnings = []
        
        # Input validation
        if spot <= 0 or strike <= 0:
            raise ValueError("Spot and strike must be positive")
        if time_to_expiry < 0:
            raise ValueError("Time to expiry cannot be negative")
        if volatility < 0:
            raise ValueError("Volatility cannot be negative")
        
        if n_steps < 50:
            warnings.append(f"Low step count ({n_steps}): consider 100+ for accuracy")
        
        price = binomial_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility,
            option_type, dividend_yield, n_steps, exercise_style
        )
        
        # Greeks
        if compute_greeks and time_to_expiry > 0:
            greeks = _estimate_greeks(
                spot, strike, time_to_expiry, risk_free_rate, volatility,
                option_type, dividend_yield, n_steps, exercise_style
            )
        else:
            greeks = NumericalGreeks(0, 0, 0, 0, 0)
        
        # Early exercise count for American options
        if exercise_style == ExerciseStyle.AMERICAN:
            early_ex = _count_early_exercise_nodes(
                spot, strike, time_to_expiry, risk_free_rate, volatility,
                option_type, dividend_yield, n_steps
            )
        else:
            early_ex = 0
        
        # Convergence estimate (compare with doubled steps)
        if n_steps >= 10:
            price_double = binomial_price(
                spot, strike, time_to_expiry, risk_free_rate, volatility,
                option_type, dividend_yield, n_steps * 2, exercise_style
            )
            convergence_est = abs(price_double - price)
        else:
            convergence_est = None
        
        return BinomialResult(
            price=price,
            greeks=greeks,
            steps=n_steps,
            exercise_style=exercise_style,
            early_exercise_nodes=early_ex,
            convergence_estimate=convergence_est,
            model_id=self.MODEL_ID,
            warnings=warnings if warnings else None,
        )
    
    def convergence_analysis(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = 0.0,
        exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
        reference_price: float | None = None,
        step_range: tuple[int, int, int] = (10, 500, 10),
    ) -> ConvergenceAnalysis:
        """
        Analyze convergence of binomial prices as steps increase.
        
        Parameters
        ----------
        reference_price : float, optional
            Known reference price (e.g., Black-Scholes for European)
        step_range : tuple
            (start, stop, step) for number of tree steps
            
        Returns
        -------
        ConvergenceAnalysis
            Steps, prices, errors, and estimated convergence rate
        """
        from volatilitylab.pricing.black_scholes import black_scholes_price
        
        steps_arr = np.arange(*step_range)
        prices = np.array([
            binomial_price(
                spot, strike, time_to_expiry, risk_free_rate, volatility,
                option_type, dividend_yield, int(n), exercise_style
            )
            for n in steps_arr
        ])
        
        # Use Black-Scholes as reference for European options
        if reference_price is None and exercise_style == ExerciseStyle.EUROPEAN:
            reference_price = black_scholes_price(
                spot, strike, time_to_expiry, risk_free_rate, volatility,
                option_type, dividend_yield
            )
        elif reference_price is None:
            # Use highest-resolution price as reference
            reference_price = prices[-1]
        
        errors = np.abs(prices - reference_price)
        
        # Estimate convergence rate: error ~ O(1/N^rate)
        # log(error) ~ -rate * log(N) + const
        valid = errors > 1e-12
        if np.sum(valid) > 2:
            log_n = np.log(steps_arr[valid])
            log_err = np.log(errors[valid])
            slope, _ = np.polyfit(log_n, log_err, 1)
            rate = -slope
        else:
            rate = 0.0
        
        return ConvergenceAnalysis(
            steps=steps_arr,
            prices=prices,
            reference_price=reference_price,
            errors=errors,
            convergence_rate=rate,
        )
