"""
Implied Volatility Solver

Implements a robust implied volatility solver using Brent's method with:
- No-arbitrage bound checking and rejection
- Bid/Ask/Mid IV calculation
- Numerical stability for edge cases

No-arbitrage bounds for European options:
    Call: max(0, S*e^(-qT) - K*e^(-rT)) ≤ C ≤ S*e^(-qT)
    Put:  max(0, K*e^(-rT) - S*e^(-qT)) ≤ P ≤ K*e^(-rT)
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from volatilitylab.pricing.black_scholes import OptionType, black_scholes_price


class IVStatus(str, Enum):
    """Status of implied volatility calculation."""
    SUCCESS = "success"
    NO_SOLUTION = "no_solution"
    ARBITRAGE_VIOLATION = "arbitrage_violation"
    CONVERGENCE_FAILED = "convergence_failed"
    INVALID_INPUT = "invalid_input"


class ArbitrageBounds(NamedTuple):
    """No-arbitrage price bounds."""
    lower: float
    upper: float
    forward: float
    discount_factor: float


@dataclass(frozen=True)
class ImpliedVolResult:
    """Result container for implied volatility calculation."""
    iv: float | None
    status: IVStatus
    price_error: float | None
    iterations: int
    bounds: ArbitrageBounds
    message: str | None = None


def _compute_arbitrage_bounds(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> ArbitrageBounds:
    """
    Compute no-arbitrage price bounds for European options.
    
    Returns
    -------
    ArbitrageBounds
        Lower bound, upper bound, forward price, and discount factor
    """
    discount = np.exp(-risk_free_rate * time_to_expiry)
    forward_discount = np.exp(-dividend_yield * time_to_expiry)
    forward = spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)
    
    if option_type == OptionType.CALL:
        # Call bounds: max(0, S*e^(-qT) - K*e^(-rT)) ≤ C ≤ S*e^(-qT)
        intrinsic = max(0, spot * forward_discount - strike * discount)
        upper = spot * forward_discount
    else:
        # Put bounds: max(0, K*e^(-rT) - S*e^(-qT)) ≤ P ≤ K*e^(-rT)
        intrinsic = max(0, strike * discount - spot * forward_discount)
        upper = strike * discount
    
    return ArbitrageBounds(
        lower=intrinsic,
        upper=upper,
        forward=forward,
        discount_factor=discount,
    )


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType = OptionType.CALL,
    dividend_yield: float = 0.0,
    initial_guess: float = 0.20,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float | None:
    """
    Calculate implied volatility using Brent's method.
    
    Parameters
    ----------
    price : float
        Market option price
    spot : float
        Current underlying price
    strike : float
        Option strike price
    time_to_expiry : float
        Time to expiration in years
    risk_free_rate : float
        Continuously compounded risk-free rate
    option_type : OptionType
        CALL or PUT
    dividend_yield : float
        Continuous dividend yield
    initial_guess : float
        Starting volatility for search
    tolerance : float
        Price tolerance for convergence
    max_iterations : int
        Maximum solver iterations
        
    Returns
    -------
    float or None
        Implied volatility if found, None otherwise
    """
    result = ImpliedVolSolver().solve(
        price, spot, strike, time_to_expiry, risk_free_rate,
        option_type, dividend_yield, tolerance, max_iterations
    )
    return result.iv


class ImpliedVolSolver:
    """
    Robust implied volatility solver with diagnostics.
    
    Uses Brent's method for root finding with careful handling of:
    - No-arbitrage bound violations
    - Near-expiry options
    - Deep ITM/OTM options
    - Numerical edge cases
    
    Example
    -------
    >>> solver = ImpliedVolSolver()
    >>> result = solver.solve(
    ...     price=5.50, spot=100, strike=100, time_to_expiry=0.25,
    ...     risk_free_rate=0.05, option_type=OptionType.CALL
    ... )
    >>> if result.status == IVStatus.SUCCESS:
    ...     print(f"IV: {result.iv:.2%}")
    """
    
    # Volatility search bounds
    VOL_MIN = 1e-6
    VOL_MAX = 5.0  # 500% annualized
    
    def solve(
        self,
        price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = 0.0,
        tolerance: float = 1e-8,
        max_iterations: int = 100,
    ) -> ImpliedVolResult:
        """
        Solve for implied volatility with full diagnostics.
        
        Returns
        -------
        ImpliedVolResult
            Contains IV (if found), status, error, and bounds
        """
        # Input validation
        if spot <= 0 or strike <= 0:
            return ImpliedVolResult(
                iv=None, status=IVStatus.INVALID_INPUT, price_error=None, iterations=0,
                bounds=ArbitrageBounds(0, 0, 0, 0), message="Spot and strike must be positive"
            )
        if time_to_expiry < 0:
            return ImpliedVolResult(
                iv=None, status=IVStatus.INVALID_INPUT, price_error=None, iterations=0,
                bounds=ArbitrageBounds(0, 0, 0, 0), message="Time to expiry cannot be negative"
            )
        if price <= 0:
            return ImpliedVolResult(
                iv=None, status=IVStatus.INVALID_INPUT, price_error=None, iterations=0,
                bounds=ArbitrageBounds(0, 0, 0, 0), message="Price must be positive"
            )
        
        # Compute no-arbitrage bounds
        bounds = _compute_arbitrage_bounds(
            spot, strike, time_to_expiry, risk_free_rate, dividend_yield, option_type
        )
        
        # Check arbitrage bounds
        if price < bounds.lower - tolerance:
            return ImpliedVolResult(
                iv=None, status=IVStatus.ARBITRAGE_VIOLATION, price_error=None, iterations=0,
                bounds=bounds, message=f"Price {price:.4f} below intrinsic value {bounds.lower:.4f}"
            )
        if price > bounds.upper + tolerance:
            return ImpliedVolResult(
                iv=None, status=IVStatus.ARBITRAGE_VIOLATION, price_error=None, iterations=0,
                bounds=bounds, message=f"Price {price:.4f} above upper bound {bounds.upper:.4f}"
            )
        
        # Handle near-expiry case
        if time_to_expiry < 1e-6:
            return ImpliedVolResult(
                iv=None, status=IVStatus.NO_SOLUTION, price_error=None, iterations=0,
                bounds=bounds, message="At expiry: IV undefined"
            )
        
        # Define objective function
        def objective(vol: float) -> float:
            model_price = black_scholes_price(
                spot, strike, time_to_expiry, risk_free_rate, vol, option_type, dividend_yield
            )
            return model_price - price
        
        # Check boundary values
        try:
            f_low = objective(self.VOL_MIN)
            f_high = objective(self.VOL_MAX)
        except Exception:
            return ImpliedVolResult(
                iv=None, status=IVStatus.CONVERGENCE_FAILED, price_error=None, iterations=0,
                bounds=bounds, message="Failed to evaluate objective at boundaries"
            )
        
        # Check if solution exists in interval
        if f_low * f_high > 0:
            # No sign change - check if price is at boundary
            if abs(f_low) < tolerance:
                return ImpliedVolResult(
                    iv=self.VOL_MIN, status=IVStatus.SUCCESS, price_error=abs(f_low),
                    iterations=1, bounds=bounds, message="Near minimum volatility"
                )
            if abs(f_high) < tolerance:
                return ImpliedVolResult(
                    iv=self.VOL_MAX, status=IVStatus.SUCCESS, price_error=abs(f_high),
                    iterations=1, bounds=bounds, message="Near maximum volatility"
                )
            return ImpliedVolResult(
                iv=None, status=IVStatus.NO_SOLUTION, price_error=None, iterations=0,
                bounds=bounds, message="No volatility in valid range produces target price"
            )
        
        # Solve using Brent's method
        try:
            result = optimize.brentq(
                objective, self.VOL_MIN, self.VOL_MAX,
                xtol=tolerance, maxiter=max_iterations, full_output=True
            )
            iv, info = result[0], result[1]
            
            # Verify solution
            final_price = black_scholes_price(
                spot, strike, time_to_expiry, risk_free_rate, iv, option_type, dividend_yield
            )
            error = abs(final_price - price)
            
            return ImpliedVolResult(
                iv=iv, status=IVStatus.SUCCESS, price_error=error,
                iterations=info.iterations, bounds=bounds
            )
        except ValueError as e:
            return ImpliedVolResult(
                iv=None, status=IVStatus.CONVERGENCE_FAILED, price_error=None, iterations=0,
                bounds=bounds, message=str(e)
            )
    
    def solve_bid_ask_mid(
        self,
        bid: float,
        ask: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = 0.0,
        tolerance: float = 1e-8,
    ) -> dict[str, ImpliedVolResult]:
        """
        Calculate bid, ask, and mid implied volatilities.
        
        Returns
        -------
        dict
            Keys: 'bid_iv', 'ask_iv', 'mid_iv' with ImpliedVolResult values
        """
        mid = (bid + ask) / 2
        
        return {
            'bid_iv': self.solve(
                bid, spot, strike, time_to_expiry, risk_free_rate,
                option_type, dividend_yield, tolerance
            ),
            'ask_iv': self.solve(
                ask, spot, strike, time_to_expiry, risk_free_rate,
                option_type, dividend_yield, tolerance
            ),
            'mid_iv': self.solve(
                mid, spot, strike, time_to_expiry, risk_free_rate,
                option_type, dividend_yield, tolerance
            ),
        }
    
    def round_trip_check(
        self,
        iv: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = 0.0,
        tolerance: float = 1e-6,
    ) -> tuple[bool, float]:
        """
        Verify IV round-trip: IV → Price → IV matches original.
        
        Returns
        -------
        tuple[bool, float]
            (passes, absolute_error)
        """
        # Price at given IV
        price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, iv, option_type, dividend_yield
        )
        
        # Solve back to IV
        result = self.solve(
            price, spot, strike, time_to_expiry, risk_free_rate,
            option_type, dividend_yield, tolerance
        )
        
        if result.iv is None:
            return False, float('inf')
        
        error = abs(result.iv - iv)
        return error < tolerance, error
