"""
Monte Carlo Option Pricing Engine

Implements risk-neutral geometric Brownian motion simulation for European options.
Features:
- Configurable paths, time steps, and random seed for reproducibility
- Standard errors and confidence intervals
- Antithetic variates variance reduction
- Control variates using delta hedge
- Convergence diagnostics

GBM dynamics:
    dS_t = (r - q) S_t dt + σ S_t dW_t
    
Discretized:
    S_{t+Δt} = S_t * exp((r - q - σ²/2)Δt + σ√Δt * Z)
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from volatilitylab.pricing.black_scholes import OptionType, black_scholes_price


class VarianceReduction(str, Enum):
    """Variance reduction technique."""
    NONE = "none"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    BOTH = "both"


class ConfidenceInterval(NamedTuple):
    """Confidence interval for Monte Carlo estimate."""
    lower: float
    upper: float
    level: float


class ConvergenceDiagnostics(NamedTuple):
    """Convergence diagnostics for Monte Carlo simulation."""
    batch_prices: NDArray[np.floating]
    batch_std_errors: NDArray[np.floating]
    cumulative_means: NDArray[np.floating]
    final_std_error: float
    effective_samples: int


@dataclass(frozen=True)
class MonteCarloResult:
    """Result container for Monte Carlo pricing."""
    price: float
    std_error: float
    confidence_interval: ConfidenceInterval
    n_paths: int
    n_steps: int
    seed: int | None
    variance_reduction: VarianceReduction
    control_variate_beta: float | None
    convergence: ConvergenceDiagnostics | None
    model_id: str = "monte-carlo-gbm"
    warnings: list[str] | None = None


def _simulate_gbm_paths(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> NDArray[np.floating]:
    """
    Simulate GBM paths using log-normal discretization.
    
    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1)
        Simulated price paths including initial spot
    """
    dt = time_to_expiry / n_steps
    drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt)
    
    if antithetic:
        # Generate half the paths, then mirror
        half_paths = n_paths // 2
        Z = rng.standard_normal((half_paths, n_steps))
        Z = np.vstack([Z, -Z])
        if n_paths % 2 == 1:
            # Add one more path if odd
            Z = np.vstack([Z, rng.standard_normal((1, n_steps))])
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    
    log_returns = drift + diffusion * Z
    log_prices = np.cumsum(log_returns, axis=1)
    
    # Prepend initial spot
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(log_prices)
    
    return paths


def _compute_payoffs(
    terminal_prices: NDArray[np.floating],
    strike: float,
    option_type: OptionType,
) -> NDArray[np.floating]:
    """Compute option payoffs at expiry."""
    if option_type == OptionType.CALL:
        return np.maximum(terminal_prices - strike, 0)
    return np.maximum(strike - terminal_prices, 0)


def _control_variate_adjustment(
    payoffs: NDArray[np.floating],
    terminal_prices: NDArray[np.floating],
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: OptionType,
) -> tuple[NDArray[np.floating], float]:
    """
    Apply control variate using underlying asset as control.
    
    The underlying's expected terminal value under risk-neutral measure is:
    E[S_T] = S_0 * e^((r-q)T)
    
    Returns
    -------
    adjusted_payoffs : ndarray
        Payoffs adjusted by control variate
    beta : float
        Optimal control variate coefficient
    """
    expected_terminal = spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)
    control = terminal_prices - expected_terminal
    
    # Optimal beta minimizes variance
    cov_matrix = np.cov(payoffs, control)
    var_control = cov_matrix[1, 1]
    
    if var_control > 1e-12:
        beta = cov_matrix[0, 1] / var_control
    else:
        beta = 0.0
    
    adjusted = payoffs - beta * control
    return adjusted, beta


def monte_carlo_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType = OptionType.CALL,
    dividend_yield: float = 0.0,
    n_paths: int = 100000,
    n_steps: int = 100,
    seed: int | None = None,
    variance_reduction: VarianceReduction = VarianceReduction.ANTITHETIC,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate option price using Monte Carlo simulation.
    
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
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps per path
    seed : int, optional
        Random seed for reproducibility
    variance_reduction : VarianceReduction
        Variance reduction technique
        
    Returns
    -------
    float
        Option price estimate
    """
    if time_to_expiry <= 0:
        # At expiry
        if option_type == OptionType.CALL:
            return max(spot - strike, 0)
        return max(strike - spot, 0)
    
    rng = np.random.default_rng(seed)
    
    use_antithetic = variance_reduction in (VarianceReduction.ANTITHETIC, VarianceReduction.BOTH)
    use_control = variance_reduction in (VarianceReduction.CONTROL_VARIATE, VarianceReduction.BOTH)
    
    paths = _simulate_gbm_paths(
        spot, time_to_expiry, risk_free_rate, dividend_yield, volatility,
        n_paths, n_steps, rng, antithetic=use_antithetic
    )
    
    terminal_prices = paths[:, -1]
    payoffs = _compute_payoffs(terminal_prices, strike, option_type)
    
    if use_control:
        payoffs, _ = _control_variate_adjustment(
            payoffs, terminal_prices, spot, strike, time_to_expiry,
            risk_free_rate, dividend_yield, volatility, option_type
        )
    
    discount = np.exp(-risk_free_rate * time_to_expiry)
    price = discount * np.mean(payoffs)
    
    return float(price)


class MonteCarloModel:
    """
    Monte Carlo pricing engine with full diagnostics.
    
    Example
    -------
    >>> model = MonteCarloModel(n_paths=100000, seed=42)
    >>> result = model.price(
    ...     spot=100, strike=100, time_to_expiry=0.25,
    ...     risk_free_rate=0.05, volatility=0.20, option_type=OptionType.CALL
    ... )
    >>> print(f"Price: {result.price:.4f} ± {result.std_error:.4f}")
    """
    
    MODEL_ID = "monte-carlo-gbm"
    
    def __init__(
        self,
        n_paths: int = 100000,
        n_steps: int = 100,
        seed: int | None = None,
        variance_reduction: VarianceReduction = VarianceReduction.ANTITHETIC,
    ):
        """
        Initialize Monte Carlo engine.
        
        Parameters
        ----------
        n_paths : int
            Default number of simulation paths
        n_steps : int
            Default number of time steps
        seed : int, optional
            Random seed for reproducibility
        variance_reduction : VarianceReduction
            Default variance reduction technique
        """
        if n_paths < 100:
            raise ValueError("n_paths must be at least 100")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.variance_reduction = variance_reduction
    
    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = 0.0,
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = None,
        variance_reduction: VarianceReduction | None = None,
        confidence_level: float = 0.95,
        compute_convergence: bool = True,
        n_batches: int = 10,
    ) -> MonteCarloResult:
        """
        Calculate option price with full diagnostics.
        
        Parameters
        ----------
        confidence_level : float
            Confidence level for interval (default 0.95)
        compute_convergence : bool
            Whether to compute convergence diagnostics
        n_batches : int
            Number of batches for convergence analysis
            
        Returns
        -------
        MonteCarloResult
            Contains price, std error, CI, and diagnostics
        """
        paths = n_paths if n_paths is not None else self.n_paths
        steps = n_steps if n_steps is not None else self.n_steps
        random_seed = seed if seed is not None else self.seed
        var_red = variance_reduction if variance_reduction is not None else self.variance_reduction
        
        warnings = []
        
        # Input validation
        if spot <= 0 or strike <= 0:
            raise ValueError("Spot and strike must be positive")
        if time_to_expiry < 0:
            raise ValueError("Time to expiry cannot be negative")
        if volatility < 0:
            raise ValueError("Volatility cannot be negative")
        
        if time_to_expiry <= 0:
            # At expiry
            if option_type == OptionType.CALL:
                intrinsic = max(spot - strike, 0)
            else:
                intrinsic = max(strike - spot, 0)
            return MonteCarloResult(
                price=intrinsic,
                std_error=0.0,
                confidence_interval=ConfidenceInterval(intrinsic, intrinsic, confidence_level),
                n_paths=0,
                n_steps=0,
                seed=random_seed,
                variance_reduction=var_red,
                control_variate_beta=None,
                convergence=None,
                model_id=self.MODEL_ID,
                warnings=["At expiry: returning intrinsic value"],
            )
        
        rng = np.random.default_rng(random_seed)
        
        use_antithetic = var_red in (VarianceReduction.ANTITHETIC, VarianceReduction.BOTH)
        use_control = var_red in (VarianceReduction.CONTROL_VARIATE, VarianceReduction.BOTH)
        
        # Simulate paths
        all_paths = _simulate_gbm_paths(
            spot, time_to_expiry, risk_free_rate, dividend_yield, volatility,
            paths, steps, rng, antithetic=use_antithetic
        )
        
        terminal_prices = all_paths[:, -1]
        payoffs = _compute_payoffs(terminal_prices, strike, option_type)
        
        control_beta = None
        if use_control:
            payoffs, control_beta = _control_variate_adjustment(
                payoffs, terminal_prices, spot, strike, time_to_expiry,
                risk_free_rate, dividend_yield, volatility, option_type
            )
        
        discount = np.exp(-risk_free_rate * time_to_expiry)
        discounted_payoffs = discount * payoffs
        
        # Price and standard error
        price = float(np.mean(discounted_payoffs))
        std_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(paths))
        
        # Confidence interval
        z = stats.norm.ppf((1 + confidence_level) / 2)
        ci = ConfidenceInterval(
            lower=price - z * std_error,
            upper=price + z * std_error,
            level=confidence_level,
        )
        
        # Convergence diagnostics
        convergence = None
        if compute_convergence and n_batches > 1:
            batch_size = paths // n_batches
            batch_prices = []
            batch_stds = []
            cumulative_means = []
            
            for i in range(n_batches):
                end_idx = (i + 1) * batch_size
                batch = discounted_payoffs[:end_idx]
                batch_prices.append(np.mean(batch))
                batch_stds.append(np.std(batch, ddof=1) / np.sqrt(end_idx))
                cumulative_means.append(np.mean(batch))
            
            # Effective sample size (accounting for antithetic correlation)
            if use_antithetic:
                half = paths // 2
                corr = np.corrcoef(discounted_payoffs[:half], discounted_payoffs[half:2*half])[0, 1]
                effective = int(paths / (1 + corr)) if corr > -0.99 else paths
            else:
                effective = paths
            
            convergence = ConvergenceDiagnostics(
                batch_prices=np.array(batch_prices),
                batch_std_errors=np.array(batch_stds),
                cumulative_means=np.array(cumulative_means),
                final_std_error=std_error,
                effective_samples=effective,
            )
        
        if paths < 10000:
            warnings.append("Low path count: consider 100k+ for stable estimates")
        
        return MonteCarloResult(
            price=price,
            std_error=std_error,
            confidence_interval=ci,
            n_paths=paths,
            n_steps=steps,
            seed=random_seed,
            variance_reduction=var_red,
            control_variate_beta=control_beta,
            convergence=convergence,
            model_id=self.MODEL_ID,
            warnings=warnings if warnings else None,
        )
    
    def verify_confidence_interval(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = 0.0,
        n_experiments: int = 100,
        n_paths: int = 10000,
        confidence_level: float = 0.95,
    ) -> dict:
        """
        Verify CI coverage by repeated experiments.
        
        Returns
        -------
        dict
            Contains coverage rate, expected rate, and pass/fail
        """
        # Get reference price from Black-Scholes
        bs_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility,
            option_type, dividend_yield
        )
        
        covered = 0
        for i in range(n_experiments):
            result = self.price(
                spot, strike, time_to_expiry, risk_free_rate, volatility,
                option_type, dividend_yield, n_paths=n_paths, seed=i,
                confidence_level=confidence_level, compute_convergence=False
            )
            if result.confidence_interval.lower <= bs_price <= result.confidence_interval.upper:
                covered += 1
        
        coverage = covered / n_experiments
        # Use normal approximation for expected variation
        expected_std = np.sqrt(confidence_level * (1 - confidence_level) / n_experiments)
        
        return {
            "coverage_rate": coverage,
            "expected_rate": confidence_level,
            "n_experiments": n_experiments,
            "within_2_std": abs(coverage - confidence_level) < 2 * expected_std,
            "reference_price": bs_price,
        }
