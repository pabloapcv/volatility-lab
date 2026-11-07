"""Pydantic schemas for pricing endpoints."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class VarianceReduction(str, Enum):
    NONE = "none"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    BOTH = "both"


class GreeksResponse(BaseModel):
    """Option Greeks."""
    delta: float = Field(description="Delta (dV/dS)")
    gamma: float = Field(description="Gamma (d²V/dS²)")
    vega: float = Field(description="Vega per 1% vol change")
    theta: float = Field(description="Theta per calendar day")
    rho: float = Field(description="Rho per 1% rate change")


class BlackScholesRequest(BaseModel):
    """Black-Scholes pricing request."""
    spot: float = Field(gt=0, description="Current underlying price")
    strike: float = Field(gt=0, description="Option strike price")
    time_to_expiry: float = Field(ge=0, description="Time to expiration in years")
    risk_free_rate: float = Field(description="Continuously compounded risk-free rate")
    volatility: float = Field(ge=0, description="Annualized volatility")
    option_type: OptionType = Field(default=OptionType.CALL)
    dividend_yield: float = Field(default=0.0, description="Continuous dividend yield")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "spot": 500.0,
                    "strike": 510.0,
                    "time_to_expiry": 0.25,
                    "risk_free_rate": 0.04,
                    "volatility": 0.22,
                    "option_type": "call",
                    "dividend_yield": 0.01
                }
            ]
        }
    }


class BlackScholesResponse(BaseModel):
    """Black-Scholes pricing response."""
    price: float = Field(description="Theoretical option price")
    greeks: GreeksResponse
    d1: float = Field(description="d1 parameter")
    d2: float = Field(description="d2 parameter")
    model_id: Literal["black-scholes-merton"] = "black-scholes-merton"
    assumptions: list[str] = Field(
        default=[
            "Log-normal price distribution",
            "Constant volatility",
            "Continuous trading",
            "No transaction costs",
            "European exercise only"
        ]
    )
    warnings: list[str] | None = None


class BinomialRequest(BaseModel):
    """Binomial pricing request."""
    spot: float = Field(gt=0, description="Current underlying price")
    strike: float = Field(gt=0, description="Option strike price")
    time_to_expiry: float = Field(ge=0, description="Time to expiration in years")
    risk_free_rate: float = Field(description="Continuously compounded risk-free rate")
    volatility: float = Field(ge=0, description="Annualized volatility")
    option_type: OptionType = Field(default=OptionType.CALL)
    dividend_yield: float = Field(default=0.0, description="Continuous dividend yield")
    steps: int = Field(default=100, ge=1, le=1000, description="Number of tree steps")
    exercise_style: ExerciseStyle = Field(default=ExerciseStyle.EUROPEAN)
    
    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v: int) -> int:
        if v > 500:
            pass  # Warning: high step count may be slow
        return v


class BinomialResponse(BaseModel):
    """Binomial pricing response."""
    price: float = Field(description="Option price")
    greeks: GreeksResponse
    steps: int
    exercise_style: ExerciseStyle
    early_exercise_nodes: int = Field(
        description="Number of nodes where early exercise is optimal"
    )
    convergence_estimate: float | None = Field(
        description="Estimated error from doubling steps"
    )
    model_id: Literal["crr-binomial"] = "crr-binomial"
    assumptions: list[str] = Field(
        default=[
            "Recombining binomial tree",
            "Cox-Ross-Rubinstein parameters",
            "Discrete time steps"
        ]
    )
    warnings: list[str] | None = None


class ConfidenceInterval(BaseModel):
    """Confidence interval for Monte Carlo estimate."""
    lower: float
    upper: float
    level: float = Field(description="Confidence level (e.g., 0.95)")


class MonteCarloRequest(BaseModel):
    """Monte Carlo pricing request."""
    spot: float = Field(gt=0, description="Current underlying price")
    strike: float = Field(gt=0, description="Option strike price")
    time_to_expiry: float = Field(ge=0, description="Time to expiration in years")
    risk_free_rate: float = Field(description="Continuously compounded risk-free rate")
    volatility: float = Field(ge=0, description="Annualized volatility")
    option_type: OptionType = Field(default=OptionType.CALL)
    dividend_yield: float = Field(default=0.0, description="Continuous dividend yield")
    n_paths: int = Field(default=100000, ge=100, le=10000000, description="Number of paths")
    n_steps: int = Field(default=100, ge=1, le=1000, description="Time steps per path")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    variance_reduction: VarianceReduction = Field(default=VarianceReduction.ANTITHETIC)
    confidence_level: float = Field(default=0.95, ge=0.5, lt=1.0)


class MonteCarloResponse(BaseModel):
    """Monte Carlo pricing response."""
    price: float = Field(description="Option price estimate")
    std_error: float = Field(description="Standard error of estimate")
    confidence_interval: ConfidenceInterval
    n_paths: int
    n_steps: int
    seed: int | None
    variance_reduction: VarianceReduction
    control_variate_beta: float | None = Field(
        description="Control variate coefficient if used"
    )
    model_id: Literal["monte-carlo-gbm"] = "monte-carlo-gbm"
    assumptions: list[str] = Field(
        default=[
            "Geometric Brownian Motion",
            "Risk-neutral valuation",
            "Constant parameters"
        ]
    )
    warnings: list[str] | None = None
