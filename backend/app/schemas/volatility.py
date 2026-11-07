"""Pydantic schemas for volatility endpoints."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.schemas.pricing import OptionType


class IVStatus(str, Enum):
    SUCCESS = "success"
    NO_SOLUTION = "no_solution"
    ARBITRAGE_VIOLATION = "arbitrage_violation"
    CONVERGENCE_FAILED = "convergence_failed"
    INVALID_INPUT = "invalid_input"


class DataQuality(str, Enum):
    OBSERVED = "observed"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    SPARSE = "sparse"


class ArbitrageBounds(BaseModel):
    """No-arbitrage price bounds."""
    lower: float = Field(description="Lower bound (intrinsic value)")
    upper: float = Field(description="Upper bound")
    forward: float = Field(description="Forward price")
    discount_factor: float


class ImpliedVolRequest(BaseModel):
    """Implied volatility calculation request."""
    price: float = Field(gt=0, description="Market option price")
    spot: float = Field(gt=0, description="Current underlying price")
    strike: float = Field(gt=0, description="Option strike price")
    time_to_expiry: float = Field(gt=0, description="Time to expiration in years")
    risk_free_rate: float = Field(description="Risk-free rate")
    option_type: OptionType = Field(default=OptionType.CALL)
    dividend_yield: float = Field(default=0.0)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "price": 5.50,
                    "spot": 100.0,
                    "strike": 100.0,
                    "time_to_expiry": 0.25,
                    "risk_free_rate": 0.05,
                    "option_type": "call"
                }
            ]
        }
    }


class ImpliedVolResponse(BaseModel):
    """Implied volatility response."""
    implied_volatility: float | None = Field(description="Solved implied volatility")
    status: IVStatus
    price_error: float | None = Field(description="Repricing error")
    iterations: int
    bounds: ArbitrageBounds
    message: str | None = None


class BidAskIVRequest(BaseModel):
    """Bid/Ask implied volatility request."""
    bid: float = Field(ge=0, description="Bid price")
    ask: float = Field(gt=0, description="Ask price")
    spot: float = Field(gt=0, description="Current underlying price")
    strike: float = Field(gt=0, description="Option strike price")
    time_to_expiry: float = Field(gt=0, description="Time to expiration in years")
    risk_free_rate: float = Field(description="Risk-free rate")
    option_type: OptionType = Field(default=OptionType.CALL)
    dividend_yield: float = Field(default=0.0)


class BidAskIVResponse(BaseModel):
    """Bid/Ask/Mid implied volatility response."""
    bid_iv: float | None
    ask_iv: float | None
    mid_iv: float | None
    bid_status: IVStatus
    ask_status: IVStatus
    mid_status: IVStatus


class SurfacePoint(BaseModel):
    """Single point on volatility surface."""
    strike: float
    expiry: float
    iv: float
    quality: DataQuality
    moneyness: float
    bid_iv: float | None = None
    ask_iv: float | None = None


class SmileResponse(BaseModel):
    """Volatility smile slice."""
    expiry: float
    time_to_expiry: float
    strikes: list[float]
    ivs: list[float | None]
    moneyness: list[float]
    qualities: list[DataQuality]
    atm_iv: float | None
    skew: float | None = Field(description="Slope dIV/dK at ATM")
    convexity: float | None = Field(description="Curvature d²IV/dK² at ATM")


class TermStructurePoint(BaseModel):
    """Point on ATM term structure."""
    expiry: float
    time_to_expiry: float
    atm_iv: float
    atm_strike: float
    quality: DataQuality


class TermStructureResponse(BaseModel):
    """ATM term structure response."""
    points: list[TermStructurePoint]
    underlying: str
    spot: float


class SurfaceRequest(BaseModel):
    """Volatility surface request."""
    snapshot_id: str | None = Field(default=None, description="Snapshot ID to use")
    symbol: str | None = Field(default=None, description="Symbol to load fresh")
    n_strikes: int = Field(default=20, ge=5, le=50)
    n_expiries: int = Field(default=10, ge=2, le=20)
    moneyness_range: tuple[float, float] = Field(default=(-0.3, 0.3))


class SurfaceGridResponse(BaseModel):
    """3D surface grid for visualization."""
    moneyness: list[list[float]]
    expiry: list[list[float]]
    iv: list[list[float | None]]
    quality_mask: list[list[int]]
    observed_points: list[tuple[float, float, float]]


class SurfaceResponse(BaseModel):
    """Full volatility surface response."""
    metadata: dict
    points: list[SurfacePoint]
    grid: SurfaceGridResponse | None = None
    warnings: list[str] = []
