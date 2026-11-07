"""Pydantic schemas for data endpoints."""

from pydantic import BaseModel, Field


class OptionsChainRequest(BaseModel):
    """Options chain request."""
    symbol: str = Field(description="Underlying symbol (e.g., SPY, QQQ, NVDA)")
    expirations: list[str] | None = Field(
        default=None, 
        description="Specific expirations to load"
    )
    valid_only: bool = Field(
        default=True, 
        description="Return only valid quality contracts"
    )
    save_snapshot: bool = Field(
        default=False, 
        description="Save as snapshot for reproducibility"
    )


class QuoteQualitySummary(BaseModel):
    """Summary of quote quality distribution."""
    total: int
    valid: int = 0
    stale: int = 0
    crossed: int = 0
    zero_bid: int = 0
    wide_spread: int = 0
    low_open_interest: int = 0
    invalid: int = 0
    valid_pct: float


class OptionsChainResponse(BaseModel):
    """Options chain response."""
    symbol: str
    spot: float
    observed_at: str
    expirations: list[str]
    total_contracts: int
    valid_contracts: int
    quality_summary: QuoteQualitySummary
    snapshot_id: str | None = None
    contracts: list[dict]  # Simplified contract data


class SnapshotInfo(BaseModel):
    """Snapshot metadata."""
    snapshot_id: str
    symbol: str
    provider: str
    observed_at: str
    ingested_at: str
    spot: float
    expirations: list[str]
    total_contracts: int
    valid_contracts: int


class SnapshotListResponse(BaseModel):
    """List of available snapshots."""
    snapshots: list[SnapshotInfo]
    total: int
