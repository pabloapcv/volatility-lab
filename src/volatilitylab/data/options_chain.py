"""
Options Chain Data Ingestion

Loads and normalizes options chain data from various providers (yfinance).
Applies quote quality filters and preserves data provenance.

Quality filters:
- Stale quotes (last trade > N days ago)
- Crossed markets (bid > ask)
- Zero/missing bids
- Invalid contracts (negative prices, impossible strikes)
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


class QuoteQuality(StrEnum):
    """Quote quality classification."""

    VALID = "valid"
    STALE = "stale"
    CROSSED = "crossed"
    ZERO_BID = "zero_bid"
    WIDE_SPREAD = "wide_spread"
    LOW_OI = "low_open_interest"
    INVALID = "invalid"


@dataclass
class QuoteQualityFilter:
    """Configuration for quote quality filtering."""

    max_stale_days: int = 5
    max_spread_pct: float = 0.50  # 50% bid-ask spread
    min_open_interest: int = 10
    min_volume: int = 0
    min_bid: float = 0.05

    def apply(self, df: pd.DataFrame, spot: float) -> pd.DataFrame:
        """
        Apply quality filters and add quality column.

        Parameters
        ----------
        df : DataFrame
            Raw options chain with bid, ask, volume, openInterest columns
        spot : float
            Current underlying spot price

        Returns
        -------
        DataFrame
            With added 'quality' and 'quality_reason' columns
        """
        df = df.copy()

        quality = []
        reasons = []

        for _, row in df.iterrows():
            bid = row.get("bid", 0) or 0
            ask = row.get("ask", 0) or 0
            oi = row.get("open_interest", row.get("openInterest", 0)) or 0
            _volume = row.get("volume", 0) or 0

            # Check for invalid data
            if bid < 0 or ask < 0:
                quality.append(QuoteQuality.INVALID)
                reasons.append("negative_price")
                continue

            # Zero bid
            if bid < self.min_bid:
                quality.append(QuoteQuality.ZERO_BID)
                reasons.append(f"bid={bid:.4f}")
                continue

            # Crossed market
            if bid > ask and ask > 0:
                quality.append(QuoteQuality.CROSSED)
                reasons.append(f"bid={bid:.2f}>ask={ask:.2f}")
                continue

            # Wide spread
            mid = (bid + ask) / 2 if ask > 0 else bid
            spread_pct = (ask - bid) / mid if mid > 0 else 1
            if spread_pct > self.max_spread_pct:
                quality.append(QuoteQuality.WIDE_SPREAD)
                reasons.append(f"spread={spread_pct:.1%}")
                continue

            # Low open interest
            if oi < self.min_open_interest:
                quality.append(QuoteQuality.LOW_OI)
                reasons.append(f"oi={oi}")
                continue

            quality.append(QuoteQuality.VALID)
            reasons.append("")

        df["quality"] = [q.value for q in quality]
        df["quality_reason"] = reasons

        return df


def _normalize_yfinance_chain(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    expiration: str,
    spot: float,
    observed_at: datetime,
) -> pd.DataFrame:
    """Normalize yfinance option chain format."""

    def process_side(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["option_type"] = opt_type
        df["expiration"] = pd.to_datetime(expiration)
        df["observed_at"] = observed_at

        # Calculate time to expiry
        exp_date = pd.to_datetime(expiration)
        tte = (exp_date - pd.Timestamp.now()).days / 365
        df["time_to_expiry"] = max(tte, 0)

        # Standardize column names
        df = df.rename(
            columns={
                "contractSymbol": "contract_symbol",
                "lastTradeDate": "last_trade_date",
                "lastPrice": "last_price",
                "openInterest": "open_interest",
                "impliedVolatility": "yf_implied_vol",
                "inTheMoney": "in_the_money",
            }
        )

        # Add mid price
        df["mid_price"] = (df["bid"] + df["ask"]) / 2
        df["mid_price"] = df["mid_price"].where((df["bid"] > 0) & (df["ask"] > 0), df["last_price"])

        # Add moneyness
        df["moneyness"] = df["strike"] / spot
        df["log_moneyness"] = np.log(df["moneyness"])

        return df

    calls_norm = process_side(calls, "call")
    puts_norm = process_side(puts, "put")

    return pd.concat([calls_norm, puts_norm], ignore_index=True)


def load_option_chain(
    symbol: str,
    provider: Literal["yfinance"] = "yfinance",
    expirations: list[str] | None = None,
    apply_filters: bool = True,
    filter_config: QuoteQualityFilter | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load options chain for a symbol.

    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g., 'SPY', 'QQQ', 'NVDA')
    provider : str
        Data provider ('yfinance')
    expirations : list[str], optional
        Specific expirations to load. If None, loads all available.
    apply_filters : bool
        Whether to apply quality filters
    filter_config : QuoteQualityFilter, optional
        Custom filter configuration

    Returns
    -------
    tuple[DataFrame, dict]
        (options chain, metadata)
    """
    import yfinance as yf

    observed_at = datetime.utcnow()

    ticker = yf.Ticker(symbol)

    # Get spot price
    try:
        hist = ticker.history(period="1d")
        if hist.empty:
            raise ValueError(f"No price data for {symbol}")
        spot = float(hist["Close"].iloc[-1])
    except Exception as e:
        raise ValueError(f"Failed to get spot price for {symbol}: {e}") from e

    # Get available expirations
    available_exps = list(ticker.options)
    if not available_exps:
        raise ValueError(f"No options available for {symbol}")

    if expirations is not None:
        exps_to_load = [e for e in expirations if e in available_exps]
    else:
        exps_to_load = available_exps

    all_chains = []

    for exp in exps_to_load:
        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls
            puts = chain.puts

            normalized = _normalize_yfinance_chain(calls, puts, exp, spot, observed_at)
            all_chains.append(normalized)
        except Exception:
            continue

    if not all_chains:
        raise ValueError(f"Failed to load any option chains for {symbol}")

    df = pd.concat(all_chains, ignore_index=True)

    # Apply quality filters
    if apply_filters:
        filter_cfg = filter_config or QuoteQualityFilter()
        df = filter_cfg.apply(df, spot)

    metadata = {
        "symbol": symbol,
        "provider": provider,
        "spot": spot,
        "observed_at": observed_at.isoformat(),
        "expirations": exps_to_load,
        "total_contracts": len(df),
        "valid_contracts": len(df[df["quality"] == "valid"])
        if "quality" in df.columns
        else len(df),
    }

    return df, metadata


class OptionsChainLoader:
    """
    Options chain loader with caching and snapshot support.

    Example
    -------
    >>> loader = OptionsChainLoader()
    >>> chain, meta = loader.load('SPY')
    >>> print(f"Loaded {meta['valid_contracts']} valid contracts")
    """

    SUPPORTED_SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "TSLA", "AMD", "IWM"]

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        filter_config: QuoteQualityFilter | None = None,
    ):
        """
        Initialize loader.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory for caching chains
        filter_config : QuoteQualityFilter, optional
            Custom quality filter configuration
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.filter_config = filter_config or QuoteQualityFilter()

    def load(
        self,
        symbol: str,
        expirations: list[str] | None = None,
        use_cache: bool = True,
        cache_max_age_minutes: int = 15,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Load options chain with optional caching.

        Returns
        -------
        tuple[DataFrame, dict]
            (options chain, metadata)
        """
        symbol = symbol.upper()

        # Check cache
        if use_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{symbol}_chain.parquet"
            if cache_file.exists():
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                age_minutes = (datetime.now() - mtime).total_seconds() / 60
                if age_minutes < cache_max_age_minutes:
                    df = pd.read_parquet(cache_file)
                    meta_file = self.cache_dir / f"{symbol}_meta.json"
                    if meta_file.exists():
                        import json

                        with open(meta_file) as f:
                            metadata = json.load(f)
                        metadata["from_cache"] = True
                        return df, metadata

        # Load fresh data
        df, metadata = load_option_chain(
            symbol,
            expirations=expirations,
            apply_filters=True,
            filter_config=self.filter_config,
        )

        # Cache if configured
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.cache_dir / f"{symbol}_chain.parquet")

            import json

            with open(self.cache_dir / f"{symbol}_meta.json", "w") as f:
                json.dump(metadata, f, indent=2)

        metadata["from_cache"] = False
        return df, metadata

    def get_valid_contracts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only valid quality contracts."""
        if "quality" in df.columns:
            return df[df["quality"] == QuoteQuality.VALID.value].copy()
        return df.copy()

    def quality_summary(self, df: pd.DataFrame) -> dict:
        """Get summary of quote quality distribution."""
        if "quality" not in df.columns:
            return {"total": len(df), "no_filter_applied": True}

        summary = df["quality"].value_counts().to_dict()
        summary["total"] = len(df)
        summary["valid_pct"] = summary.get("valid", 0) / len(df) * 100
        return summary
