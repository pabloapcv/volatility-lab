#!/usr/bin/env python3
"""
Fetch one compact real trading-day of SPY/QQQ/NVDA option chains via yfinance.

Keeps a few nearby expirations and near-ATM strikes so the committed Parquet
stays well under a few MB. Synthetic snapshots remain as an offline fallback.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

KEEP_COLUMNS = [
    "contract_symbol",
    "strike",
    "expiration",
    "option_type",
    "bid",
    "ask",
    "last_price",
    "mid_price",
    "volume",
    "open_interest",
    "time_to_expiry",
    "moneyness",
    "log_moneyness",
    "yf_implied_vol",
    "observed_at",
    "quality",
    "quality_reason",
]


def _compact(df: pd.DataFrame, spot: float, n_expiries: int = 8, moneyness_band: float = 0.15) -> pd.DataFrame:
    """Keep nearby expiries and a strike window around spot."""
    out = df.copy()
    out["expiration"] = pd.to_datetime(out["expiration"])
    expiries = sorted(out["expiration"].unique())[:n_expiries]
    out = out[out["expiration"].isin(expiries)]
    lo, hi = spot * (1 - moneyness_band), spot * (1 + moneyness_band)
    out = out[(out["strike"] >= lo) & (out["strike"] <= hi)]
    if "quality" in out.columns:
        valid = out[out["quality"] == "valid"]
        if len(valid) >= 40:
            out = valid
    keep = [c for c in KEEP_COLUMNS if c in out.columns]
    return out[keep].reset_index(drop=True)


def _save(symbol: str, df: pd.DataFrame, metadata: dict, snapshot_dir: Path) -> Path:
    snapshot_id = f"{symbol}_yfinance_sample"
    compact_meta = {
        **metadata,
        "snapshot_id": snapshot_id,
        "symbol": symbol,
        "provider": "yfinance",
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "total_contracts": int(len(df)),
        "valid_contracts": int((df["quality"] == "valid").sum()) if "quality" in df.columns else int(len(df)),
        "note": (
            "Compact real options chain from yfinance for dashboard demo. "
            "Nearby expirations and near-ATM strikes only."
        ),
        "is_synthetic": False,
    }
    parquet_path = snapshot_dir / f"{snapshot_id}_cleaned.parquet"
    df.to_parquet(parquet_path, index=False)
    with open(snapshot_dir / f"{snapshot_id}_metadata.json", "w") as f:
        json.dump(compact_meta, f, indent=2, default=str)
    return parquet_path


def fetch_symbol(symbol: str) -> tuple[pd.DataFrame, dict]:
    from volatilitylab.data.options_chain import load_option_chain

    df, metadata = load_option_chain(symbol, apply_filters=True)
    compact = _compact(df, spot=float(metadata["spot"]))
    # Keep the first 8 expirations that actually remain after compacting.
    remaining_exps = sorted(pd.to_datetime(compact["expiration"]).dt.strftime("%Y-%m-%d").unique().tolist())
    metadata["expirations"] = remaining_exps
    metadata["spot"] = float(metadata["spot"])
    return compact, metadata


def compact_from_cache(symbol: str, snapshot_dir: Path) -> tuple[pd.DataFrame, dict] | None:
    cache_parquet = snapshot_dir / "cache" / f"{symbol}_chain.parquet"
    cache_meta = snapshot_dir / "cache" / f"{symbol}_meta.json"
    if not cache_parquet.exists() or not cache_meta.exists():
        return None
    df = pd.read_parquet(cache_parquet)
    with open(cache_meta) as f:
        metadata = json.load(f)
    compact = _compact(df, spot=float(metadata["spot"]))
    remaining_exps = sorted(pd.to_datetime(compact["expiration"]).dt.strftime("%Y-%m-%d").unique().tolist())
    metadata["expirations"] = remaining_exps
    metadata["provider"] = "yfinance"
    return compact, metadata


def main() -> None:
    snapshot_dir = Path(__file__).parent.parent / "data" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for symbol in ("SPY", "QQQ", "NVDA"):
        print(f"Fetching {symbol}...")
        result = None
        try:
            result = fetch_symbol(symbol)
            print(f"  live yfinance: {len(result[0])} contracts, spot={result[1]['spot']:.2f}")
        except Exception as exc:
            print(f"  live fetch failed ({exc}); trying cache...")
            result = compact_from_cache(symbol, snapshot_dir)
            if result is None:
                print(f"  SKIP {symbol}: no live data and no cache")
                continue
            print(f"  cache: {len(result[0])} contracts, spot={result[1]['spot']:.2f}")

        df, metadata = result
        path = _save(symbol, df, metadata, snapshot_dir)
        size_kb = path.stat().st_size / 1024
        print(f"  saved {path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
