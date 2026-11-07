"""Data API endpoints for options chains and snapshots."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.schemas.data import (
    OptionsChainRequest,
    OptionsChainResponse,
    QuoteQualitySummary,
    SnapshotInfo,
    SnapshotListResponse,
)
from volatilitylab.data.options_chain import OptionsChainLoader, QuoteQuality
from volatilitylab.data.snapshots import SnapshotManager

router = APIRouter(prefix="/data", tags=["data"])

SNAPSHOT_DIR = Path("./data/snapshots")


@router.get("/underlyings")
async def list_underlyings() -> dict:
    """List supported underlying symbols."""
    return {
        "underlyings": [
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "type": "ETF"},
            {"symbol": "QQQ", "name": "Invesco QQQ Trust", "type": "ETF"},
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "type": "Stock"},
            {"symbol": "AAPL", "name": "Apple Inc.", "type": "Stock"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "type": "Stock"},
            {"symbol": "TSLA", "name": "Tesla Inc.", "type": "Stock"},
            {"symbol": "AMD", "name": "Advanced Micro Devices", "type": "Stock"},
            {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "type": "ETF"},
        ]
    }


@router.post("/chain", response_model=OptionsChainResponse)
async def load_options_chain(request: OptionsChainRequest) -> OptionsChainResponse:
    """
    Load options chain for a symbol.
    
    Applies quote quality filters and optionally saves as snapshot for
    reproducible analysis.
    """
    try:
        loader = OptionsChainLoader(cache_dir=SNAPSHOT_DIR / "cache")
        df, metadata = loader.load(
            symbol=request.symbol,
            expirations=request.expirations,
            use_cache=True,
        )
        
        # Get quality summary
        quality_counts = df['quality'].value_counts().to_dict() if 'quality' in df.columns else {}
        quality_summary = QuoteQualitySummary(
            total=len(df),
            valid=quality_counts.get('valid', 0),
            stale=quality_counts.get('stale', 0),
            crossed=quality_counts.get('crossed', 0),
            zero_bid=quality_counts.get('zero_bid', 0),
            wide_spread=quality_counts.get('wide_spread', 0),
            low_open_interest=quality_counts.get('low_open_interest', 0),
            invalid=quality_counts.get('invalid', 0),
            valid_pct=quality_counts.get('valid', 0) / len(df) * 100 if len(df) > 0 else 0,
        )
        
        # Filter to valid if requested
        if request.valid_only and 'quality' in df.columns:
            df = df[df['quality'] == QuoteQuality.VALID.value]
        
        # Save snapshot if requested
        snapshot_id = None
        if request.save_snapshot:
            manager = SnapshotManager(SNAPSHOT_DIR)
            snapshot_id = manager.save(df, metadata)
        
        # Convert to contract list (simplified)
        contracts = []
        for _, row in df.head(500).iterrows():  # Limit to 500 for response size
            contracts.append({
                'strike': float(row['strike']),
                'expiration': str(row['expiration']),
                'option_type': row['option_type'],
                'bid': float(row.get('bid', 0) or 0),
                'ask': float(row.get('ask', 0) or 0),
                'mid_price': float(row.get('mid_price', 0) or 0),
                'volume': int(row.get('volume', 0) or 0),
                'open_interest': int(row.get('open_interest', 0) or 0),
                'quality': row.get('quality', 'unknown'),
            })
        
        return OptionsChainResponse(
            symbol=metadata['symbol'],
            spot=metadata['spot'],
            observed_at=metadata['observed_at'],
            expirations=metadata['expirations'],
            total_contracts=metadata['total_contracts'],
            valid_contracts=metadata['valid_contracts'],
            quality_summary=quality_summary,
            snapshot_id=snapshot_id,
            contracts=contracts,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chain: {str(e)}")


@router.get("/snapshots", response_model=SnapshotListResponse)
async def list_snapshots(symbol: str | None = None) -> SnapshotListResponse:
    """List available snapshots, optionally filtered by symbol."""
    try:
        manager = SnapshotManager(SNAPSHOT_DIR)
        snapshots = manager.list_snapshots(symbol)
        
        return SnapshotListResponse(
            snapshots=[
                SnapshotInfo(
                    snapshot_id=s['snapshot_id'],
                    symbol=s.get('symbol', 'UNKNOWN'),
                    provider=s.get('provider', 'yfinance'),
                    observed_at=s.get('observed_at', ''),
                    ingested_at=s.get('ingested_at', ''),
                    spot=s.get('spot', 0),
                    expirations=s.get('expirations', []),
                    total_contracts=s.get('total_contracts', 0),
                    valid_contracts=s.get('valid_contracts', 0),
                    is_synthetic=(
                        str(s.get('provider', '')).lower() in ('synthetic', 'demo')
                        or str(s.get('snapshot_id', '')).endswith('_sample_offline')
                    ),
                    note=s.get('note'),
                )
                for s in snapshots
            ],
            total=len(snapshots),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list snapshots: {str(e)}")


@router.get("/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: str, valid_only: bool = True) -> dict:
    """Load a specific snapshot."""
    try:
        from volatilitylab.data.snapshots import load_snapshot
        
        snapshot_path = SNAPSHOT_DIR / f"{snapshot_id}_cleaned.parquet"
        if not snapshot_path.exists():
            raise HTTPException(status_code=404, detail=f"Snapshot {snapshot_id} not found")
        
        df, metadata = load_snapshot(snapshot_path, valid_only=valid_only)
        
        contracts = []
        for _, row in df.head(500).iterrows():
            contracts.append({
                'strike': float(row['strike']),
                'expiration': str(row['expiration']),
                'option_type': row['option_type'],
                'bid': float(row.get('bid', 0) or 0),
                'ask': float(row.get('ask', 0) or 0),
                'mid_price': float(row.get('mid_price', 0) or 0),
                'time_to_expiry': float(row.get('time_to_expiry', 0)),
            })
        
        return {
            'metadata': metadata,
            'contracts': contracts,
            'total': len(df),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load snapshot: {str(e)}")
