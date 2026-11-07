"""
Snapshot Management for Reproducible Analysis

Saves and loads market data snapshots as Parquet files with full provenance
metadata (observed_at, ingested_at, provider, filters applied).
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SnapshotMetadata:
    """Metadata for a saved snapshot."""
    snapshot_id: str
    symbol: str
    provider: str
    observed_at: str
    ingested_at: str
    spot: float
    expirations: list[str]
    total_contracts: int
    valid_contracts: int
    filters_applied: dict[str, Any]
    version: str = "1.0"


def save_snapshot(
    df: pd.DataFrame,
    metadata: dict,
    snapshot_dir: Path | str,
    snapshot_id: str | None = None,
    save_raw: bool = True,
) -> tuple[Path, Path | None]:
    """
    Save options chain snapshot to Parquet with metadata.
    
    Parameters
    ----------
    df : DataFrame
        Options chain data
    metadata : dict
        Chain metadata from loader
    snapshot_dir : Path
        Directory to save snapshots
    snapshot_id : str, optional
        Custom snapshot ID. If None, generates from timestamp.
    save_raw : bool
        Whether to also save raw (unfiltered) data
        
    Returns
    -------
    tuple[Path, Path | None]
        (cleaned_path, raw_path or None)
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate snapshot ID
    if snapshot_id is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        symbol = metadata.get('symbol', 'UNKNOWN')
        snapshot_id = f"{symbol}_{timestamp}"
    
    # Add ingested_at timestamp
    df = df.copy()
    df['ingested_at'] = datetime.utcnow().isoformat()
    
    # Save cleaned data
    cleaned_path = snapshot_dir / f"{snapshot_id}_cleaned.parquet"
    df.to_parquet(cleaned_path, index=False)
    
    # Save metadata
    full_metadata = {
        'snapshot_id': snapshot_id,
        'version': '1.0',
        'ingested_at': datetime.utcnow().isoformat(),
        **metadata,
    }
    
    meta_path = snapshot_dir / f"{snapshot_id}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(full_metadata, f, indent=2, default=str)
    
    # Optionally save raw (unfiltered) data
    raw_path = None
    if save_raw and 'quality' in df.columns:
        raw_path = snapshot_dir / f"{snapshot_id}_raw.parquet"
        df.to_parquet(raw_path, index=False)
    
    return cleaned_path, raw_path


def load_snapshot(
    snapshot_path: Path | str,
    valid_only: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a saved snapshot.
    
    Parameters
    ----------
    snapshot_path : Path
        Path to snapshot Parquet file or directory with snapshot_id
    valid_only : bool
        If True, filter to only valid quality contracts
        
    Returns
    -------
    tuple[DataFrame, dict]
        (options chain, metadata)
    """
    snapshot_path = Path(snapshot_path)
    
    # If directory, look for cleaned file
    if snapshot_path.is_dir():
        parquet_files = list(snapshot_path.glob("*_cleaned.parquet"))
        if not parquet_files:
            parquet_files = list(snapshot_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {snapshot_path}")
        snapshot_path = parquet_files[0]
    
    df = pd.read_parquet(snapshot_path)
    
    # Load metadata
    meta_path = snapshot_path.parent / snapshot_path.name.replace(
        '_cleaned.parquet', '_metadata.json'
    ).replace('.parquet', '_metadata.json')
    
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    
    # Filter to valid only if requested
    if valid_only and 'quality' in df.columns:
        df = df[df['quality'] == 'valid'].copy()
    
    return df, metadata


class SnapshotManager:
    """
    Manage collection of market data snapshots.
    
    Example
    -------
    >>> manager = SnapshotManager('./data/snapshots')
    >>> snapshots = manager.list_snapshots()
    >>> df, meta = manager.load_latest('SPY')
    """
    
    def __init__(self, snapshot_dir: Path | str):
        """
        Initialize snapshot manager.
        
        Parameters
        ----------
        snapshot_dir : Path
            Base directory for snapshots
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    def list_snapshots(self, symbol: str | None = None) -> list[dict]:
        """
        List available snapshots.
        
        Returns
        -------
        list[dict]
            List of snapshot metadata dictionaries
        """
        snapshots = []
        
        for meta_file in self.snapshot_dir.glob("*_metadata.json"):
            with open(meta_file) as f:
                meta = json.load(f)
            
            if symbol is None or meta.get('symbol', '').upper() == symbol.upper():
                snapshots.append(meta)
        
        # Sort by observed_at descending
        snapshots.sort(key=lambda x: x.get('observed_at', ''), reverse=True)
        return snapshots
    
    def load_latest(
        self,
        symbol: str,
        valid_only: bool = True,
    ) -> tuple[pd.DataFrame, dict] | None:
        """
        Load the most recent snapshot for a symbol.
        
        Returns
        -------
        tuple[DataFrame, dict] or None
            (options chain, metadata) or None if no snapshots
        """
        snapshots = self.list_snapshots(symbol)
        if not snapshots:
            return None
        
        latest = snapshots[0]
        snapshot_id = latest['snapshot_id']
        
        return load_snapshot(
            self.snapshot_dir / f"{snapshot_id}_cleaned.parquet",
            valid_only=valid_only,
        )
    
    def save(
        self,
        df: pd.DataFrame,
        metadata: dict,
        snapshot_id: str | None = None,
    ) -> str:
        """
        Save a new snapshot.
        
        Returns
        -------
        str
            Snapshot ID
        """
        if snapshot_id is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            symbol = metadata.get('symbol', 'UNKNOWN')
            snapshot_id = f"{symbol}_{timestamp}"
        
        save_snapshot(df, metadata, self.snapshot_dir, snapshot_id)
        return snapshot_id
    
    def get_sample_snapshot_path(self) -> Path | None:
        """
        Get path to sample snapshot for offline testing.
        
        Returns
        -------
        Path or None
            Path to sample snapshot if exists
        """
        sample_files = list(self.snapshot_dir.glob("*_sample_*.parquet"))
        if sample_files:
            return sample_files[0]
        
        # Fall back to any snapshot
        all_files = list(self.snapshot_dir.glob("*.parquet"))
        if all_files:
            return all_files[0]
        
        return None
