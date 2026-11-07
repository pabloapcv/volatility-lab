"""Real yfinance snapshots should win over the synthetic offline demo."""

from pathlib import Path

import pandas as pd

from volatilitylab.data.snapshots import SnapshotManager, save_snapshot


def test_load_latest_prefers_yfinance(tmp_path: Path):
    manager = SnapshotManager(tmp_path)

    synth = pd.DataFrame({"strike": [100.0], "option_type": ["call"], "quality": ["valid"]})
    save_snapshot(
        synth,
        {
            "symbol": "SPY",
            "provider": "synthetic",
            "observed_at": "2026-11-07T12:00:00+00:00",
            "spot": 500.0,
            "expirations": ["2026-11-14"],
            "total_contracts": 1,
            "valid_contracts": 1,
        },
        tmp_path,
        snapshot_id="SPY_sample_offline",
    )
    real = pd.DataFrame({"strike": [101.0], "option_type": ["call"], "quality": ["valid"]})
    save_snapshot(
        real,
        {
            "symbol": "SPY",
            "provider": "yfinance",
            "observed_at": "2026-11-06T20:00:00+00:00",
            "spot": 670.0,
            "expirations": ["2026-11-09"],
            "total_contracts": 1,
            "valid_contracts": 1,
        },
        tmp_path,
        snapshot_id="SPY_yfinance_sample",
    )

    loaded = manager.load_latest("SPY")
    assert loaded is not None
    df, meta = loaded
    assert meta["snapshot_id"] == "SPY_yfinance_sample"
    assert meta["provider"] == "yfinance"
    assert float(df["strike"].iloc[0]) == 101.0
