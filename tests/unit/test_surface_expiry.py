"""Surface expiry coordinates must be calendar days, not years."""

import numpy as np
import pandas as pd

from volatilitylab.pricing.black_scholes import OptionType, black_scholes_price
from volatilitylab.volatility.surface import VolatilitySurface


def _tiny_chain():
    rows = []
    spot = 100.0
    vol = 0.20
    for tte_years, strike in [
        (7 / 365, 100.0),
        (21 / 365, 100.0),
        (42 / 365, 105.0),
        (42 / 365, 95.0),
    ]:
        mid = float(black_scholes_price(spot, strike, tte_years, 0.05, vol, OptionType.CALL, 0.0))
        rows.append(
            {
                "strike": strike,
                "option_type": "call",
                "time_to_expiry": tte_years,
                "mid_price": mid,
                "bid": mid * 0.98,
                "ask": mid * 1.02,
            }
        )
    return pd.DataFrame(rows), spot


def test_expiry_stored_in_days():
    chain, spot = _tiny_chain()
    surface = VolatilitySurface.from_option_chain(chain, spot=spot, underlying="TEST")
    expiries = sorted({p.expiry for p in surface.points})
    assert min(expiries) > 1.0, f"expiry looks like years: {expiries}"
    assert abs(min(expiries) - 7.0) < 0.05
    assert abs(max(expiries) - 42.0) < 0.05


def test_spy_committed_snapshot_point_count_stable():
    """Vectorized IV must keep the same observed-point set as the committed SPY chain."""
    from pathlib import Path

    from volatilitylab.data.snapshots import SnapshotManager

    result = SnapshotManager(Path("./data/snapshots")).load_latest("SPY")
    assert result is not None
    df, meta = result
    surface = VolatilitySurface.from_option_chain(
        df,
        spot=float(meta["spot"]),
        risk_free_rate=0.045,
        underlying="SPY",
        observed_at=str(meta.get("observed_at", "")),
    )
    assert surface.metadata.n_observed_points == 889
    spot = float(meta["spot"])
    near = [p for p in surface.points if abs(p.strike - spot) < 0.5 and abs(p.expiry - 1.0) < 0.05]
    assert near
    assert abs(near[0].iv - 0.3311064549402278) < 1e-6


def test_surface_grid_expiry_in_days():
    chain, spot = _tiny_chain()
    surface = VolatilitySurface.from_option_chain(chain, spot=spot, underlying="TEST")
    grid = surface.get_surface_grid(n_strikes=8, n_expiries=6)
    expiry_axis = np.array(grid["expiry"])
    assert expiry_axis.min() > 1.0
    assert expiry_axis.max() < 100.0
