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


def test_surface_grid_expiry_in_days():
    chain, spot = _tiny_chain()
    surface = VolatilitySurface.from_option_chain(chain, spot=spot, underlying="TEST")
    grid = surface.get_surface_grid(n_strikes=8, n_expiries=6)
    expiry_axis = np.array(grid["expiry"])
    assert expiry_axis.min() > 1.0
    assert expiry_axis.max() < 100.0
