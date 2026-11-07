"""Quote quality filters must read normalized column names."""

import pandas as pd

from volatilitylab.data.options_chain import QuoteQualityFilter


def test_open_interest_column_not_openInterest():
    df = pd.DataFrame(
        {
            "bid": [1.0],
            "ask": [1.1],
            "open_interest": [500],
            "volume": [20],
        }
    )
    out = QuoteQualityFilter().apply(df, spot=100.0)
    assert out["quality"].iloc[0] == "valid"
