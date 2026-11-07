"""
Feature Engineering for Volatility Forecasting

Point-in-time features using only information available at prediction time:
- Lagged returns
- Realized volatility at multiple windows
- Volatility-of-volatility
- Volume features
- VIX (lagged to ensure availability)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class VolatilityFeatures:
    """Container for computed features."""
    features: pd.DataFrame
    target_5d: pd.Series
    target_20d: pd.Series
    feature_names: list[str]
    metadata: dict


def compute_realized_variance(
    returns: pd.Series,
    window: int,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute realized variance over a rolling window.
    
    Parameters
    ----------
    returns : pd.Series
        Log returns series
    window : int
        Rolling window size (trading days)
    annualize : bool
        Whether to annualize (multiply by 252)
        
    Returns
    -------
    pd.Series
        Realized variance series
    """
    rv = returns.rolling(window=window).var()
    if annualize:
        rv = rv * 252
    return rv


def compute_realized_vol(
    returns: pd.Series,
    window: int,
    annualize: bool = True,
) -> pd.Series:
    """Compute realized volatility (sqrt of variance)."""
    rv = compute_realized_variance(returns, window, annualize)
    return np.sqrt(rv)


class FeatureEngine:
    """
    Feature engineering for volatility forecasting.
    
    All features are strictly point-in-time to prevent lookahead bias.
    
    Example
    -------
    >>> engine = FeatureEngine(horizons=[5, 20])
    >>> features = engine.compute_features(price_df, vix_df)
    """
    
    RV_WINDOWS = [5, 10, 21, 63]  # 1 week, 2 weeks, 1 month, 3 months
    RETURN_LAGS = [1, 2, 3, 5, 10]
    
    def __init__(
        self,
        horizons: list[int] = [5, 20],
        include_vix: bool = True,
        include_volume: bool = True,
    ):
        """
        Initialize feature engine.
        
        Parameters
        ----------
        horizons : list[int]
            Forecast horizons in trading days
        include_vix : bool
            Whether to include VIX features (requires VIX data)
        include_volume : bool
            Whether to include volume features
        """
        self.horizons = horizons
        self.include_vix = include_vix
        self.include_volume = include_volume
    
    def compute_features(
        self,
        prices: pd.DataFrame,
        vix: pd.Series | None = None,
    ) -> VolatilityFeatures:
        """
        Compute all features for volatility forecasting.
        
        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame with 'close' and optionally 'volume' columns
        vix : pd.Series, optional
            VIX index series (lagged by 1 day for point-in-time)
            
        Returns
        -------
        VolatilityFeatures
            Features, targets, and metadata
        """
        # Ensure we have required columns
        if 'close' not in prices.columns:
            if 'Close' in prices.columns:
                prices = prices.rename(columns={'Close': 'close', 'Volume': 'volume'})
            else:
                raise ValueError("prices must have 'close' column")
        
        # Compute log returns
        log_returns = np.log(prices['close'] / prices['close'].shift(1))
        
        features = pd.DataFrame(index=prices.index)
        feature_names = []
        
        # 1. Lagged returns
        for lag in self.RETURN_LAGS:
            col = f'ret_lag_{lag}'
            features[col] = log_returns.shift(lag)
            feature_names.append(col)
        
        # 2. Realized volatility at multiple windows
        for window in self.RV_WINDOWS:
            col = f'rv_{window}d'
            features[col] = compute_realized_vol(log_returns, window)
            feature_names.append(col)
        
        # 3. Realized variance (target-related)
        for window in self.RV_WINDOWS:
            col = f'rvar_{window}d'
            features[col] = compute_realized_variance(log_returns, window)
            feature_names.append(col)
        
        # 4. Volatility-of-volatility
        rv_21 = compute_realized_vol(log_returns, 21)
        features['vol_of_vol'] = rv_21.rolling(window=21).std()
        feature_names.append('vol_of_vol')
        
        # 5. Return momentum
        features['ret_5d'] = log_returns.rolling(5).sum()
        features['ret_21d'] = log_returns.rolling(21).sum()
        feature_names.extend(['ret_5d', 'ret_21d'])
        
        # 6. Absolute return features
        abs_ret = log_returns.abs()
        features['abs_ret_5d'] = abs_ret.rolling(5).mean()
        features['abs_ret_21d'] = abs_ret.rolling(21).mean()
        feature_names.extend(['abs_ret_5d', 'abs_ret_21d'])
        
        # 7. High-low range volatility proxy (if available)
        if 'high' in prices.columns and 'low' in prices.columns:
            hl_range = np.log(prices['high'] / prices['low'])
            features['hl_vol_5d'] = hl_range.rolling(5).mean()
            features['hl_vol_21d'] = hl_range.rolling(21).mean()
            feature_names.extend(['hl_vol_5d', 'hl_vol_21d'])
        
        # 8. Volume features
        if self.include_volume and 'volume' in prices.columns:
            vol = prices['volume']
            vol_ma = vol.rolling(21).mean()
            features['volume_ratio'] = vol / vol_ma
            features['volume_std'] = vol.rolling(21).std() / vol_ma
            feature_names.extend(['volume_ratio', 'volume_std'])
        
        # 9. VIX features (lagged by 1 for point-in-time)
        if self.include_vix and vix is not None:
            # Lag VIX by 1 day to ensure point-in-time
            vix_aligned = vix.reindex(prices.index).shift(1)
            features['vix'] = vix_aligned / 100  # Scale to decimal
            features['vix_lag5'] = vix_aligned.shift(5) / 100
            features['vix_change'] = (vix_aligned - vix_aligned.shift(5)) / 100
            feature_names.extend(['vix', 'vix_lag5', 'vix_change'])
        
        # 10. Market stress indicator
        if 'vix' in features.columns:
            features['high_vol_regime'] = (features['vix'] > 0.20).astype(float)
            feature_names.append('high_vol_regime')
        else:
            # Use realized vol as proxy
            rv_21 = features.get('rv_21d', compute_realized_vol(log_returns, 21))
            features['high_vol_regime'] = (rv_21 > rv_21.quantile(0.75)).astype(float)
            feature_names.append('high_vol_regime')
        
        # Compute forward realized variance targets
        target_5d = self._compute_forward_rv(log_returns, 5)
        target_20d = self._compute_forward_rv(log_returns, 20)
        
        # Remove NaN rows (need sufficient history)
        min_history = max(self.RV_WINDOWS) + max(self.RETURN_LAGS) + 1
        valid_idx = ~(features.isna().any(axis=1) | target_5d.isna() | target_20d.isna())
        
        features = features.loc[valid_idx]
        target_5d = target_5d.loc[valid_idx]
        target_20d = target_20d.loc[valid_idx]
        
        metadata = {
            'n_samples': len(features),
            'n_features': len(feature_names),
            'date_range': (features.index.min(), features.index.max()),
            'horizons': self.horizons,
            'rv_windows': self.RV_WINDOWS,
        }
        
        return VolatilityFeatures(
            features=features,
            target_5d=target_5d,
            target_20d=target_20d,
            feature_names=feature_names,
            metadata=metadata,
        )
    
    def _compute_forward_rv(
        self,
        returns: pd.Series,
        horizon: int,
    ) -> pd.Series:
        """Compute forward-looking realized variance (target)."""
        # Shift returns backward to get future values
        # sum of squared returns over next `horizon` days, annualized
        forward_sq_ret = returns.shift(-horizon).rolling(window=horizon).apply(
            lambda x: (x ** 2).sum() * 252 / horizon,
            raw=True
        )
        # Shift back to align with prediction time
        return forward_sq_ret.shift(horizon)


def load_price_data(
    symbol: str,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load historical price data for a symbol.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str, optional
        End date (defaults to today)
        
    Returns
    -------
    pd.DataFrame
        OHLCV data
    """
    import yfinance as yf
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    # Standardize column names
    df.columns = df.columns.str.lower()
    
    # Remove timezone info for consistent indexing
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    return df


def load_vix_data(
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.Series:
    """
    Load VIX index data.
    
    Returns
    -------
    pd.Series
        VIX closing values
    """
    import yfinance as yf
    
    vix = yf.Ticker("^VIX")
    df = vix.history(start=start_date, end=end_date)
    
    result = df['Close'].rename('VIX')
    
    # Remove timezone info for consistent indexing
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    
    return result
