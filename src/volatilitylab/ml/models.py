"""
Volatility Forecasting Models

Models:
- Historical Volatility (rolling window)
- EWMA (exponentially weighted)
- GARCH(1,1) via arch library
- XGBoost regression
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Container for forecast results."""
    predictions: pd.Series
    model_name: str
    params: dict
    fit_time_ms: float


class VolatilityModel(ABC):
    """Base class for volatility models."""
    
    @abstractmethod
    def fit(self, returns: pd.Series, **kwargs) -> None:
        """Fit the model to historical returns."""
        pass
    
    @abstractmethod
    def predict(self, returns: pd.Series, horizon: int) -> float:
        """Predict forward realized variance."""
        pass
    
    @abstractmethod
    def get_params(self) -> dict:
        """Get model parameters."""
        pass


class HistoricalVolModel(VolatilityModel):
    """
    Historical volatility baseline.
    
    Predicts forward variance as the rolling historical variance.
    """
    
    def __init__(self, window: int = 21):
        """
        Initialize HV model.
        
        Parameters
        ----------
        window : int
            Rolling window for variance calculation
        """
        self.window = window
        self.last_rv = None
    
    def fit(self, returns: pd.Series, **kwargs) -> None:
        """Fit by computing historical variance."""
        if len(returns) < self.window:
            raise ValueError(f"Need at least {self.window} observations")
        
        # Annualized variance
        self.last_rv = returns.iloc[-self.window:].var() * 252
    
    def predict(self, returns: pd.Series, horizon: int) -> float:
        """
        Predict forward variance.
        
        Historical vol is horizon-independent (same forecast for all horizons).
        """
        if self.last_rv is None:
            self.fit(returns)
        return self.last_rv
    
    def get_params(self) -> dict:
        return {'window': self.window, 'last_rv': self.last_rv}


class EWMAModel(VolatilityModel):
    """
    Exponentially Weighted Moving Average variance model.
    
    σ²_t = λ * σ²_{t-1} + (1 - λ) * r²_{t-1}
    """
    
    def __init__(self, decay: float = 0.94):
        """
        Initialize EWMA model.
        
        Parameters
        ----------
        decay : float
            Decay factor (lambda), typically 0.94 for RiskMetrics
        """
        self.decay = decay
        self.current_var = None
    
    def fit(self, returns: pd.Series, **kwargs) -> None:
        """Fit by computing EWMA variance recursively."""
        if len(returns) < 2:
            raise ValueError("Need at least 2 observations")
        
        # Initialize with sample variance
        var = returns.iloc[:21].var() * 252 if len(returns) >= 21 else returns.var() * 252
        
        # EWMA recursion
        for r in returns.iloc[1:]:
            var = self.decay * var + (1 - self.decay) * (r ** 2) * 252
        
        self.current_var = var
    
    def predict(self, returns: pd.Series, horizon: int) -> float:
        """Predict forward variance."""
        if self.current_var is None:
            self.fit(returns)
        return self.current_var
    
    def get_params(self) -> dict:
        return {'decay': self.decay, 'current_var': self.current_var}


class GARCHModel(VolatilityModel):
    """
    GARCH(1,1) model via arch library.
    
    σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH model.
        
        Parameters
        ----------
        p : int
            ARCH order
        q : int
            GARCH order
        """
        self.p = p
        self.q = q
        self.model = None
        self.result = None
    
    def fit(self, returns: pd.Series, **kwargs) -> None:
        """Fit GARCH model."""
        from arch import arch_model
        
        # Scale returns to percentage for numerical stability
        returns_pct = returns * 100
        
        self.model = arch_model(
            returns_pct,
            vol='Garch',
            p=self.p,
            q=self.q,
            rescale=False,
        )
        
        self.result = self.model.fit(disp='off', show_warning=False)
    
    def predict(self, returns: pd.Series, horizon: int) -> float:
        """
        Predict forward variance.
        
        Uses GARCH forecast, averaged over horizon.
        """
        if self.result is None:
            self.fit(returns)
        
        # Forecast conditional variance
        forecasts = self.result.forecast(horizon=horizon, reindex=False)
        
        # Get mean variance over horizon, convert back from pct^2 to decimal
        mean_var = forecasts.variance.iloc[-1].mean() / 10000
        
        # Annualize
        return mean_var * 252
    
    def get_params(self) -> dict:
        if self.result is None:
            return {'p': self.p, 'q': self.q, 'fitted': False}
        
        return {
            'p': self.p,
            'q': self.q,
            'omega': float(self.result.params.get('omega', 0)),
            'alpha': float(self.result.params.get('alpha[1]', 0)),
            'beta': float(self.result.params.get('beta[1]', 0)),
            'fitted': True,
        }


class HARRVModel(VolatilityModel):
    """
    Heterogeneous Autoregressive model for Realized Volatility (HAR-RV).
    
    Standard benchmark in volatility forecasting literature.
    Uses daily, weekly (5d), and monthly (22d) RV components:
    
    RV_{t+h} = β₀ + β_d * RV_d + β_w * RV_w + β_m * RV_m + ε
    
    Reference: Corsi (2009) "A Simple Approximate Long-Memory Model 
               of Realized Volatility"
    """
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self._fitted = False
        self._feature_spec = []  # (column_name, transform_fn)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Fit HAR-RV model using OLS.
        
        Parameters
        ----------
        X : pd.DataFrame
            Must contain ret_lag_1, rvar_5d, and rvar_21d
        y : pd.Series
            Target forward RV
        """
        from sklearn.linear_model import LinearRegression
        
        features = []
        feature_names = []
        self._feature_spec = []
        
        # Daily RV: squared daily return (annualized)
        # ret_lag_1 is the return from t-1 to t
        if 'ret_lag_1' in X.columns:
            daily_rv = (X['ret_lag_1'].values ** 2) * 252
            features.append(daily_rv)
            feature_names.append('rv_d')
            self._feature_spec.append(('ret_lag_1', lambda x: (x ** 2) * 252))
        
        # Weekly RV: 5-day realized variance (already annualized in features)
        if 'rvar_5d' in X.columns:
            features.append(X['rvar_5d'].values)
            feature_names.append('rv_w')
            self._feature_spec.append(('rvar_5d', lambda x: x))
        
        # Monthly RV: 21-day realized variance
        if 'rvar_21d' in X.columns:
            features.append(X['rvar_21d'].values)
            feature_names.append('rv_m')
            self._feature_spec.append(('rvar_21d', lambda x: x))
        
        if len(features) < 3:
            raise ValueError(f"HAR-RV needs daily, weekly, monthly RV. Found: {feature_names}")
        
        X_har = np.column_stack(features)
        
        # Fit OLS
        model = LinearRegression()
        model.fit(X_har, y)
        
        self.coefficients = dict(zip(feature_names, model.coef_))
        self.intercept = model.intercept_
        self._model = model
        self._fitted = True
    
    def predict(self, X: pd.DataFrame, horizon: int = None) -> np.ndarray:
        """Predict forward RV using HAR model."""
        if not self._fitted:
            raise ValueError("Model not fitted")
        
        features = []
        for col_name, transform_fn in self._feature_spec:
            features.append(transform_fn(X[col_name].values))
        
        X_har = np.column_stack(features)
        return self._model.predict(X_har)
    
    def get_params(self) -> dict:
        return {
            'intercept': self.intercept,
            'coefficients': self.coefficients,
            'fitted': self._fitted,
        }


class XGBoostVolModel(VolatilityModel):
    """
    XGBoost regression model for volatility forecasting.
    
    Uses point-in-time features to predict forward realized variance.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize XGBoost model.
        
        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_names = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> None:
        """
        Fit XGBoost model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target (forward realized variance)
        """
        import xgboost as xgb
        
        self.feature_names = list(X.columns)
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective='reg:squarederror',
            n_jobs=-1,
        )
        
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame, horizon: int = None) -> np.ndarray:
        """
        Predict forward variance.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        horizon : int, optional
            Ignored (model is horizon-specific)
            
        Returns
        -------
        np.ndarray
            Predicted variances
        """
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X)
    
    def get_params(self) -> dict:
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
        }
        
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            ))
            params['feature_importance'] = importance
        
        return params
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if self.model is None:
            return pd.Series(dtype=float)
        
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names,
        ).sort_values(ascending=False)


def create_model(
    model_type: str,
    **kwargs,
) -> VolatilityModel:
    """
    Factory function to create volatility models.
    
    Parameters
    ----------
    model_type : str
        One of 'hv', 'ewma', 'garch', 'xgboost'
    **kwargs
        Model-specific parameters
        
    Returns
    -------
    VolatilityModel
        Instantiated model
    """
    models = {
        'hv': HistoricalVolModel,
        'ewma': EWMAModel,
        'garch': GARCHModel,
        'har': HARRVModel,
        'xgboost': XGBoostVolModel,
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type.lower()](**kwargs)
