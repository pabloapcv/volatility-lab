"""
Walk-Forward Validation Framework

Implements expanding-window walk-forward validation with:
- Purge/embargo gap for overlapping targets
- Preprocessing fit only on training windows
- Identical test timestamps across all models
"""

from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class ValidationConfig:
    """Configuration for walk-forward validation."""

    initial_train_size: int = 252 * 2  # 2 years
    test_size: int = 21  # 1 month
    step_size: int = 21  # 1 month steps
    purge_gap: int = 5  # Purge gap for 5-day horizon
    embargo_gap: int = 20  # Embargo for 20-day horizon
    final_test_size: int = 252  # 1 year held-out final test


@dataclass
class ValidationFold:
    """Single fold of walk-forward validation."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class ValidationResult:
    """Results from a single fold."""

    fold_id: int
    predictions: pd.Series
    actuals: pd.Series
    model_params: dict


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation results."""

    model_name: str
    horizon: int
    folds: list[ValidationResult]
    all_predictions: pd.Series
    all_actuals: pd.Series
    test_dates: pd.DatetimeIndex
    config: ValidationConfig
    metadata: dict = field(default_factory=dict)


class WalkForwardValidator:
    """
    Walk-forward cross-validation with purge/embargo.

    Implements expanding-window validation where:
    1. Training window expands over time
    2. A purge gap removes data points that would leak target info
    3. An embargo gap ensures test predictions don't overlap with training targets
    4. An untouched final test period is held out

    Example
    -------
    >>> config = ValidationConfig(initial_train_size=500, test_size=21)
    >>> validator = WalkForwardValidator(config)
    >>> for fold in validator.split(features, target):
    ...     model.fit(X_train, y_train)
    ...     preds = model.predict(X_test)
    """

    def __init__(self, config: ValidationConfig | None = None):
        """
        Initialize validator.

        Parameters
        ----------
        config : ValidationConfig, optional
            Validation configuration
        """
        self.config = config or ValidationConfig()

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int = 5,
    ) -> Generator[ValidationFold, None, None]:
        """
        Generate walk-forward validation folds.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with DatetimeIndex
        y : pd.Series
            Target series
        horizon : int
            Forecast horizon (determines purge/embargo)

        Yields
        ------
        ValidationFold
            Train/test split for each fold
        """
        n = len(X)
        dates = X.index

        # Reserve final test period
        final_test_start = n - self.config.final_test_size

        # Determine gap based on horizon
        gap = max(self.config.purge_gap, horizon + 1)

        fold_id = 0
        train_end_idx = self.config.initial_train_size

        while train_end_idx + gap + self.config.test_size <= final_test_start:
            # Training indices
            train_idx = np.arange(0, train_end_idx)

            # Test indices (after gap)
            test_start_idx = train_end_idx + gap
            test_end_idx = min(test_start_idx + self.config.test_size, final_test_start)
            test_idx = np.arange(test_start_idx, test_end_idx)

            yield ValidationFold(
                fold_id=fold_id,
                train_start=dates[0],
                train_end=dates[train_end_idx - 1],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx - 1],
                train_idx=train_idx,
                test_idx=test_idx,
            )

            fold_id += 1
            train_end_idx += self.config.step_size

    def get_final_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the held-out final test period.

        This is the untouched test set for final evaluation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (train_idx, test_idx) for final evaluation
        """
        n = len(X)
        gap = max(self.config.purge_gap, horizon + 1)

        final_test_start = n - self.config.final_test_size
        train_end = final_test_start - gap

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(final_test_start, n)

        return train_idx, test_idx

    def validate_no_leakage(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int,
    ) -> dict:
        """
        Verify there's no data leakage in the validation setup.

        Returns
        -------
        dict
            Leakage test results
        """
        results = {
            "passed": True,
            "issues": [],
        }

        # Check that target is forward-looking
        # (can't directly verify without knowing target construction)

        # Check fold separation
        folds = list(self.split(X, y, horizon))

        for i, fold in enumerate(folds):
            # Verify gap between train and test
            gap = fold.test_idx[0] - fold.train_idx[-1] - 1
            min_gap = horizon + 1

            if gap < min_gap:
                results["passed"] = False
                results["issues"].append(f"Fold {i}: gap {gap} < min {min_gap}")

        # Check final test is truly held out
        train_idx, test_idx = self.get_final_test_split(X, y, horizon)

        if np.any(np.isin(test_idx, np.concatenate([f.train_idx for f in folds]))):
            results["passed"] = False
            results["issues"].append("Final test overlaps with training data")

        if np.any(np.isin(test_idx, np.concatenate([f.test_idx for f in folds]))):
            results["passed"] = False
            results["issues"].append("Final test overlaps with validation test data")

        return results


def check_time_split_validity(
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> dict:
    """
    Check that time split is valid (no future data in training).

    Returns
    -------
    dict
        Validation results
    """
    results = {
        "valid": True,
        "train_range": (train_dates.min(), train_dates.max()),
        "test_range": (test_dates.min(), test_dates.max()),
        "issues": [],
    }

    if train_dates.max() >= test_dates.min():
        results["valid"] = False
        results["issues"].append(
            f"Training end ({train_dates.max()}) >= Test start ({test_dates.min()})"
        )

    return results


def create_regime_splits(
    features: pd.DataFrame,
    regime_col: str = "high_vol_regime",
    threshold: float = 0.5,
) -> tuple[pd.Index, pd.Index]:
    """
    Split data into low-vol and high-vol regimes for H3 testing.

    Returns
    -------
    tuple[pd.Index, pd.Index]
        (low_vol_idx, high_vol_idx)
    """
    if regime_col in features.columns:
        high_vol = features[regime_col] > threshold
    else:
        # Use 21-day RV as proxy
        rv_col = "rv_21d" if "rv_21d" in features.columns else features.columns[0]
        median_rv = features[rv_col].median()
        high_vol = features[rv_col] > median_rv

    return features.index[~high_vol], features.index[high_vol]
