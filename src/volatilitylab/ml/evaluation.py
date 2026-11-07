"""
Evaluation Metrics and Statistical Tests

Metrics:
- MAE, RMSE, QLIKE

Statistical Tests:
- Stationary block bootstrap confidence intervals
- Diebold-Mariano test with HAC variance and HLN correction
- Holm correction for multiple comparisons
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MetricResult:
    """Container for metric computation results."""

    mae: float
    rmse: float
    qlike: float
    n_samples: int


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval."""

    estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_bootstrap: int
    confidence_level: float


@dataclass
class DMTestResult:
    """Diebold-Mariano test result."""

    dm_statistic: float
    p_value: float
    mean_loss_diff: float
    model1_better: bool
    hac_variance: float
    hln_corrected: bool


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def compute_qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute QLIKE loss for variance forecasts.

    QLIKE = log(σ²_pred) + σ²_actual / σ²_pred

    This is the proper loss function for variance forecasts as it's
    robust to scale and handles heteroskedasticity correctly.
    """
    # Ensure positive predictions
    predicted = np.maximum(predicted, 1e-10)
    actual = np.maximum(actual, 1e-10)

    return np.mean(np.log(predicted) + actual / predicted)


def compute_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> MetricResult:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    actual : array-like
        Actual realized variance
    predicted : array-like
        Predicted variance

    Returns
    -------
    MetricResult
        MAE, RMSE, QLIKE metrics
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Remove NaN pairs
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[valid]
    predicted = predicted[valid]

    return MetricResult(
        mae=compute_mae(actual, predicted),
        rmse=compute_rmse(actual, predicted),
        qlike=compute_qlike(actual, predicted),
        n_samples=len(actual),
    )


def stationary_block_bootstrap(
    data: np.ndarray,
    statistic_fn: callable,
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Stationary block bootstrap for dependent time series.

    Parameters
    ----------
    data : np.ndarray
        Time series data
    statistic_fn : callable
        Function to compute the statistic
    n_bootstrap : int
        Number of bootstrap samples
    block_size : int, optional
        Average block size (defaults to n^(1/3))
    random_state : int
        Random seed

    Returns
    -------
    np.ndarray
        Bootstrap distribution of the statistic
    """
    rng = np.random.default_rng(random_state)
    n = len(data)

    if block_size is None:
        block_size = max(1, int(n ** (1 / 3)))

    # Geometric probability for block length
    p = 1 / block_size

    bootstrap_stats = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Generate bootstrap sample
        bootstrap_sample = []
        while len(bootstrap_sample) < n:
            # Random starting point
            start = rng.integers(0, n)

            # Random block length (geometric distribution)
            length = rng.geometric(p)

            # Add block (with wraparound)
            for i in range(length):
                if len(bootstrap_sample) >= n:
                    break
                bootstrap_sample.append(data[(start + i) % n])

        bootstrap_sample = np.array(bootstrap_sample[:n])
        bootstrap_stats[b] = statistic_fn(bootstrap_sample)

    return bootstrap_stats


def block_bootstrap_ci(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric: Literal["mae", "rmse", "qlike"] = "mae",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    block_size: int | None = None,
    random_state: int = 42,
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval for a metric.

    Uses stationary block bootstrap to account for time series dependence.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values
    metric : str
        Which metric ('mae', 'rmse', 'qlike')
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95)
    block_size : int, optional
        Average block size
    random_state : int
        Random seed

    Returns
    -------
    BootstrapCI
        Bootstrap confidence interval
    """
    # Compute loss/error series
    if metric == "mae":
        loss = np.abs(actual - predicted)
        statistic_fn = np.mean
    elif metric == "rmse":
        loss = (actual - predicted) ** 2

        def statistic_fn(x):
            return np.sqrt(np.mean(x))
    elif metric == "qlike":
        predicted = np.maximum(predicted, 1e-10)
        actual = np.maximum(actual, 1e-10)
        loss = np.log(predicted) + actual / predicted
        statistic_fn = np.mean
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Point estimate
    point_estimate = statistic_fn(loss)

    # Bootstrap distribution
    bootstrap_dist = stationary_block_bootstrap(
        loss, statistic_fn, n_bootstrap, block_size, random_state
    )

    # Confidence interval (percentile method)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_dist, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_dist, 100 * (1 - alpha / 2))

    return BootstrapCI(
        estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=np.std(bootstrap_dist),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def newey_west_variance(
    x: np.ndarray,
    max_lag: int | None = None,
) -> float:
    """
    Compute Newey-West HAC variance estimator.

    Parameters
    ----------
    x : np.ndarray
        Time series
    max_lag : int, optional
        Maximum lag (defaults to floor(n^(1/3)))

    Returns
    -------
    float
        HAC variance estimate
    """
    n = len(x)

    if max_lag is None:
        max_lag = int(np.floor(n ** (1 / 3)))

    # Sample variance
    x_demean = x - np.mean(x)
    gamma_0 = np.sum(x_demean**2) / n

    # Autocovariances with Bartlett kernel weights
    nw_var = gamma_0
    for j in range(1, max_lag + 1):
        weight = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.sum(x_demean[j:] * x_demean[:-j]) / n
        nw_var += 2 * weight * gamma_j

    return nw_var / n


def diebold_mariano_test(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    loss_fn: Literal["se", "ae", "qlike"] = "qlike",
    horizon: int = 1,
    hln_correction: bool = True,
) -> DMTestResult:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    pred1 : np.ndarray
        Predictions from model 1
    pred2 : np.ndarray
        Predictions from model 2
    loss_fn : str
        Loss function ('se', 'ae', 'qlike')
    horizon : int
        Forecast horizon (for HAC correction)
    hln_correction : bool
        Apply Harvey-Leybourne-Newbold small-sample correction

    Returns
    -------
    DMTestResult
        Test statistic, p-value, and interpretation
    """
    n = len(actual)

    # Compute loss differential
    if loss_fn == "se":
        loss1 = (actual - pred1) ** 2
        loss2 = (actual - pred2) ** 2
    elif loss_fn == "ae":
        loss1 = np.abs(actual - pred1)
        loss2 = np.abs(actual - pred2)
    elif loss_fn == "qlike":
        pred1 = np.maximum(pred1, 1e-10)
        pred2 = np.maximum(pred2, 1e-10)
        actual = np.maximum(actual, 1e-10)
        loss1 = np.log(pred1) + actual / pred1
        loss2 = np.log(pred2) + actual / pred2
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    d = loss1 - loss2
    d_mean = np.mean(d)

    # HAC variance with Newey-West
    max_lag = max(1, horizon - 1)
    hac_var = newey_west_variance(d, max_lag)

    if hac_var <= 0:
        hac_var = np.var(d) / n

    # DM statistic
    dm_stat = d_mean / np.sqrt(hac_var)

    # HLN small-sample correction
    if hln_correction:
        correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        dm_stat = dm_stat * correction

    # Two-sided p-value (normal approximation)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return DMTestResult(
        dm_statistic=dm_stat,
        p_value=p_value,
        mean_loss_diff=d_mean,
        model1_better=d_mean < 0,  # Negative means model 1 has lower loss
        hac_variance=hac_var,
        hln_corrected=hln_correction,
    )


def holm_correction(p_values: list[float], alpha: float = 0.05) -> list[tuple[int, float, bool]]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : list[float]
        Original p-values
    alpha : float
        Family-wise error rate

    Returns
    -------
    list[tuple[int, float, bool]]
        List of (original_index, adjusted_p, significant)
    """
    n = len(p_values)

    # Sort p-values with original indices
    indexed_pvals = sorted(enumerate(p_values), key=lambda x: x[1])

    results = []
    rejected = True

    for rank, (orig_idx, pval) in enumerate(indexed_pvals):
        # Holm threshold
        threshold = alpha / (n - rank)

        # Once we fail to reject, all subsequent are not rejected
        if not rejected or pval > threshold:
            rejected = False

        # Adjusted p-value (capped at 1)
        adjusted_p = min(1.0, pval * (n - rank))

        results.append((orig_idx, adjusted_p, rejected and pval <= threshold))

    # Sort back to original order
    results.sort(key=lambda x: x[0])

    return results


def compare_models_dm(
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    baseline_model: str = "hv",
    loss_fn: str = "qlike",
    horizon: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare multiple models using DM tests with Holm correction.

    Parameters
    ----------
    actual : np.ndarray
        Actual values
    predictions : dict[str, np.ndarray]
        Model name → predictions
    baseline_model : str
        Model to compare against
    loss_fn : str
        Loss function for comparison
    horizon : int
        Forecast horizon
    alpha : float
        Significance level

    Returns
    -------
    pd.DataFrame
        Comparison results with DM stats and corrected p-values
    """
    if baseline_model not in predictions:
        raise ValueError(f"Baseline model '{baseline_model}' not in predictions")

    baseline_preds = predictions[baseline_model]
    other_models = [m for m in predictions.keys() if m != baseline_model]

    results = []
    p_values = []

    for model in other_models:
        dm_result = diebold_mariano_test(
            actual, predictions[model], baseline_preds, loss_fn=loss_fn, horizon=horizon
        )

        results.append(
            {
                "model": model,
                "dm_statistic": dm_result.dm_statistic,
                "p_value": dm_result.p_value,
                "mean_loss_diff": dm_result.mean_loss_diff,
                "model_better": dm_result.model1_better,
            }
        )
        p_values.append(dm_result.p_value)

    # Apply Holm correction
    holm_results = holm_correction(p_values, alpha)

    for i, (_, adj_p, significant) in enumerate(holm_results):
        results[i]["adjusted_p_value"] = adj_p
        results[i]["significant"] = significant

    return pd.DataFrame(results)
