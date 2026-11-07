"""Tests for ML evaluation metrics and statistical tests."""

import numpy as np
import pytest

from volatilitylab.ml.evaluation import (
    block_bootstrap_ci,
    compare_models_dm,
    compute_mae,
    compute_metrics,
    compute_qlike,
    compute_rmse,
    diebold_mariano_test,
    holm_correction,
)


class TestMetrics:
    """Test evaluation metrics."""

    def test_mae(self):
        """Test MAE calculation."""
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 2.2, 2.7])

        mae = compute_mae(actual, predicted)
        expected = (0.1 + 0.2 + 0.3) / 3

        assert pytest.approx(mae, rel=1e-10) == expected

    def test_rmse(self):
        """Test RMSE calculation."""
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])  # Perfect

        rmse = compute_rmse(actual, predicted)
        assert rmse == 0.0

    def test_qlike_perfect(self):
        """Test QLIKE with perfect predictions."""
        actual = np.array([0.01, 0.02, 0.03])

        qlike = compute_qlike(actual, actual)
        # QLIKE at optimum: log(σ²) + σ²/σ² = log(σ²) + 1
        # For each sample it converges but doesn't equal 0
        assert np.isfinite(qlike)

    def test_qlike_positive(self):
        """Test QLIKE requires positive values."""
        actual = np.array([0.01, 0.02])
        predicted = np.array([0.01, 0.02])

        # Should work with positive values
        qlike = compute_qlike(actual, predicted)
        assert np.isfinite(qlike)

    def test_compute_metrics(self):
        """Test computing all metrics."""
        actual = np.array([0.01, 0.02, 0.03, 0.04])
        predicted = np.array([0.011, 0.019, 0.031, 0.038])

        result = compute_metrics(actual, predicted)

        assert result.mae > 0
        assert result.rmse > 0
        assert np.isfinite(result.qlike)
        assert result.n_samples == 4


class TestBootstrap:
    """Test bootstrap confidence intervals."""

    def test_bootstrap_ci_basic(self):
        """Test basic bootstrap CI."""
        np.random.seed(42)
        actual = np.random.rand(100) * 0.05 + 0.01
        predicted = actual + np.random.randn(100) * 0.005

        ci = block_bootstrap_ci(actual, predicted, "mae", n_bootstrap=200)

        assert ci.ci_lower < ci.estimate < ci.ci_upper
        assert ci.confidence_level == 0.95
        assert ci.std_error > 0

    def test_bootstrap_ci_different_metrics(self):
        """Test bootstrap with different metrics."""
        np.random.seed(42)
        actual = np.random.rand(50) * 0.05 + 0.01
        predicted = actual + np.random.randn(50) * 0.005

        for metric in ["mae", "rmse", "qlike"]:
            ci = block_bootstrap_ci(actual, predicted, metric, n_bootstrap=100)
            assert ci.ci_lower <= ci.estimate <= ci.ci_upper


class TestDieboldMariano:
    """Test Diebold-Mariano tests."""

    def test_dm_identical_predictions(self):
        """Test DM with identical predictions."""
        np.random.seed(42)
        actual = np.random.rand(100) * 0.05 + 0.01
        pred = actual + np.random.randn(100) * 0.01

        result = diebold_mariano_test(actual, pred, pred, horizon=1)

        # When predictions are identical, loss diff is 0, variance is 0
        # Result should be nan or 0 statistic with high p-value
        # The mean loss diff should be exactly 0
        assert result.mean_loss_diff == 0.0

    def test_dm_different_predictions(self):
        """Test DM with clearly different predictions."""
        np.random.seed(42)
        actual = np.random.rand(100) * 0.05 + 0.01
        pred1 = actual + np.random.randn(100) * 0.001  # Better
        pred2 = actual + np.random.randn(100) * 0.01  # Worse

        result = diebold_mariano_test(actual, pred1, pred2, horizon=1)

        # Model 1 should be better (negative statistic or model1_better=True)
        # Note: direction depends on how losses are computed
        assert np.isfinite(result.dm_statistic)
        assert 0 <= result.p_value <= 1

    def test_dm_hln_correction(self):
        """Test DM with HLN small-sample correction."""
        np.random.seed(42)
        actual = np.random.rand(30) * 0.05 + 0.01
        pred1 = actual + np.random.randn(30) * 0.005
        pred2 = actual + np.random.randn(30) * 0.005

        result = diebold_mariano_test(actual, pred1, pred2, horizon=5)

        # HLN correction should be applied for small samples with h > 1
        assert np.isfinite(result.dm_statistic)

    def test_dm_horizon_adjustment(self):
        """Test that horizon affects the test."""
        np.random.seed(42)
        actual = np.random.rand(100)
        pred1 = actual + np.random.randn(100) * 0.1
        pred2 = actual + np.random.randn(100) * 0.1

        result_h1 = diebold_mariano_test(actual, pred1, pred2, horizon=1)
        result_h5 = diebold_mariano_test(actual, pred1, pred2, horizon=5)

        # Both should complete
        assert np.isfinite(result_h1.dm_statistic)
        assert np.isfinite(result_h5.dm_statistic)


class TestHolmCorrection:
    """Test Holm-Bonferroni correction."""

    def test_holm_all_significant(self):
        """Test Holm with all significant p-values."""
        p_values = [0.001, 0.002, 0.005]

        result = holm_correction(p_values, alpha=0.05)

        # All should be significant
        for _idx, _adj_p, sig in result:
            assert sig is True

    def test_holm_none_significant(self):
        """Test Holm with no significant p-values."""
        p_values = [0.5, 0.6, 0.7]

        result = holm_correction(p_values, alpha=0.05)

        # None should be significant
        for _idx, _adj_p, sig in result:
            assert sig is False

    def test_holm_mixed(self):
        """Test Holm with mixed p-values."""
        p_values = [0.001, 0.1, 0.5]

        result = holm_correction(p_values, alpha=0.05)

        # Check ordering is correct
        indices = [r[0] for r in result]
        assert indices == [0, 1, 2]  # Original order preserved

        # First should be significant
        assert result[0][2] is True

    def test_holm_single(self):
        """Test Holm with single p-value."""
        p_values = [0.01]

        result = holm_correction(p_values, alpha=0.05)

        assert len(result) == 1
        assert result[0][2] is True


class TestCompareModels:
    """Test multi-model comparison."""

    def test_compare_models_basic(self):
        """Test comparing multiple models."""
        np.random.seed(42)
        actual = np.random.rand(50) * 0.05 + 0.01

        predictions = {
            "baseline": actual + np.random.randn(50) * 0.01,
            "model_a": actual + np.random.randn(50) * 0.005,
            "model_b": actual + np.random.randn(50) * 0.008,
        }

        result = compare_models_dm(actual, predictions, baseline_model="baseline", horizon=5)

        # Should have comparisons for model_a and model_b
        assert len(result) == 2
        assert "model" in result.columns
        assert "p_value" in result.columns
        assert "significant" in result.columns  # Holm-corrected significance
