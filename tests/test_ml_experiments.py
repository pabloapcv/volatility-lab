"""Tests for ML experiment tracking."""

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from volatilitylab.ml.experiments import (
    ExperimentTracker,
    ExperimentResult,
    ModelResult,
    compute_data_hash,
)


class TestDataHash:
    """Test data hashing."""
    
    def test_hash_deterministic(self):
        """Same data produces same hash."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
        h1 = compute_data_hash(df)
        h2 = compute_data_hash(df)
        assert h1 == h2
    
    def test_hash_different_data(self):
        """Different data produces different hash."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [1, 2, 4]})
        assert compute_data_hash(df1) != compute_data_hash(df2)


class TestExperimentTracker:
    """Test experiment tracking."""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ExperimentTracker(tmpdir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        })
    
    def test_start_experiment(self, tracker, sample_data):
        """Test starting an experiment."""
        exp_id = tracker.start_experiment(
            'SPY',
            {'param1': 1},
            seed=42,
            data=sample_data,
        )
        
        assert exp_id is not None
        assert 'SPY' in exp_id
    
    def test_log_model_result(self, tracker, sample_data):
        """Test logging model result."""
        exp_id = tracker.start_experiment('SPY', {}, 42, sample_data)
        
        result = ModelResult(
            model_name='hv_5d',
            model_type='historical_volatility',
            params={'window': 21},
            metrics_5d={'mae': 0.01, 'rmse': 0.02, 'qlike': 0.5},
            metrics_20d={},
        )
        
        tracker.log_model_result(exp_id, result)
        
        # Check it's stored
        assert len(tracker._active_experiments[exp_id]['models']) == 1
    
    def test_log_dm_comparison(self, tracker, sample_data):
        """Test logging DM comparison."""
        exp_id = tracker.start_experiment('SPY', {}, 42, sample_data)
        
        tracker.log_dm_comparison(exp_id, 5, {
            'model': 'xgboost',
            'dm_statistic': -2.5,
            'p_value': 0.01,
        })
        
        assert len(tracker._active_experiments[exp_id]['dm_comparisons_5d']) == 1
    
    def test_log_hypothesis_test(self, tracker, sample_data):
        """Test logging hypothesis test."""
        exp_id = tracker.start_experiment('SPY', {}, 42, sample_data)
        
        tracker.log_hypothesis_test(
            exp_id,
            'H1: ML beats baselines',
            'SUPPORTED',
            {'p_value': 0.01},
        )
        
        assert len(tracker._active_experiments[exp_id]['hypothesis_tests']) == 1
    
    def test_finish_experiment(self, tracker, sample_data):
        """Test finishing an experiment."""
        exp_id = tracker.start_experiment('SPY', {}, 42, sample_data)
        
        # Add some data
        tracker.log_model_result(exp_id, ModelResult(
            model_name='test', model_type='test', params={},
            metrics_5d={'mae': 0.01}, metrics_20d={},
        ))
        
        result = tracker.finish_experiment(exp_id)
        
        assert result.status == 'completed'
        assert result.symbol == 'SPY'
        assert len(result.models) == 1
        
        # Check persisted to disk
        exp_dir = tracker.base_dir / exp_id
        assert (exp_dir / 'result.json').exists()
    
    def test_finish_experiment_with_error(self, tracker, sample_data):
        """Test finishing experiment with error."""
        exp_id = tracker.start_experiment('SPY', {}, 42, sample_data)
        result = tracker.finish_experiment(exp_id, error="Test error")
        
        assert result.status == 'error'
        assert result.error == "Test error"
    
    def test_get_experiment(self, tracker, sample_data):
        """Test loading experiment from disk."""
        exp_id = tracker.start_experiment('SPY', {}, 42, sample_data)
        tracker.finish_experiment(exp_id)
        
        # Load it back
        loaded = tracker.get_experiment(exp_id)
        
        assert loaded is not None
        assert loaded.symbol == 'SPY'
        assert loaded.seed == 42
    
    def test_list_experiments(self, tracker, sample_data):
        """Test listing experiments."""
        # Create a few experiments
        exp1 = tracker.start_experiment('SPY', {}, 42, sample_data)
        tracker.finish_experiment(exp1)
        
        exp2 = tracker.start_experiment('QQQ', {}, 42, sample_data)
        tracker.finish_experiment(exp2)
        
        # List all
        all_exps = tracker.list_experiments()
        assert len(all_exps) == 2
        
        # Filter by symbol
        spy_exps = tracker.list_experiments(symbol='SPY')
        assert len(spy_exps) == 1
    
    def test_save_predictions(self, tracker, sample_data):
        """Test saving predictions."""
        exp_id = tracker.start_experiment('SPY', {}, 42, sample_data)
        
        preds = pd.DataFrame({
            'actual': np.random.randn(50),
            'predicted': np.random.randn(50),
        })
        
        path = tracker.save_predictions(exp_id, preds)
        
        assert Path(path).exists()
        assert tracker._active_experiments[exp_id]['predictions_path'] == path


class TestExperimentResult:
    """Test ExperimentResult dataclass."""
    
    def test_create_result(self):
        """Test creating result."""
        result = ExperimentResult(
            experiment_id='test_123',
            created_at='2024-01-01T00:00:00',
            symbol='SPY',
            config={},
            data_hash='abc123',
            seed=42,
            models=[],
            dm_comparisons_5d=[],
            dm_comparisons_20d=[],
            bootstrap_cis={},
            hypothesis_tests=[],
        )
        
        assert result.experiment_id == 'test_123'
        assert result.status == 'completed'
