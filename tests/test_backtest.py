# Unit tests for backtesting (placeholder)
import pytest
from src.backtest import backtest_momentum_strategy
from src.models import IntelligentHFTModel
from src.features import IntelligentFeatureExtractor
from src.utils import fetch_tick_data

def test_backtest():
    config = {'seq_len': 100, 'feature_dim': 54, 'hidden_dim': 128, 'n_heads': 8, 'n_layers': 2, 'action_dim': 3}
    model = IntelligentHFTModel(config).to('cpu')
    feature_extractor = IntelligentFeatureExtractor()
    tick_data = fetch_tick_data(use_sample=True)
    metrics = backtest_momentum_strategy(tick_data, model, feature_extractor, config)
    assert 'accuracy' in metrics
    assert 'profit' in metrics
