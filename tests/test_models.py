# Unit tests for model (placeholder)
import pytest
import torch
from src.models import IntelligentHFTModel

def test_model_forward():
    config = {'seq_len': 100, 'feature_dim': 54, 'hidden_dim': 128, 'n_heads': 8, 'n_layers': 2, 'action_dim': 3}
    model = IntelligentHFTModel(config).to('cpu')
    input_data = torch.randn(1, config['seq_len'], config['feature_dim']).to('cpu')
    q_values, uncertainty, attentions = model(input_data)
    assert q_values.shape == (1, config['action_dim'])
    assert uncertainty.shape == (1, config['action_dim'])
    assert len(attentions) == config['n_layers']
