# Unit tests for feature extractor
import pytest
import numpy as np
from src.features import IntelligentFeatureExtractor

def test_feature_extraction():
    extractor = IntelligentFeatureExtractor()
    tick_data = {
        'price': 1800.0,
        'volume': 100,
        'order_flow': 0.1,
        'bid': 1799.9,
        'ask': 1800.1,
        'timestamp': 1698768000
    }
    # Simulate 1000 ticks
    for _ in range(1000):
        tick_data['price'] += np.random.normal(0, 0.1)
        tick_data['volume'] += np.random.randint(-10, 20)
        tick_data['order_flow'] += np.random.normal(0, 0.01)
        tick_data['bid'] = tick_data['price'] - 0.1
        tick_data['ask'] = tick_data['price'] + 0.1
        tick_data['timestamp'] += 1
        extractor.update_buffers(tick_data)
    features = extractor.extract_intelligent_features()
    assert features is not None
    assert len(features) == 54  # Includes MACD, HMM
    assert abs(features[-5]) < 10  # cx7
    assert 1700 < features[-4] < 1900  # lowest_low_mean
    assert abs(features[-1]) < 3  # regime

if __name__ == "__main__":
    pytest.main()
