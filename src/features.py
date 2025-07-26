# Extracts features for HFT model, including cx7, OBI, VWAP, MACD, and HMM
import numpy as np
from collections import deque
import pandas as pd
from hmmlearn.hmm import GaussianHMM

class IntelligentFeatureExtractor:
    def __init__(self, lookback_periods=[5, 10, 20, 50]):
        self.lookback_periods = lookback_periods
        self.price_buffer = deque(maxlen=1000)
        self.volume_buffer = deque(maxlen=1000)
        self.order_flow_buffer = deque(maxlen=1000)
        self.trade_buffer = deque(maxlen=1000)
        self.cx7_buffer = deque(maxlen=1000)
        self.fibb7 = 0.1  # Scaling factor for cx7
        # Lightweight HMM for regime detection (trending, mean-reverting, volatile)
        self.hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=50)
    
    def update_buffers(self, tick_data):
        # Update buffers with new tick data
        self.price_buffer.append(tick_data['price'])
        self.volume_buffer.append(tick_data['volume'])
        self.order_flow_buffer.append(tick_data['order_flow'])
        self.trade_buffer.append(tick_data)
        # Calculate cx7: (price change % / volume) * fibb7 * 7
        change = ((tick_data['price'] - self.price_buffer[-2]) / self.price_buffer[-2] * 100) if len(self.price_buffer) > 1 else 0
        vol_c = tick_data['volume']
        cx7 = self.fibb7 * change / (vol_c + 1e-6) * 7
        self.cx7_buffer.append(cx7)
    
    def extract_intelligent_features(self):
        # Extract features if enough data is available
        if len(self.price_buffer) < max(self.lookback_periods):
            return None
        prices = np.array(self.price_buffer)
        volumes = np.array(self.volume_buffer)
        flows = np.array(self.order_flow_buffer)
        lowest_low_mean = np.mean([min(prices[i:i+20]) for i in range(0, len(prices), 20) if i+20 <= len(prices)])
        features = {}
        # Technical and statistical features for multiple lookback periods
        for period in self.lookback_periods:
            p = prices[-period:]
            v = volumes[-period:]
            f = flows[-period:]
            features[f'return_{period}'] = (p[-1] - p[0]) / p[0]
            features[f'volatility_{period}'] = np.std(np.diff(p) / p[:-1]) if len(p) > 1 else 0
            features[f'momentum_{period}'] = np.mean(np.diff(p)) if len(p) > 1 else 0
            features[f'rsi_{period}'] = self.calculate_rsi(p)
            features[f'volume_ratio_{period}'] = v[-1] / np.mean(v[:-1]) if len(v) > 1 else 1
            features[f'volume_volatility_{period}'] = np.std(v)
            features[f'price_volume_corr_{period}'] = np.corrcoef(p, v)[0, 1] if len(v) > 1 else 0
            features[f'flow_imbalance_{period}'] = np.sum(f[-100:]) / 100 if len(f) >= 100 else 0
        # Market microstructure features
        features['bid_ask_spread'] = self.calculate_spread()
        features['trade_intensity'] = self.calculate_trade_intensity()
        features['price_impact'] = self.calculate_price_impact()
        features['market_depth'] = self.calculate_market_depth()
        features['volatility_regime'] = self.detect_volatility_regime()
        features['trend_strength'] = self.calculate_trend_strength()
        features['mean_reversion'] = self.calculate_mean_reversion_signal()
        features['cx7'] = self.cx7_buffer[-1] if self.cx7_buffer else 0
        features['lowest_low_mean'] = lowest_low_mean
        features['vwap'] = np.sum(prices[-50:] * volumes[-50:]) / np.sum(volumes[-50:]) if np.sum(volumes[-50:]) > 0 else prices[-1]
        # MACD for trend detection
        macd_line, signal_line, macd_histogram = self.calculate_macd(prices)
        features['macd_line'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = macd_histogram
        # HMM regime detection
        features['regime'] = self.detect_regime(prices)
        return np.array(list(features.values()), dtype=np.float32)
    
    def calculate_rsi(self, prices, period=14):
        # Relative Strength Index
        if len(prices) < period + 1:
            return 0.5
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        rs = avg_gain / (avg_loss + 1e-6)
        return 1 - (1 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        # Moving Average Convergence Divergence
        prices = np.array(prices)
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line.iloc[-1], signal_line.iloc[-1], macd_histogram.iloc[-1]
    
    def detect_regime(self, prices):
        # Lightweight HMM for market regime detection
        if len(prices) < 100:
            return 0
        prices = prices[-100:].reshape(-1, 1)
        try:
            self.hmm_model.fit(prices)
            return self.hmm_model.predict(prices)[-1]
        except:
            return 0
    
    def calculate_spread(self):
        return self.trade_buffer[-1]['ask'] - self.trade_buffer[-1]['bid'] if self.trade_buffer else 0.1
    
    def calculate_trade_intensity(self):
        return len(self.trade_buffer) / 100 if len(self.trade_buffer) > 0 else 0
    
    def calculate_price_impact(self):
        if len(self.price_buffer) < 2:
            return 0
        return abs(self.price_buffer[-1] - self.price_buffer[-2]) / (self.volume_buffer[-1] + 1e-6)
    
    def calculate_market_depth(self):
        return np.mean([abs(self.order_flow_buffer[i]) for i in range(-100, 0) if i < 0]) if len(self.order_flow_buffer) >= 100 else 0
    
    def detect_volatility_regime(self):
        if len(self.price_buffer) < 20:
            return 0
        return np.std(np.diff(list(self.price_buffer)[-20:]))
    
    def calculate_trend_strength(self):
        if len(self.price_buffer) < 50:
            return 0
        prices = list(self.price_buffer)[-50:]
        return (prices[-1] - prices[0]) / np.std(prices)
    
    def calculate_mean_reversion_signal(self):
        if len(self.price_buffer) < 20:
            return 0
        prices = list(self.price_buffer)[-20:]
        mean = np.mean(prices)
        return (mean - prices[-1]) / np.std(prices)
