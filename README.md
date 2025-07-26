# IntelligentHFTModel

A high-frequency trading (HFT) model for gold (XAU/USD) in MetaTrader 5 (MT5), optimized for CPU-only systems, using Transformers, DQN, and sentiment analysis to achieve maximum profitability with minimal risks.

## Features
- **Machine Learning**: Transformer-based model with DQN for action prediction (buy, sell, hold).
- **Features**: `cx7`, Order Book Imbalance (OBI), VWAP, MACD, and HMM regime detection.
- **Sentiment Analysis**: Integrates Alpha Vantage `NEWS_SENTIMENT` API (free plan: 25 requests/day).
- **Risk Management**: Volatility-based stop-losses (2*ATR), circuit breakers (5% drawdown).
- **CPU Optimization**: Reduced model size and data for efficient CPU execution.
- **Deployment**: ONNX export for MT5 integration with MQL5 EA.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/IntelligentHFTModel.git
   cd IntelligentHFTModel
