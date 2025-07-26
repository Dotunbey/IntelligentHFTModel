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
nstall dependencies:bash

pip install -r requirements.txt

Create directories:bash

mkdir -p gold_checkpoints data

(Optional) Obtain a free Alpha Vantage API key from https://www.alphavantage.co for real sentiment data.

UsageGenerate Synthetic Data (no API key needed):bash

python data/generate_sample_data.py

Train the Model:bash

bash scripts/run_training.sh

Backtest:bash

bash scripts/run_backtest.sh

Visualize Results:bash

python src/visualize.py
jupyter notebook notebooks/analyze_metrics.ipynb

Deploy in MT5 (simulation):bash

python src/mt5_integration.py
# Copy hft_model.onnx and trades.csv to MT5 Files folder
# Copy src/XAUUSD_EA.mq5 to MT5 Experts folder
# Compile in MetaEditor and run in Strategy Tester

Backtest with Real Data:Update src/backtest.py with your API key.
Switch to real data:bash

sed -i 's/use_sample=True/use_sample=False/' src/backtest.py
bash scripts/run_backtest.sh

RequirementsHardware: CPU with 8GB RAM (16GB recommended).
Software: Python 3.8+, MT5 (optional), Jupyter Notebook.
Metrics:60% accuracy

2.0 Sharpe ratio

65% win rate

<5% drawdown
<1ms latency (HFT target, may vary on CPU).

NotesUses synthetic data for testing to avoid API costs.
Complies with Alpha Vantage free plan (25 requests/day, 5/minute) via caching.
Optimized for CPU with reduced model size and data.

LicenseMIT License

**Explanation**:
- Updated to reflect CPU-only execution and free API plan compliance.
- Clarified setup and usage, including API key instructions.
- Listed target metrics to align with your goal.

