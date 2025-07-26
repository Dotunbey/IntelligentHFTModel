# Backtests the HFT momentum strategy with sentiment and HMM, optimized for CPU
import numpy as np
import torch
import torch.profiler
from src.models import IntelligentHFTModel
from src.features import IntelligentFeatureExtractor
from src.utils import calculate_atr, fetch_news_sentiment
import pandas as pd
from datetime import datetime
from bisect import bisect_left

def get_latest_sentiment(news_df, tick_time):
    # Find the latest sentiment score before tick_time
    idx = bisect_left(news_df['time_published'], tick_time)
    if idx == 0:
        return np.nan
    else:
        return news_df.iloc[idx - 1]['overall_sentiment_score']

def backtest_momentum_strategy(tick_data, model, feature_extractor, config, use_sample=True):
    # Load synthetic or real sentiment data
    if use_sample:
        news_df = pd.read_csv("data/synthetic_news.csv")
        news_df['time_published'] = pd.to_datetime(news_df['time_published'])
        print("Loaded synthetic news sentiment from data/synthetic_news.csv")
    else:
        apikey = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Replace with your free API key
        from_date = '20250101T0000'
        to_date = '20250201T0000'
        news_df = fetch_news_sentiment('FOREX:XAUUSD', from_date, to_date, apikey)
        if news_df.empty:
            print("No news data available, using neutral sentiment")
            news_df = pd.DataFrame({'time_published': [pd.to_datetime('2025-01-01')], 'overall_sentiment_score': [0.0]})
    
    news_df = news_df.sort_values('time_published')
    
    trades = []
    position = 0
    capital = 10000
    returns = []
    predictions = []
    actuals = []
    confidences = []
    latencies = []
    atr = calculate_atr(tick_data)
    
    # Profile CPU performance
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True
    )
    
    model.eval()
    with torch.no_grad():
        for i in range(config['seq_len'], len(tick_data['price']) - 1):
            # Update feature buffers
            feature_extractor.update_buffers({
                'price': tick_data['price'][i],
                'volume': tick_data['volume'][i],
                'order_flow': tick_data['order_flow'][i],
                'bid': tick_data['bid'][i],
                'ask': tick_data['ask'][i],
                'timestamp': tick_data['timestamp'][i]
            })
            features = feature_extractor.extract_intelligent_features()
            if features is None:
                continue
            
            # Prepare market data for model
            market_data = {
                'features': torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to('cpu'),
                'prices': torch.tensor(list(feature_extractor.price_buffer), dtype=torch.float32).unsqueeze(0).to('cpu'),
                'volumes': torch.tensor(list(feature_extractor.volume_buffer), dtype=torch.float32).unsqueeze(0).to('cpu'),
                'order_flow': torch.tensor(list(feature_extractor.order_flow_buffer), dtype=torch.float32).unsqueeze(0).to('cpu')
            }
            
            with profiler.record_function("model_inference"):
                pred = model.predict_with_confidence(market_data)
            
            action = pred['action']
            confidence = pred['confidence']
            cx7 = features[-5]
            order_flow_imbalance = features[-7]
            vwap = features[-4]
            
            # Define ground truth for accuracy
            price_change = tick_data['price'][i+1] - tick_data['price'][i]
            actual_action = 1 if price_change > 0.1 else 2 if price_change < -0.1 else 0
            predictions.append(action)
            actuals.append(actual_action)
            confidences.append(confidence)
            
            # Get sentiment
            tick_time = pd.to_datetime(tick_data['timestamp'][i], unit='s')
            sentiment = get_latest_sentiment(news_df, tick_time)
            if np.isnan(sentiment):
                sentiment = 0.0
            
            # Trading logic with strict filters
            if confidence > 0.8 and abs(order_flow_imbalance) > 0.5:
                if action == 1 and sentiment > 0.2 and cx7 > 0.5 and tick_data['price'][i] > vwap and position != 1:
                    trades.append({
                        'action': 'buy', 'price': tick_data['ask'][i], 'time': tick_data['timestamp'][i],
                        'sl': tick_data['price'][i] - 2*atr, 'cx7': cx7, 'order_flow_imbalance': order_flow_imbalance,
                        'sentiment': sentiment
                    })
                    position = 1
                    returns.append(-(tick_data['ask'][i] - tick_data['bid'][i] + 0.3) / capital)  # Include slippage (0.2 pips) and commission (0.1 pip)
                elif action == 2 and sentiment < -0.2 and cx7 < -0.5 and tick_data['price'][i] < vwap and position != -1:
                    trades.append({
                        'action': 'sell', 'price': tick_data['bid'][i], 'time': tick_data['timestamp'][i],
                        'sl': tick_data['price'][i] + 2*atr, 'cx7': cx7, 'order_flow_imbalance': order_flow_imbalance,
                        'sentiment': sentiment
                    })
                    position = -1
                    returns.append(-(tick_data['ask'][i] - tick_data['bid'][i] + 0.3) / capital)
                elif action == 0 and position != 0:
                    price_change = (tick_data['price'][i] - tick_data['price'][i-1]) * position
                    returns.append(price_change / capital)
                    trades.append({
                        'action': 'close', 'price': tick_data['price'][i], 'time': tick_data['timestamp'][i],
                        'cx7': cx7, 'order_flow_imbalance': order_flow_imbalance,
                        'sentiment': sentiment
                    })
                    position = 0
                # Stop-loss check
                if position != 0 and trades[-1].get('sl') is not None:
                    if (position == 1 and tick_data['price'][i] < trades[-1]['sl']) or \
                       (position == -1 and tick_data['price'][i] > trades[-1]['sl']):
                        price_change = (tick_data['price'][i] - tick_data['price'][i-1]) * position
                        returns.append(price_change / capital)
                        trades.append({
                            'action': 'close', 'price': tick_data['price'][i], 'time': tick_data['timestamp'][i],
                            'cx7': cx7, 'order_flow_imbalance': order_flow_imbalance,
                            'sentiment': sentiment
                        })
                        position = 0
                # Circuit breaker: pause if drawdown > 5%
                if np.min(np.cumsum(returns)) * capital < -0.05 * capital:
                    print("Circuit breaker triggered: Drawdown > 5%")
                    break
            
            latencies.append(profiler.self_cpu_time_total / 1000)
        
        # Compute metrics
        accuracy = np.mean(np.array(predictions) == np.array(actuals))
        trade_outcomes = [1 if r > 0 else 0 for r, t in zip(returns, trades) if t['action'] != 'close']
        confidence_mae = np.mean(np.abs(np.array(confidences[:len(trade_outcomes)]) - np.array(trade_outcomes)))
        profit = np.sum(returns) * capital
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        win_rate = np.mean(np.array([r for r, t in zip(returns, trades) if t['action'] != 'close']) > 0)
        max_drawdown = np.min(np.cumsum(returns) - np.maximum.accumulate(np.cumsum(returns))) * capital
        trade_frequency = len(trades) / ((tick_data['timestamp'][-1] - tick_data['timestamp'][0]) / (24 * 3600))
        avg_latency = np.mean(latencies)
        
        # Save results
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv("trades.csv", index=False)
        
        metrics = {
            'accuracy': accuracy,
            'confidence_mae': confidence_mae,
            'profit': profit,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'trade_frequency': trade_frequency,
            'avg_latency_ms': avg_latency
        }
        pd.DataFrame([metrics]).to_csv("metrics.csv", index=False)
        
        # Print summary
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confidence MAE: {confidence_mae:.4f}")
        print(f"Profit: ${profit:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Win Rate: {win_rate:.2f}")
        print(f"Max Drawdown: ${max_drawdown:.2f}")
        print(f"Trade Frequency: {trade_frequency:.2f} trades/day")
        print(f"Average Latency: {avg_latency:.2f} ms")
        
        profiler.export_chrome_trace("gold_checkpoints/profiler_trace.json")
        
        return metrics

if __name__ == "__main__":
    # Config optimized for CPU
    config = {'seq_len': 100, 'feature_dim': 54, 'hidden_dim': 128, 'n_heads': 8, 'n_layers': 2, 'action_dim': 3}
    model = IntelligentHFTModel(config).to('cpu')
    feature_extractor = IntelligentFeatureExtractor()
    from src.utils import fetch_tick_data, load_best_checkpoint
    tick_data = fetch_tick_data(use_sample=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    load_best_checkpoint(model, optimizer)
    metrics = backtest_momentum_strategy(tick_data, model, feature_extractor, config)
