# Utility functions for data fetching, checkpointing, and sentiment caching
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import requests
import time
import os
import torch
import numpy as np

def fetch_tick_data(symbol="XAUUSD", start_time=datetime(2025, 1, 1), end_time=datetime(2025, 2, 1), use_sample=True):
    # Fetch synthetic or real tick data
    if use_sample:
        df = pd.read_csv("data/xauusd_sample_ticks.csv")
        print("Loaded synthetic data from data/xauusd_sample_ticks.csv")
        return {
            'price': df['price'].values,
            'volume': df['volume'].values,
            'order_flow': df['order_flow'].values,
            'bid': df['bid'].values,
            'ask': df['ask'].values,
            'timestamp': df['timestamp'].values
        }
    mt5.initialize()
    ticks = mt5.copy_ticks_range(symbol, mt5.TIMEFRAME_M1, start_time, end_time)
    df = pd.DataFrame(ticks)
    return {
        'price': df['last'].values,
        'volume': df['volume'].values,
        'order_flow': (df['volume'] * (df['flags'] & mt5.TICK_FLAG_BUY - df['flags'] & mt5.TICK_FLAG_SELL)).values,
        'bid': df['bid'].values,
        'ask': df['ask'].values,
        'timestamp': df['time'].values
    }

def fetch_news_sentiment(ticker, from_date, to_date, apikey, cache_file="data/news_sentiment.csv"):
    # Fetch sentiment with caching to stay within free API limits
    if os.path.exists(cache_file):
        cached_df = pd.read_csv(cache_file)
        cached_df['time_published'] = pd.to_datetime(cached_df['time_published'])
        if (cached_df['time_published'].min() <= pd.to_datetime(from_date)) and \
           (cached_df['time_published'].max() >= pd.to_datetime(to_date)):
            print(f"Using cached sentiment data from {cache_file}")
            return cached_df
    
    # Fetch daily sentiment to minimize requests
    news_data = []
    current_date = pd.to_datetime(from_date)
    end_date = pd.to_datetime(to_date)
    while current_date <= end_date:
        next_date = min(current_date + timedelta(days=1), end_date)
        from_str = current_date.strftime("%Y%m%dT%H%M")
        to_str = next_date.strftime("%Y%m%dT%H%M")
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={from_str}&time_to={to_str}&apikey={apikey}"
        try:
            response = requests.get(url)
            data = response.json()
            if 'feed' not in data:
                print(f"No news data for {from_str} to {to_str}")
            else:
                for item in data['feed']:
                    news_data.append({
                        'time_published': pd.to_datetime(item['timestamp']),
                        'overall_sentiment_score': float(item['overall_sentiment_score'])
                    })
        except Exception as e:
            print(f"Error fetching sentiment for {from_str}: {e}")
        current_date = next_date
        time.sleep(12)  # 12-second delay for 5 requests/minute
    news_df = pd.DataFrame(news_data)
    if not news_df.empty:
        news_df.to_csv(cache_file, index=False)
        print(f"Saved sentiment data to {cache_file}")
    return news_df

def calculate_atr(tick_data, period=14):
    # Calculate Average True Range
    highs = tick_data['price'][:-1]
    lows = tick_data['price'][1:]
    return np.mean(np.abs(highs - lows)[-period:])

def save_checkpoint(model, optimizer, epoch, metrics, path="gold_checkpoints/best.pth"):
    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }, path)

def load_best_checkpoint(model, optimizer, checkpoint_dir="gold_checkpoints"):
    # Load best checkpoint based on Sharpe ratio
    best_sharpe = -float('inf')
    best_path = None
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            checkpoint = torch.load(os.path.join(checkpoint_dir, file))
            if checkpoint['metrics']['sharpe_ratio'] > best_sharpe:
                best_sharpe = checkpoint['metrics']['sharpe_ratio']
                best_path = file
    if best_path:
        checkpoint = torch.load(os.path.join(checkpoint_dir, best_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded best checkpoint: {best_path}, Sharpe: {best_sharpe}")
        return checkpoint['epoch'], checkpoint['metrics']
    return 0, None
