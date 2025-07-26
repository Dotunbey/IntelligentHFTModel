# Generates synthetic XAU/USD tick data and news sentiment for testing
# Reduced n_ticks to 500,000 for CPU efficiency
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_news(start_timestamp, end_timestamp, interval=3600):
    # Generate hourly sentiment scores (-1 to 1) to mimic Alpha Vantage
    timestamps = np.arange(start_timestamp, end_timestamp, interval)
    sentiments = np.random.uniform(-1, 1, len(timestamps))
    news_df = pd.DataFrame({'time_published': timestamps, 'overall_sentiment_score': sentiments})
    news_df['time_published'] = pd.to_datetime(news_df['time_published'], unit='s')
    return news_df

def generate_synthetic_xauusd_ticks(n_ticks=500000, start_date=datetime(2025, 1, 1)):
    # Initialize arrays for tick data
    prices = np.zeros(n_ticks)
    volumes = np.zeros(n_ticks, dtype=int)
    order_flows = np.zeros(n_ticks)
    bids = np.zeros(n_ticks)
    asks = np.zeros(n_ticks)
    timestamps = np.zeros(n_ticks, dtype=int)
    
    # Set initial values
    prices[0] = 1800.0
    volumes[0] = 500
    order_flows[0] = 0.0
    bids[0] = prices[0] - 0.1
    asks[0] = prices[0] + 0.1
    timestamps[0] = int(start_date.timestamp())
    
    # Generate tick data with realistic volatility
    for i in range(1, n_ticks):
        price_change = np.random.normal(0, 0.1)  # Small random changes
        if np.random.random() < 0.01:  # Occasional larger moves
            price_change += np.random.choice([-0.5, 0.5])
        prices[i] = max(prices[i-1] + price_change, 1700)  # Ensure price > 1700
        volumes[i] = np.random.randint(100, 1000)
        if abs(price_change) > 0.2:  # Increase volume on large moves
            volumes[i] += np.random.randint(200, 500)
        order_flows[i] = np.clip(np.random.normal(0.1 * np.sign(price_change), 0.2), -1, 1)
        bids[i] = prices[i] - 0.1
        asks[i] = prices[i] + 0.1
        timestamps[i] = timestamps[i-1] + 1
    
    # Create data dictionary
    data = {
        'price': prices,
        'volume': volumes,
        'order_flow': order_flows,
        'bid': bids,
        'ask': asks,
        'timestamp': timestamps
    }
    
    # Generate and save news sentiment
    start_timestamp = int(start_date.timestamp())
    end_timestamp = start_timestamp + n_ticks
    news_df = generate_synthetic_news(start_timestamp, end_timestamp)
    news_df.to_csv("data/synthetic_news.csv", index=False)
    print("Synthetic news saved to data/synthetic_news.csv")
    
    # Save tick data
    df = pd.DataFrame(data)
    df.to_csv("data/xauusd_sample_ticks.csv", index=False)
    print(f"Synthetic data saved to data/xauusd_sample_ticks.csv ({n_ticks} ticks)")

if __name__ == "__main__":
    generate_synthetic_xauusd_ticks()
