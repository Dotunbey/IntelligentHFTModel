# Fetches real XAU/USD tick data from MT5 for backtesting or live trading
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def fetch_real_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M1, start_date=datetime(2025, 1, 1), end_date=datetime(2025, 2, 1)):
    # Initialize MT5 connection
    if not mt5.initialize():
        print("MT5 initialization failed")
        return None
    
    # Fetch tick data
    ticks = mt5.copy_ticks_range(symbol, start_date, end_date, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        print(f"No ticks found for {symbol} from {start_date} to {end_date}")
        mt5.shutdown()
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'last': 'price'}, inplace=True)
    
    # Save to CSV
    df.to_csv("data/xauusd_ticks.csv", index=False)
    print("Data saved to data/xauusd_ticks.csv")
    
    mt5.shutdown()
    return df

if __name__ == "__main__":
    fetch_real_data()
