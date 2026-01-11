import pandas as pd
import pandas_datareader.data as web
import numpy as np

def get_cleaned_data(start_date="2020-01-01", end_date="2026-01-01"):
    """
    Fetches financial data from Stooq (No API Key required), performs feature engineering,
    and prepares the dataset for the machine learning model.
    """
    print("\n Launching Stooq Global Data Source \...")
    
    # --- 1. Define Stock Universe (12 Cross-sector Blue Chips) ---
    tickers = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL',  # Tech (High Growth)
        'JPM', 'V',                               # Finance (Cyclical)
        'JNJ', 'PFE',                             # Healthcare (Defensive)
        'PG', 'KO',                               # Consumer Goods (Inflation Resistant)
        'SPY'                                     # Market Benchmark (S&P 500)
    ]
    
    all_features = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}]  Downloading: {ticker} ...")
        try:
            # Fetch data using Stooq (Robust, free, no API key needed)
            df = web.DataReader(ticker, 'stooq', start=start_date, end=end_date)
            
            # reverse it to chronological order (oldest first) for correct calculations.
            price = df['Close'].iloc[::-1]
            
            #  2. Feature Engineering 
            feat_df = pd.DataFrame(index=price.index)
            feat_df['Close'] = price
            
            # A. Momentum Factors (Trend following)
            feat_df['mom_1m'] = price.pct_change(20) # 1-month return
            feat_df['mom_3m'] = price.pct_change(60) # 3-month return
            
            # B. Volatility Factors (Risk measurement)
            # 20-day rolling standard deviation of returns
            feat_df['vol_20d'] = price.pct_change().rolling(20).std()
            
            # C. RSI (Relative Strength Index)
            # Measures the speed and change of price movements
            delta = price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9) # Add epsilon to avoid division by zero
            feat_df['rsi'] = 100 - (100 / (1 + rs))
            
            # D. Moving Average Divergence (Mean Reversion)
            # Distance from the 50-day trend line
            feat_df['ma_divergence'] = price / price.rolling(50).mean() - 1
            
            # E. Prediction Target (Future Return)
            # We want to predict the return 20 days into the future
            feat_df['target_return'] = price.shift(-20) / price - 1
            feat_df['ticker'] = ticker
            
            all_features.append(feat_df)
            print(f"✅ {ticker} processed successfully.")
            
        except Exception as e:
            print(f"❌ Failed to download {ticker}: {e}")

    if not all_features:
        print("\n❌ Critical Error: All downloads failed. Please check internet connection.")
        return None, None, None

    # --- 3. Data Merging & Cleaning ---
    full_df = pd.concat(all_features)
    
    # Remove rows with NaN values (created by rolling windows and lagging)
    full_df.dropna(inplace=True)
    
    # Outlier Removal (Winsorization / IQR Clipping)
    # Caps extreme values between the 5th and 95th percentiles to stabilize training
    features = ['mom_1m', 'mom_3m', 'vol_20d', 'rsi', 'ma_divergence']
    for col in features:
        Q1 = full_df[col].quantile(0.05)
        Q3 = full_df[col].quantile(0.95)
        full_df[col] = full_df[col].clip(lower=Q1, upper=Q3)

    print(f"\nDataset Built Successfully! Total Samples: {len(full_df)}")
    print(f"Universe: {full_df['ticker'].unique()}")
    
    return full_df[features], full_df['target_return'], full_df

if __name__ == "__main__":
    # Test the function
    # Note: Requires 'pandas_datareader'. Install via: pip install pandas_datareader
    X, y, df = get_cleaned_data()
    if X is not None:
        print("\nData Preview (Last 5 rows):")
        print(X.tail())