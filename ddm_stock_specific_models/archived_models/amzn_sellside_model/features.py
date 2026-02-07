import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

def calculate_features(df):
    """Calculates a comprehensive set of financial features for the model."""
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # MA & Ratios
    df['MA_25'] = df['Close'].rolling(25, min_periods=1).mean()
    df['Price_to_MA25'] = df['Close'] / df['MA_25']
    
    # Volatility
    df['Volatility'] = (df['High'] - df['Low']) / df['Open']
    bb = BollingerBands(close=df['Close'], window=20)
    df['Bollinger_Band_Width'] = bb.bollinger_wband()

    # Momentum Indic.
    df['ROC_14'] = ROCIndicator(close=df['Close'], window=14).roc()
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stochastic_K'] = stoch.stoch()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['One_Week_Growth'] = df['Close'].pct_change(periods=5)

    # Divergence
    price_peak = df['Close'].rolling(14).max()
    rsi_peak = df['RSI'].rolling(14).max()
    df['Bearish_Divergence'] = ((df['Close'] == price_peak) & (df['RSI'] < rsi_peak * 0.9)).astype(int)

    # Peak and Decline
    df['Days_Since_Last_Peak'] = 0
    df['Decline_Since_Last_Peak'] = 0.0
    last_peak_idx = 0
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] > df.loc[last_peak_idx, 'Close']:
            last_peak_idx = i
        last_peak_price = df.loc[last_peak_idx, 'Close']
        df.loc[i, 'Days_Since_Last_Peak'] = (df.loc[i, 'Date'] - df.loc[last_peak_idx, 'Date']).days
        df.loc[i, 'Decline_Since_Last_Peak'] = (df.loc[i, 'Close'] - last_peak_price) / last_peak_price if last_peak_price != 0 else 0
    
    # Vol
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    
    return df