"""
Load Trained Models and Run Combined Analysis
============================================

This script loads the trained buy and sell models and runs predictions.
Make sure to run 1_Train_Models.py first to train the models.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from ta import momentum
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

import csv

# read what the user wrote
with open(os.path.abspath('./x_utilities/utils.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile)
    reader_indexed = []
    for i in reader:
        reader_indexed.append(i)

start_date = reader_indexed[1][1]
training   = reader_indexed[1][2]
t = reader_indexed[1][0]
ticker = t
end_date = '2025-11-11' # The end date is different from the training end date

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load models at module level so other scripts can import
model_dir = os.path.dirname(os.path.abspath(__file__))
buy_model_path = os.path.join(model_dir, f'{t}_buy_model.joblib')
sell_model_path = os.path.join(model_dir, f'{t}_sell_model.joblib')

buy_model = None
sell_model = None

def load_models():
    """Load the trained models"""
    global buy_model, sell_model
    
    if buy_model is not None and sell_model is not None:
        return  # Already loaded
    
    print("="*60)
    print("LOADING TRAINED MODELS")
    print("="*60)
    
    if not os.path.exists(buy_model_path):
        print(f"❌ Buy model not found at {buy_model_path}")
        print("Please run 1_Train_Models.py first to train the models.")
        raise FileNotFoundError(f"Buy model not found at {buy_model_path}")
    
    if not os.path.exists(sell_model_path):
        print(f"❌ Sell model not found at {sell_model_path}")
        print("Please run 1_Train_Models.py first to train the models.")
        raise FileNotFoundError(f"Sell model not found at {sell_model_path}")
    
    print(f"Loading buy model from: {buy_model_path}")
    buy_model = joblib.load(buy_model_path)
    print("✅ Buy model loaded successfully")
    
    print(f"Loading sell model from: {sell_model_path}")
    sell_model = joblib.load(sell_model_path)
    print("✅ Sell model loaded successfully")

# Load models when module is imported
load_models()

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    print(f"\nFetching data for {ticker} from {start_date} to {end_date}")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date).reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df

def compute_buy_side_features(df):
    """Compute all features needed for buy side model"""
    print("Computing buy side features...")
    
    # Base features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = (df['High'] - df['Low']) / df['Open']
    df['MA_25'] = df['Close'].rolling(window=25, min_periods=1).mean()
    df['MA_100'] = df['Close'].rolling(window=100, min_periods=1).mean()
    df['RSI_14'] = momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # Month growth rate
    def compute_growth_rate(i, closes):
        if i < 12:
            return (closes[i] - closes[0]) / closes[0]
        else:
            return (closes[i] - closes[i - 12]) / closes[i - 12]
    
    df['Month_Growth_Rate'] = df['Close'].rolling(window=len(df), min_periods=1).apply(
        lambda x: compute_growth_rate(len(x)-1, x.values), raw=False
    )

    # Stochastic %K and %D
    low50 = df['Low'].rolling(window=50, min_periods=1).min()
    high50 = df['High'].rolling(window=50, min_periods=1).max()
    df['STOCH_%K'] = (df['Close'] - low50) / (high50 - low50) * 100
    df['STOCH_%D'] = df['STOCH_%K'].rolling(window=20, min_periods=1).mean()

    # Volatility z-score
    df['Roll_Vol_20d'] = df['Daily_Return'].rolling(window=20, min_periods=1).std()
    vol_mean = df['Roll_Vol_20d'].rolling(window=126, min_periods=1).mean()
    vol_std = df['Roll_Vol_20d'].rolling(window=126, min_periods=1).std()
    df['Vol_zscore'] = (df['Roll_Vol_20d'] - vol_mean) / vol_std

    # Ulcer Index
    ui = []
    for i in range(len(df)):
        window = df['Close'].iloc[max(0, i-13) : i+1]
        peak = window.max()
        dd = (peak - window) / peak
        ui.append(np.sqrt((dd**2).mean()))
    df['Ulcer_Index'] = ui

    # Return entropy
    def calc_entropy(series):
        arr = series[~np.isnan(series)]
        if arr.size == 0:
            return 0.0
        counts, _ = np.histogram(arr, bins=10)
        return entropy(counts + 1)
    
    df['Return_Entropy_50d'] = (
        df['Daily_Return']
        .rolling(window=50, min_periods=1)
        .apply(calc_entropy, raw=False)
    )

    # VIX
    try:
        vix = yf.Ticker("^VIX").history(start=df['Date'].min(), end=df['Date'].max()).reset_index()
        vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
        vix = vix[['Date','Close']].rename(columns={'Close':'VIX'})
        df = df.merge(vix, on='Date', how='left')
    except:
        df['VIX'] = 20  # Default VIX value

    # Dip-related features
    df['Growth_Since_Last_Bottom'] = 0.0
    df['Days_Since_Last_Bottom'] = 0
    
    def compute_week_growth_rate(i, closes):
        if i < 5:
            return (closes[i] - closes[0]) / closes[0]
        else:
            return (closes[i] - closes[i - 5]) / closes[i - 5]
    
    df['Week_Growth_Rate'] = df['Close'].rolling(window=len(df), min_periods=1).apply(
        lambda x: compute_week_growth_rate(len(x)-1, x.values), raw=False
    )

    # Compute Growth/Days since last bottom
    last_bot = 0
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] < df.loc[last_bot, 'Close']:
            last_bot = i
        df.loc[i, 'Growth_Since_Last_Bottom'] = (
            df.loc[i, 'Close'] - df.loc[last_bot, 'Close']
        ) / df.loc[last_bot, 'Close']
        df.loc[i, 'Days_Since_Last_Bottom'] = i - last_bot

    # Backfill missing values
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def compute_sell_side_features(df):
    """Compute all features needed for sell side model"""
    print("Computing sell side features...")
    
    # Basic technical indicators
    df['MA_25'] = df['Close'].rolling(25, min_periods=1).mean()
    df['MA_100'] = df['Close'].rolling(100, min_periods=1).mean()
    df['Volatility'] = (df['High'] - df['Low']) / df['Open']

    # Days since last peak
    last_peak_idx = 0
    df['Days_Since_Last_Peak'] = 0
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] > df.loc[last_peak_idx, 'Close']:
            last_peak_idx = i
        df.loc[i, 'Days_Since_Last_Peak'] = (df.loc[i, 'Date'] - df.loc[last_peak_idx, 'Date']).days

    # Decline since last peak
    last_peak_price = df.loc[last_peak_idx, 'Close']
    df['Decline_Since_Last_Peak'] = (df['Close'] - last_peak_price) / last_peak_price
    
    # One week growth
    df['One_Week_Growth'] = df['Close'].pct_change(periods=5)

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['RSI'] = 100 - 100 / (1 + rs)
    df['RSI'] = df['RSI'].fillna(50)

    # Price to MA ratios
    df['Price_to_MA25'] = df['Close'] / df['MA_25']
    df['Price_to_MA100'] = df['Close'] / df['MA_100']
    
    # Fill any remaining NaN values
    df.fillna(0, inplace=True)
    
    return df

def run_combined_analysis(ticker="NVDA", start_date="2024-01-02", end_date="2025-07-25"):
    """Run combined buy and sell analysis"""
    print("\n" + "="*60)
    print("COMBINED BUY & SELL ANALYSIS")
    print("="*60)
    
    # Fetch data
    df = fetch_stock_data(ticker, start_date, end_date)
    df = compute_buy_side_features(df)
    df = compute_sell_side_features(df)
    
    # Prepare buy side features
    buy_features = [
        'Low', 'Close', 'Volatility', 'MA_25', 'MA_100', 'RSI_14', 'Month_Growth_Rate',
        'STOCH_%K', 'STOCH_%D', 'Vol_zscore', 'Ulcer_Index', 'Return_Entropy_50d',
        'VIX', 'Growth_Since_Last_Bottom', 'Days_Since_Last_Bottom', 'Week_Growth_Rate'
    ]
    
    # Ensure all buy features exist
    for feature in buy_features:
        if feature not in df.columns:
            df[feature] = 0
    
    X_buy = df[buy_features].fillna(0)
    
    # Make buy predictions
    df['Dip_Probability'] = buy_model.predict_proba(X_buy)[:, 1]
    df['Predicted_Dip'] = df['Dip_Probability'] > 0.9
    
    # Prepare sell side features
    sell_features = [
        'Close', 'Volatility', 'MA_25', 'MA_100',
        'Decline_Since_Last_Peak', 'Days_Since_Last_Peak',
        'One_Week_Growth', 'RSI', 'Price_to_MA25', 'Price_to_MA100'
    ]
    
    # Ensure all sell features exist
    for feature in sell_features:
        if feature not in df.columns:
            df[feature] = 0
    
    X_sell = df[sell_features].fillna(0)
    
    # Make sell predictions
    df['Sell_Probability'] = sell_model.predict_proba(X_sell)[:, 1]
    df['Sell_Signal'] = df['Sell_Probability'] > 0.9
    
    # Analysis
    buy_signals = df[df['Predicted_Dip']]
    sell_signals = df[df['Sell_Signal']]
    
    print(f"\n📊 COMBINED ANALYSIS RESULTS:")
    print(f"  Analysis period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Total trading days: {len(df)}")
    print(f"  Buy signals: {len(buy_signals)} ({len(buy_signals)/len(df)*100:.2f}%)")
    print(f"  Sell signals: {len(sell_signals)} ({len(sell_signals)/len(df)*100:.2f}%)")
    
    if len(buy_signals) > 0:
        print(f"  Buy signal dates:")
        for _, signal in buy_signals.iterrows():
            print(f"    {signal['Date'].strftime('%Y-%m-%d')}: ${signal['Close']:.2f} (Prob: {signal['Dip_Probability']:.3f})")
    
    if len(sell_signals) > 0:
        print(f"  Sell signal dates:")
        for _, signal in sell_signals.iterrows():
            print(f"    {signal['Date'].strftime('%Y-%m-%d')}: ${signal['Close']:.2f} (Prob: {signal['Sell_Probability']:.3f})")
    
    # Create combined visualization
    print("\n📈 Creating combined visualization...")
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue'),
        hoverinfo='skip'
    ))
    
    # Add buy signals - green circles like individual model
    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(
            x=buy_signals['Date'],
            y=buy_signals['Close'],
            mode='markers+text',
            name='Predicted Dip',
            marker=dict(color='green', size=8),
            text=[f"Prob {p:.2f}" for p in buy_signals['Dip_Probability']],
            textposition="top center",
            hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>Dip Probability: %{text}<extra></extra>"
        ))
    
    # Add sell signals
    if len(sell_signals) > 0:
        fig.add_trace(go.Scatter(
            x=sell_signals['Date'],
            y=sell_signals['Close'],
            mode='markers+text',
            name='Predicted Sell',
            marker=dict(color='red', size=8),
            text=[f"Prob {p:.2f}" for p in sell_signals['Sell_Probability']],
            textposition="top center",
            hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>Sell Probability: %{text}<extra></extra>"
        ))
    
    # Update layout - full width like individual models
    fig.update_layout(
        title=f"Stock Price with Predicted Dips and Sell Signals \
            \n<span style='font-size:16px; color:blue;'>TICKER: {t}, FROM {start_date} TO {training}</span>",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="closest",
        template="plotly_white",
        autosize=True,
        width=None,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right", 
            x=1
        )
    )
    
    fig.add_vrect(
        x0=start_date, 
        x1=training,
        fillcolor="yellow", 
        opacity=0.3,
        layer="below",
        line_width=0
    )

    fig.show()
    
    # Save results
    output_file = os.path.join(model_dir, f"combined_analysis_{ticker}_{df['Date'].min().strftime('%Y%m%d')}_{df['Date'].max().strftime('%Y%m%d')}.csv")
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    return df

# Run the analysis
if __name__ == "__main__":
    # Default parameters - same timeframe as individual models
    ticker = t
    start_date = start_date
    end_date = end_date
    
    results = run_combined_analysis(ticker, start_date, end_date)
    
    print("\n✅ Combined analysis complete!")
