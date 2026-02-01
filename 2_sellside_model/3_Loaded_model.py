import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib, xgboost as xgb, yfinance as yf
from datetime import datetime, timedelta
import os, csv

# read what the user wrote
with open(os.path.abspath('./x_utilities/utils.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile)
    reader_indexed = []
    for i in reader:
        reader_indexed.append(i)

start_date = reader_indexed[1][1]
end_date   = "2025-11-07"
t = reader_indexed[1][0]

sell_model = joblib.load(f"2_sellside_model/sell_signals/{t}_sell_model.joblib")

def predict_sell_signal(ticker="NVDA", start_date=None, end_date=None, days_back=60, prob_threshold=0.9):
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)
    if start_date is None:
        start_date = end_date - timedelta(days=days_back)
    else:
        start_date = pd.to_datetime(start_date)

    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date).reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    df['MA_25'] = df['Close'].rolling(25, min_periods=1).mean()
    df['MA_100'] = df['Close'].rolling(100, min_periods=1).mean()
    df['Volatility'] = (df['High'] - df['Low']) / df['Open']

    last_peak_idx = 0
    df['Days_Since_Last_Peak'] = 0
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] > df.loc[last_peak_idx, 'Close']:
            last_peak_idx = i
        df.loc[i, 'Days_Since_Last_Peak'] = (df.loc[i, 'Date'] - df.loc[last_peak_idx, 'Date']).days

    last_peak_price = df.loc[last_peak_idx, 'Close']
    df['Decline_Since_Last_Peak'] = (df['Close'] - last_peak_price) / last_peak_price
    df['One_Week_Growth'] = df['Close'].pct_change(periods=5)

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['RSI'] = 100 - 100 / (1 + rs)
    df['RSI'] = df['RSI'].fillna(50)

    df['Price_to_MA25'] = df['Close'] / df['MA_25']
    df['Price_to_MA100'] = df['Close'] / df['MA_100']

    feats = [
        'Close', 'Volatility', 'MA_25', 'MA_100',
        'Decline_Since_Last_Peak', 'Days_Since_Last_Peak',
        'One_Week_Growth', 'RSI', 'Price_to_MA25', 'Price_to_MA100'
    ]
    X = df[feats].fillna(0)

    df['Sell_Probability'] = sell_model.predict_proba(X)[:, 1]
    df['Sell_Signal'] = df['Sell_Probability'] > prob_threshold

    return df

ticker = t
nvda_df = predict_sell_signal(
    ticker,
    start_date=start_date,
    end_date=end_date
)

# Create Plotly visualization matching buy side style
fig = go.Figure()

# Add price line
fig.add_trace(go.Scatter(
    x=nvda_df['Date'],
    y=nvda_df['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='blue'),
    hoverinfo='skip'
))

# Add sell signals - red circles like buy side model
sell_signals = nvda_df[nvda_df['Sell_Signal']]
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

# Update layout - full width like buy side model
fig.update_layout(
    title="Stock Price with Predicted Sell Signals",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="closest",
    template="plotly_white",
    autosize=True,
    width=None,
    height=600
)

fig.show()

out_name = f"{ticker}_sell_analysis_{nvda_df.Date.min().strftime('%Y%m%d')}_{nvda_df.Date.max().strftime('%Y%m%d')}.csv"
nvda_df.to_csv(out_name, index=False)
print(f"Range analysis saved to → {out_name}")