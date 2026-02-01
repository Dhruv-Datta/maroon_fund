"""
Individual Model Visualizations
================================

This script visualizes buy and sell side models separately.
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

with open(os.path.abspath('./combined_buy_sell_model/utils.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile)
    reader_indexed = []
    for i in reader:
        reader_indexed.append(i)

start_date = reader_indexed[1][1]
end_date   = reader_indexed[1][2]
t = reader_indexed[1][0]

ticker = t

# Import functions from 2_Loaded_Models
model_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(model_dir)

# Import will trigger model loading
import importlib.util
spec = importlib.util.spec_from_file_location("loaded_models", os.path.join(model_dir, "2_Loaded_Models.py"))
loaded_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loaded_models)

# Get the functions and models
fetch_stock_data = loaded_models.fetch_stock_data
compute_buy_side_features = loaded_models.compute_buy_side_features
compute_sell_side_features = loaded_models.compute_sell_side_features
buy_model = loaded_models.buy_model
sell_model = loaded_models.sell_model

def visualize_buy_model(ticker=t, start_date=start_date, end_date=end_date):
    """Visualize buy side model results"""
    print("="*60)
    print("BUY SIDE MODEL VISUALIZATION")
    print("="*60)
    
    # Fetch and prepare data
    df = fetch_stock_data(ticker, start_date, end_date)
    df = compute_buy_side_features(df)
    
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
    
    # Make predictions
    df['Dip_Probability'] = buy_model.predict_proba(X_buy)[:, 1]
    df['Predicted_Dip'] = df['Dip_Probability'] > 0.9
    
    # Analysis
    predicted_dips = df[df['Predicted_Dip']]
    print(f"\n📊 BUY SIDE RESULTS:")
    print(f"  Analysis period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Total data points: {len(df)}")
    print(f"  Predicted dips: {len(predicted_dips)} ({len(predicted_dips)/len(df)*100:.2f}%)")
    
    if len(predicted_dips) > 0:
        print(f"  Average probability: {predicted_dips['Dip_Probability'].mean():.4f}")
        print(f"  Buy signal dates:")
        for _, signal in predicted_dips.iterrows():
            print(f"    {signal['Date'].strftime('%Y-%m-%d')}: ${signal['Close']:.2f} (Prob: {signal['Dip_Probability']:.3f})")
    
    # Create visualization
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
    
    # Add predicted dips - red circles like individual model
    if len(predicted_dips) > 0:
        fig.add_trace(go.Scatter(
            x=predicted_dips['Date'],
            y=predicted_dips['Close'],
            mode='markers+text',
            name='Predicted Dip',
            marker=dict(color='red', size=8),
            text=[f"Prob {p:.2f}" for p in predicted_dips['Dip_Probability']],
            textposition="top center",
            hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>Dip Probability: %{text}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Stock Price with Predicted Dips",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="closest",
        template="plotly_white",
        autosize=True,
        width=None,
        height=600
    )
    
    fig.show()
    return df

def visualize_sell_model(ticker=t, start_date=start_date, end_date=end_date):
    """Visualize sell side model results"""
    print("\n" + "="*60)
    print("SELL SIDE MODEL VISUALIZATION")
    print("="*60)
    
    # Fetch and prepare data
    df = fetch_stock_data(ticker, start_date, end_date)
    df = compute_sell_side_features(df)
    
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
    
    # Make predictions
    df['Sell_Probability'] = sell_model.predict_proba(X_sell)[:, 1]
    df['Sell_Signal'] = df['Sell_Probability'] > 0.9
    
    # Analysis
    sell_signals = df[df['Sell_Signal']]
    print(f"\n📊 SELL SIDE RESULTS:")
    print(f"  Analysis period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Total data points: {len(df)}")
    print(f"  Sell signals: {len(sell_signals)} ({len(sell_signals)/len(df)*100:.2f}%)")
    
    if len(sell_signals) > 0:
        print(f"  Average probability: {sell_signals['Sell_Probability'].mean():.4f}")
        print(f"  Sell signal dates:")
        for _, signal in sell_signals.iterrows():
            print(f"    {signal['Date'].strftime('%Y-%m-%d')}: ${signal['Close']:.2f} (Prob: {signal['Sell_Probability']:.3f})")
    
    # Create visualization
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
    return df

if __name__ == "__main__":
    # Default parameters - same timeframe as individual models
    ticker = t
    start_date = start_date
    end_date = end_date
    
    # Visualize buy side
    buy_data = visualize_buy_model(ticker, start_date, end_date)
    
    # Visualize sell side
    sell_data = visualize_sell_model(ticker, start_date, end_date)
    
    print("\n✅ Individual visualizations complete!")
