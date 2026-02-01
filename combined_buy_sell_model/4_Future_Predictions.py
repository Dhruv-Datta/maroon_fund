"""
Future Predictions
==================

This script runs future predictions for the next N days.
Make sure to run 1_Train_Models.py first to train the models.
"""

import pandas as pd
import numpy as np
import joblib
import os, csv
import sys
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from ta import momentum
from scipy.stats import entropy
import warnings

current_script_dir = os.path.dirname(os.path.abspath(__file__))
relative_utilities_path = os.path.join(current_script_dir, '../x_utilities')
utilities_path = os.path.abspath(relative_utilities_path)

if utilities_path not in sys.path:
    sys.path.append(utilities_path)
warnings.filterwarnings('ignore')

# read what the user wrote
with open(os.path.abspath('./combined_buy_sell_model/utils.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile)
    reader_indexed = []
    for i in reader:
        reader_indexed.append(i)

start_date = reader_indexed[1][1]
end_date   = reader_indexed[1][2]
t = reader_indexed[1][0]

ticker = t

start_date = reader_indexed[1][1]
end_date   = reader_indexed[1][2]
t = reader_indexed[1][0]
daysa = input('How many days in advance to predict? Default 60: ')
if daysa == '':
    daysa = 60
else:
    daysa = int(daysa)

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

def run_future_predictions(ticker=t, days_ahead=daysa):
    """Run future predictions for the next N days"""
    print("="*60)
    print("FUTURE PREDICTIONS")
    print("="*60)
    
    # Get recent data for feature computation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=daysa + 100)  # Extra data for features
    
    print(f"Fetching data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    df = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    df = compute_buy_side_features(df)
    df = compute_sell_side_features(df)
    
    # Prepare features
    buy_features = [
        'Low', 'Close', 'Volatility', 'MA_25', 'MA_100', 'RSI_14', 'Month_Growth_Rate',
        'STOCH_%K', 'STOCH_%D', 'Vol_zscore', 'Ulcer_Index', 'Return_Entropy_50d',
        'VIX', 'Growth_Since_Last_Bottom', 'Days_Since_Last_Bottom', 'Week_Growth_Rate'
    ]
    
    sell_features = [
        'Close', 'Volatility', 'MA_25', 'MA_100',
        'Decline_Since_Last_Peak', 'Days_Since_Last_Peak',
        'One_Week_Growth', 'RSI', 'Price_to_MA25', 'Price_to_MA100'
    ]
    
    # Ensure all features exist
    for feature in buy_features:
        if feature not in df.columns:
            df[feature] = 0
    
    for feature in sell_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Make predictions
    X_buy = df[buy_features].fillna(0)
    X_sell = df[sell_features].fillna(0)
    
    df['Dip_Probability'] = buy_model.predict_proba(X_buy)[:, 1]
    df['Predicted_Dip'] = df['Dip_Probability'] > 0.9
    
    df['Sell_Probability'] = sell_model.predict_proba(X_sell)[:, 1]
    df['Sell_Signal'] = df['Sell_Probability'] > 0.9
    
    # Focus on recent predictions
    recent_df = df.tail(days_ahead).copy()
    
    print(f"\n🔮 FUTURE PREDICTIONS FOR {ticker.upper()}")
    print(f"Analysis period: {recent_df['Date'].min().strftime('%Y-%m-%d')} to {recent_df['Date'].max().strftime('%Y-%m-%d')}")
    
    buy_signals = recent_df[recent_df['Predicted_Dip']]
    sell_signals = recent_df[recent_df['Sell_Signal']]
    
    print(f"\n📈 BUY SIGNALS:")
    if len(buy_signals) > 0:
        for _, signal in buy_signals.iterrows():
            print(f"  {signal['Date'].strftime('%Y-%m-%d')}: Price ${signal['Close']:.2f}, Prob: {signal['Dip_Probability']:.3f}")
    else:
        print("  No buy signals predicted")
    
    print(f"\n📉 SELL SIGNALS:")
    if len(sell_signals) > 0:
        for _, signal in sell_signals.iterrows():
            print(f"  {signal['Date'].strftime('%Y-%m-%d')}: Price ${signal['Close']:.2f}, Prob: {signal['Sell_Probability']:.3f}")
    else:
        print("  No sell signals predicted")
    
    # Create visualization
    print("\n📊 Creating future predictions visualization...")
    fig = go.Figure()
    
    # Add price line (full period for context)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue'),
        hoverinfo='skip'
    ))
    
    # Highlight recent period
    fig.add_vrect(
        x0=recent_df['Date'].min(),
        x1=recent_df['Date'].max(),
        fillcolor="yellow",
        opacity=0.1,
        layer="below",
        line_width=0,
        annotation_text="Prediction Period"
    )
    
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
    
    fig.update_layout(
        title=f"Future Predictions - {ticker.upper()} (Next {days_ahead} Days)",
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
    
    fig.show()
    
    # Save results
    output_file = os.path.join(model_dir, f"future_predictions_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv")
    recent_df.to_csv(output_file, index=False)
    print(f"\n💾 Future predictions saved to: {output_file}")
    
    return recent_df

if __name__ == "__main__":
    # Default parameters - modify as needed
    ticker = t
    days_ahead = daysa
    
    future_data = run_future_predictions(ticker, days_ahead)
    
    print("\n✅ Future predictions complete!")
