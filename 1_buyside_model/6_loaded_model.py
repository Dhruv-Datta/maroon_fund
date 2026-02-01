import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Load data and model
data = pd.read_csv('test_input.csv')
features = [col for col in data.columns if col not in ['Date', 'Target']]
X = data[features]
y = data['Target']

model = joblib.load('xgboost_model.joblib')
data['Dip_Probability'] = model.predict_proba(X)[:, 1]
data['Predicted_Dip'] = data['Dip_Probability'] > 0.9

# Convert Date to datetime (for Plotly hover and axis formatting)
data['Date'] = pd.to_datetime(data['Date'])

# Plot using Plotly
fig = go.Figure()

# Add price line
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='blue'),
    hoverinfo='skip'
))

# Add dip points with hover tooltips
fig.add_trace(go.Scatter(
    x=data.loc[data['Predicted_Dip'], 'Date'],
    y=data.loc[data['Predicted_Dip'], 'Close'],
    mode='markers+text',
    name='Predicted Dip',
    marker=dict(color='red', size=8),
    text=[f"Prob: {p:.2f}" for p in data.loc[data['Predicted_Dip'], 'Dip_Probability']],
    hovertemplate=
        "Date: %{x}<br>" +
        "Price: %{y:.2f}<br>" +
        "Dip Probability: %{text}<extra></extra>"
))

# Optional: Add hover for all dates (not just predicted dips)
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    mode='markers',
    name='All Points',
    marker=dict(color='rgba(0,0,255,0)', size=5),  # Invisible markers for hover
    text=[f"{p:.2f}" for p in data['Dip_Probability']],
    hovertemplate=
        "Date: %{x}<br>" +
        "Price: %{y:.2f}<br>" +
        "Dip Probability: %{text}<extra></extra>",
    showlegend=False
))

# Update layout
fig.update_layout(
    title="Stock Price with Predicted Dips",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="closest",
    template="plotly_white"
)

fig.show()

# ==================== FEATURE ANALYSIS ====================
print("\n" + "="*60)
print("FEATURE ANALYSIS & PREDICTION DRIVERS")
print("="*60)

# 1. Feature Importance Analysis
xgb_model = model.named_steps['xgb']
feature_importance = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
print("-" * 40)
for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:<25} {row['Importance']:.4f}")

# 2. Feature Importance Visualization
fig_importance = px.bar(
    feature_importance_df.head(15), 
    x='Importance', 
    y='Feature',
    orientation='h',
    title='Feature Importance (Top 15)',
    color='Importance',
    color_continuous_scale='viridis'
)
fig_importance.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
fig_importance.show()

# 3. Analyze predictions vs actual features
predicted_dips = data[data['Predicted_Dip']]
print(f"\n🎯 PREDICTION SUMMARY:")
print(f"Total predictions: {len(predicted_dips)}")
print(f"Prediction rate: {len(predicted_dips)/len(data)*100:.2f}%")
print(f"Average probability: {predicted_dips['Dip_Probability'].mean():.4f}")

# 4. Feature statistics for predicted dips vs all data
print(f"\n📈 FEATURE COMPARISON: Predicted Dips vs All Data")
print("-" * 60)
comparison_features = feature_importance_df.head(8)['Feature'].tolist()

for feature in comparison_features:
    if feature in data.columns:
        dip_mean = predicted_dips[feature].mean()
        all_mean = data[feature].mean()
        dip_std = predicted_dips[feature].std()
        all_std = data[feature].std()
        
        print(f"\n{feature}:")
        print(f"  Predicted Dips: {dip_mean:.4f} ± {dip_std:.4f}")
        print(f"  All Data:       {all_mean:.4f} ± {all_std:.4f}")
        print(f"  Difference:     {dip_mean - all_mean:+.4f}")

# 5. Create feature correlation heatmap for predicted dips
if len(predicted_dips) > 1:
    print(f"\n🔗 FEATURE CORRELATIONS (Predicted Dips Only)")
    print("-" * 50)
    
    # Select top features for correlation analysis
    top_features = feature_importance_df.head(10)['Feature'].tolist()
    available_features = [f for f in top_features if f in predicted_dips.columns]
    
    if len(available_features) > 1:
        corr_matrix = predicted_dips[available_features].corr()
        
        # Create correlation heatmap
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix (Predicted Dips)",
            color_continuous_scale='RdBu_r'
        )
        fig_corr.update_layout(height=500)
        fig_corr.show()

# 6. Time series of top features with predictions
print(f"\n📅 TIME SERIES ANALYSIS")
print("-" * 30)

# Create subplot for top 4 features
top_4_features = feature_importance_df.head(4)['Feature'].tolist()
available_top_4 = [f for f in top_4_features if f in data.columns]

if len(available_top_4) >= 2:
    fig_ts = make_subplots(
        rows=2, cols=2,
        subplot_titles=available_top_4[:4],
        vertical_spacing=0.1
    )
    
    for i, feature in enumerate(available_top_4[:4]):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Add feature line
        fig_ts.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data[feature],
                mode='lines',
                name=feature,
                line=dict(color='blue', width=1)
            ),
            row=row, col=col
        )
        
        # Highlight predicted dips
        if len(predicted_dips) > 0:
            fig_ts.add_trace(
                go.Scatter(
                    x=predicted_dips['Date'],
                    y=predicted_dips[feature],
                    mode='markers',
                    name=f'{feature} (Dips)',
                    marker=dict(color='red', size=6),
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig_ts.update_layout(height=800, title_text="Top Features Over Time with Predicted Dips")
    fig_ts.show()

print(f"\n✅ Analysis complete! Check the interactive plots above for detailed insights.")

# ==================== MODEL TESTING ACROSS PERIODS ====================
print(f"\n" + "="*60)
print("MODEL TESTING ACROSS DIFFERENT TIME PERIODS")
print("="*60)

def fetch_and_compute_features(ticker_symbol, start_date, end_date):
    """Fetch stock data and compute all features used in the model"""
    import yfinance as yf
    from ta import momentum
    from scipy.stats import entropy
    
    df = yf.Ticker(ticker_symbol).history(start=start_date, end=end_date).reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

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
    vix = yf.Ticker("^VIX").history(start=start_date, end=end_date).reset_index()
    vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
    vix = vix[['Date','Close']].rename(columns={'Close':'VIX'})
    df = df.merge(vix, on='Date', how='left')

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

    # Drop raw columns
    drop_cols = ['Open', 'High', 'Volume', 'Daily_Return', 'Roll_Vol_20d', 'Dividends', 'Stock Splits', 'Capital Gains']
    cleaned = df.drop(columns=drop_cols, errors='ignore')
    
    return cleaned

def test_model_period(ticker, start_date, end_date, period_name, model, threshold=0.9):
    """Test model on a specific time period"""
    print(f"\n{'='*50}")
    print(f"TESTING: {period_name}")
    print(f"Ticker: {ticker} | Period: {start_date} to {end_date}")
    print(f"{'='*50}")
    
    try:
        # Fetch and prepare data
        test_data = fetch_and_compute_features(ticker, start_date, end_date)
        
        if len(test_data) < 50:  # Need minimum data for features
            print(f"❌ Insufficient data: {len(test_data)} days")
            return None
            
        # Prepare features
        test_features = [col for col in test_data.columns if col not in ['Date', 'Target']]
        X_test = test_data[test_features]
        
        # Make predictions
        test_data['Dip_Probability'] = model.predict_proba(X_test)[:, 1]
        test_data['Predicted_Dip'] = test_data['Dip_Probability'] > threshold
        
        # Calculate metrics
        predicted_dips = test_data[test_data['Predicted_Dip']]
        prediction_rate = len(predicted_dips) / len(test_data) * 100
        avg_prob = predicted_dips['Dip_Probability'].mean() if len(predicted_dips) > 0 else 0
        
        # Price performance analysis
        price_start = test_data['Close'].iloc[0]
        price_end = test_data['Close'].iloc[-1]
        total_return = (price_end - price_start) / price_start * 100
        
        # Volatility analysis
        daily_returns = test_data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized
        
        print(f"📊 RESULTS:")
        print(f"  Data points: {len(test_data)}")
        print(f"  Predictions: {len(predicted_dips)} ({prediction_rate:.2f}%)")
        print(f"  Avg probability: {avg_prob:.4f}")
        print(f"  Total return: {total_return:+.2f}%")
        print(f"  Volatility: {volatility:.2f}%")
        
        # Analyze prediction timing
        if len(predicted_dips) > 0:
            print(f"  Prediction dates: {predicted_dips['Date'].min().strftime('%Y-%m-%d')} to {predicted_dips['Date'].max().strftime('%Y-%m-%d')}")
            
            # Check if predictions align with actual dips (local minima)
            predicted_dates = predicted_dips['Date'].tolist()
            actual_dips = []
            
            for i in range(1, len(test_data)-1):
                if (test_data.iloc[i]['Close'] < test_data.iloc[i-1]['Close'] and 
                    test_data.iloc[i]['Close'] < test_data.iloc[i+1]['Close']):
                    actual_dips.append(test_data.iloc[i]['Date'])
            
            # Simple alignment check
            aligned_predictions = 0
            for pred_date in predicted_dates:
                for actual_date in actual_dips:
                    if abs((pred_date - actual_date).days) <= 3:  # Within 3 days
                        aligned_predictions += 1
                        break
            
            alignment_rate = aligned_predictions / len(predicted_dates) * 100 if len(predicted_dates) > 0 else 0
            print(f"  Alignment with actual dips: {alignment_rate:.1f}%")
        
        return {
            'period_name': period_name,
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'data_points': len(test_data),
            'predictions': len(predicted_dips),
            'prediction_rate': prediction_rate,
            'avg_probability': avg_prob,
            'total_return': total_return,
            'volatility': volatility,
            'data': test_data
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

# Define test periods
test_periods = [
    # Historical periods
    ("NVDA", "2020-01-01", "2020-12-31", "2020 (COVID Crash & Recovery)"),
    ("NVDA", "2021-01-01", "2021-12-31", "2021 (AI Boom)"),
    ("NVDA", "2022-01-01", "2022-12-31", "2022 (Tech Crash)"),
    ("NVDA", "2023-01-01", "2023-12-31", "2023 (AI Surge)"),
    
    # Recent periods
    ("NVDA", "2024-01-01", "2024-06-30", "2024 H1"),
    ("NVDA", "2024-07-01", "2024-12-31", "2024 H2"),
    
    # Future prediction (if data available)
    ("NVDA", "2025-01-01", "2025-06-30", "2025 H1 (Future Prediction)"),
    
    # Different tickers for robustness
    ("AAPL", "2023-01-01", "2024-12-31", "AAPL 2023-2024"),
    ("TSLA", "2023-01-01", "2024-12-31", "TSLA 2023-2024"),
    ("MSFT", "2023-01-01", "2024-12-31", "MSFT 2023-2024"),
]

# Run tests
results = []
for ticker, start_date, end_date, period_name in test_periods:
    result = test_model_period(ticker, start_date, end_date, period_name, model)
    if result:
        results.append(result)

# Create summary
print(f"\n{'='*60}")
print("SUMMARY OF ALL TESTS")
print(f"{'='*60}")

if results:
    df_summary = pd.DataFrame([
        {
            'Period': r['period_name'],
            'Ticker': r['ticker'],
            'Predictions': r['predictions'],
            'Prediction Rate (%)': f"{r['prediction_rate']:.2f}",
            'Avg Probability': f"{r['avg_probability']:.3f}",
            'Total Return (%)': f"{r['total_return']:+.2f}",
            'Volatility (%)': f"{r['volatility']:.2f}"
        }
        for r in results
    ])
    
    print(df_summary.to_string(index=False))
    
    # Future prediction analysis
    future_results = [r for r in results if 'Future' in r['period_name']]
    if future_results:
        print(f"\n🔮 FUTURE PREDICTION ANALYSIS:")
        print(f"{'='*40}")
        for result in future_results:
            print(f"Period: {result['period_name']}")
            print(f"Predictions: {result['predictions']} ({result['prediction_rate']:.2f}%)")
            if result['predictions'] > 0:
                print(f"Average confidence: {result['avg_probability']:.3f}")
                print(f"Market conditions: {result['volatility']:.1f}% volatility")

print(f"\n✅ Testing complete! Your model has been tested across multiple periods and tickers.")
