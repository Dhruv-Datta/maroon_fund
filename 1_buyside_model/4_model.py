import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory, panhandler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import joblib
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ------------------ Load and Prepare Data ------------------
data = pd.read_csv("training_input.csv")
features = [col for col in data.columns if col not in ['Date', 'Target']]
X = data[features]
y = data['Target']
data['Date'] = pd.to_datetime(data['Date'])  # Ensure date is datetime

# ------------------ Compute Class Imbalance Weight ------------------
neg, pos = Counter(y).values()
scale = neg / pos

# ------------------ Build Pipeline (No SMOTE) ------------------
pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale,
        max_depth=3,
        n_estimators=100,
        min_child_weight=5,        # less likely to split on small noisy patterns
        subsample=0.7,             # makes trees less sensitive to outliers
        colsample_bytree=0.7       # uses only part of features per tree
    ))
])


param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5]
}

tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring='f1',
    n_jobs=-1
)

# ------------------ Fit Model ------------------
grid_search.fit(X, y)
model = grid_search.best_estimator_

# ------------------ Predict on Full Dataset ------------------
data['Dip_Probability'] = model.predict_proba(X)[:, 1]
data['Predicted_Dip'] = data['Dip_Probability'] > 0.75

# ------------------ Plot Dips ------------------
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(data['Date'], data['Close'], label='Close Price', color='blue')
predicted_dips = data[data['Predicted_Dip']]
ax.scatter(predicted_dips['Date'], predicted_dips['Close'], color='red', label='Predicted Dip (>75%)', zorder=5)
ax.set_title('Stock Price with Predicted Dips')
ax.set_xlabel('Date'); ax.set_ylabel('Price'); ax.legend()
plt.xticks(rotation=45); plt.tight_layout()
zoom_factory(ax); panhandler(fig)
plt.show()

# ------------------ Evaluate Model on Last Fold ------------------
last_train_idx, last_test_idx = list(tscv.split(X))[-1]
X_test = X.iloc[last_test_idx]
y_test = y.iloc[last_test_idx]
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = y_pred_proba > 0.75

# ------------------ Metrics ------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nRecent Data Points:")
print(data.tail(5)[['Date', 'Close'] + features + ['Dip_Probability']])
print("\nNote: Model considers it a dip if probability > 0.75")

# ------------------ Extra Metrics ------------------
print("\n" + "="*50)
print("Comprehensive Model Performance Report")
print("="*50)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nPrecision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# ------------------ Feature Importance ------------------
feature_importance = model.named_steps['xgb'].feature_importances_
sorted_fi = sorted(zip(features, feature_importance), key=lambda x: x[1], reverse=True)
print("\nFeature Importance:")
for f, imp in sorted_fi:
    print(f"{f}: {imp:.4f}")

# ------------------ Prediction for Most Recent Day ------------------
most_recent = data.iloc[-1]
recent_features = most_recent[features].values.reshape(1, -1)
recent_prob = model.predict_proba(recent_features)[0, 1]
is_dip = recent_prob > 0.75

print("\n" + "="*50)
print("Prediction for the Most Recent Day")
print("="*50)
print(f"Date: {most_recent['Date']}")
print(f"Close Price: {most_recent['Close']:.2f}")
print(f"Dip Probability: {recent_prob:.4f}")
print(f"Is it a dip? {'Yes' if is_dip else 'No'}")
print(f"Confidence: {'High' if abs(recent_prob - 0.5) > 0.3 else 'Low'}")

# ------------------ Save Model and Visualize ------------------
joblib.dump(model, 'xgboost_model.joblib')
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.title('Feature Importance')
plt.show()

booster = model.named_steps['xgb'].get_booster()
booster.feature_names = features
plt.figure(figsize=(20, 12))
xgb.plot_tree(booster, num_trees=0, rankdir='LR')
plt.title('XGBoost Tree (With Feature Names)')
plt.show()

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
