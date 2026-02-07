"""
Debug script to check sell signal probabilities
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load ML models
try:
    import joblib
    import importlib.util
    
    combined_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'combined_buy_sell_model'
    )
    
    spec = importlib.util.spec_from_file_location(
        "loaded_models", 
        os.path.join(combined_model_path, '2_Loaded_Models.py')
    )
    loaded_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_models)
    
    buy_model = loaded_models.buy_model
    sell_model = loaded_models.sell_model
    
    print("Models loaded")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Load test data
test_data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '1_buyside_model',
    'test_input.csv'
)

df = pd.read_csv(test_data_path)
df['Date'] = pd.to_datetime(df['Date'])

print(f"\nData loaded: {len(df)} rows")

# Prepare features for sell model
sell_features = [
    'Close', 'Volatility', 'MA_25', 'MA_100',
    'Decline_Since_Last_Peak', 'Days_Since_Last_Peak',
    'One_Week_Growth', 'RSI', 'Price_to_MA25', 'Price_to_MA100'
]

# Check which features are missing
missing_features = [f for f in sell_features if f not in df.columns]
print(f"\nMissing sell features: {missing_features}")

# Ensure all features exist (fill missing with 0)
for feature in sell_features:
    if feature not in df.columns:
        df[feature] = 0
        print(f"   Added {feature} = 0")

# Check for None/NaN values
X_sell = df[sell_features].fillna(0)
print(f"\nSell feature stats:")
print(X_sell.describe())

# Get sell predictions
df['Sell_Probability'] = sell_model.predict_proba(X_sell)[:, 1]

print(f"\nSELL PROBABILITY ANALYSIS:")
print(f"   Min: {df['Sell_Probability'].min():.4f}")
print(f"   Max: {df['Sell_Probability'].max():.4f}")
print(f"   Mean: {df['Sell_Probability'].mean():.4f}")
print(f"   Median: {df['Sell_Probability'].median():.4f}")
print(f"   Std: {df['Sell_Probability'].std():.4f}")

# Count signals at different thresholds
print(f"\nSELL SIGNALS BY THRESHOLD:")
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
for threshold in thresholds:
    count = (df['Sell_Probability'] >= threshold).sum()
    print(f"   Threshold {threshold:.2f}: {count} signals ({count/len(df)*100:.2f}%)")

# Show top sell probabilities
print(f"\nTOP 10 SELL PROBABILITIES:")
top_sells = df.nlargest(10, 'Sell_Probability')[['Date', 'Close', 'Sell_Probability']]
for _, row in top_sells.iterrows():
    print(f"   {row['Date'].strftime('%Y-%m-%d')}: ${row['Close']:.2f} - Prob: {row['Sell_Probability']:.4f}")

# Check if we need to compute sell side features
print(f"\nCHECKING FEATURE COMPUTATION:")
print("   Need to check if sell side features need to be computed from combined_buy_sell_model")

# Compare with buy probabilities
buy_features = [
    'Low', 'Close', 'Volatility', 'MA_25', 'MA_100', 'RSI_14', 'Month_Growth_Rate',
    'STOCH_%K', 'STOCH_%D', 'Vol_zscore', 'Ulcer_Index', 'Return_Entropy_50d',
    'VIX', 'Growth_Since_Last_Bottom', 'Days_Since_Last_Bottom', 'Week_Growth_Rate'
]

for feature in buy_features:
    if feature not in df.columns:
        df[feature] = 0

X_buy = df[buy_features].fillna(0)
df['Dip_Probability'] = buy_model.predict_proba(X_buy)[:, 1]

print(f"\nBUY vs SELL PROBABILITY COMPARISON:")
print(f"   Buy Prob - Min: {df['Dip_Probability'].min():.4f}, Max: {df['Dip_Probability'].max():.4f}, Mean: {df['Dip_Probability'].mean():.4f}")
print(f"   Sell Prob - Min: {df['Sell_Probability'].min():.4f}, Max: {df['Sell_Probability'].max():.4f}, Mean: {df['Sell_Probability'].mean():.4f}")

# Check if sell features need computation from combined model
print(f"\nRECOMMENDATION:")
if df['Sell_Probability'].max() < 0.9:
    print("   Sell probabilities are low. Possible reasons:")
    print("   1. Sell model features may need to be computed (not in test_input.csv)")
    print("   2. Sell threshold (0.9) might be too high")
    print("   3. Sell model may not be detecting sell signals on this test data")
    print("\n   Try using combined_buy_sell_model/2_Loaded_Models.py to compute sell features")

