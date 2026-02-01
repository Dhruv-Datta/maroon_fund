import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1) Load your enriched dataset
df = pd.read_csv('training_input.csv', parse_dates=['Date'])

# 2) List of features to overlay (add/remove as needed)
features = [
    'Close',
    'STOCH_%K',
    'STOCH_%D',
    'MA_25',
    'MA_100',
    'Month_Growth_Rate',
    'Vol_zscore',
    'Ulcer_Index',
    'Return_Entropy_50d',
    'VIX',
    'Growth_Since_Last_Bottom',
    'Days_Since_Last_Bottom',
    'Week_Growth_Rate',
    'RSI_14'
]

# Optional: Normalize all selected features between 0 and 1
normalize = True

if normalize:
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
else:
    df_scaled = df.copy()

# 3) Plot them all together
plt.figure(figsize=(15, 8))
for feat in features:
    plt.plot(df_scaled['Date'], df_scaled[feat], label=feat)

plt.title("Normalized Dip-Detection Indicators Over Time" if normalize else "Raw Dip-Detection Indicators Over Time")
plt.xlabel('Date')
plt.ylabel('Indicator Value (Normalized)' if normalize else 'Indicator Value')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# 4) Display
plt.show()