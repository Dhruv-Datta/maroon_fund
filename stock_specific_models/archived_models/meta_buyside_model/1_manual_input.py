import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.dates as mdates
from mpl_interactions import zoom_factory, panhandler
from ta import momentum, volume, volatility
from scipy.stats import entropy

class DipSelector:
    def __init__(self, data):
        self.data = data.copy()
        self.data['Date'] = pd.to_datetime(self.data['Date']).dt.tz_localize(None)
        self.dips = []
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.scatter = None
        self.setup_plot()

    def setup_plot(self):
        self.ax.plot(self.data['Date'], self.data['Close'], label='Close Price')
        self.ax.set_title('Select Dips on Stock Chart (Use scroll to zoom, right-click to pan)')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price')
        self.ax.legend()
        plt.xticks(rotation=45)

        self.done_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.done_button = Button(self.done_button_ax, 'Done')
        self.done_button.on_clicked(self.on_done)

        self.undo_button_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.undo_button = Button(self.undo_button_ax, 'Undo')
        self.undo_button.on_clicked(self.on_undo)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        zoom_factory(self.ax)
        panhandler(self.fig)

    def on_click(self, event):
        if event.inaxes == self.ax and event.button == 1:
            x = event.xdata
            if x is not None:
                date = pd.Timestamp(mdates.num2date(x)).tz_localize(None)
                closest = min(self.data['Date'], key=lambda d: abs(d - date))
                price = self.data.loc[self.data['Date'] == closest, 'Close'].iloc[0]
                self.dips.append((closest, price))
                if self.scatter:
                    self.scatter.remove()
                self.scatter = self.ax.scatter(*zip(*self.dips), color='red', zorder=5)
                plt.draw()

    def on_undo(self, event):
        if self.dips:
            self.dips.pop()
            if self.scatter:
                self.scatter.remove()
            if self.dips:
                self.scatter = self.ax.scatter(*zip(*self.dips), color='red', zorder=5)
            else:
                self.scatter = None
            plt.draw()

    def on_done(self, event):
        plt.close()

    def get_dips(self):
        return pd.DataFrame(self.dips, columns=['Date', 'Price'])

# ---- Main script ----

start_date = "2022-11-03"
end_date   = "2025-01-17"

# Fetch SPY data
ticker = yf.Ticker("META")
df = ticker.history(start=start_date, end=end_date).reset_index()
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

# Base features
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = (df['High'] - df['Low']) / df['Open']
df['MA_25'] = df['Close'].rolling(window=25,  min_periods=1).mean()
df['MA_100'] = df['Close'].rolling(window=100, min_periods=1).mean()

def compute_growth_rate(i, closes):
    if i < 12:
        return (closes[i] - closes[0]) / closes[0]
    else:
        return (closes[i] - closes[i - 12]) / closes[i - 12]

df['Month_Growth_Rate'] = df['Close'].rolling(window=len(df), min_periods=1).apply(lambda x: compute_growth_rate(len(x)-1, x.values), raw=False)

# 1) Stochastic %K and %D
low50 = df['Low'].rolling(window=50, min_periods=1).min()
high50 = df['High'].rolling(window=50, min_periods=1).max()
df['STOCH_%K'] = (df['Close'] - low50) / (high50 - low50) * 100
df['STOCH_%D'] = df['STOCH_%K'].rolling(window=20, min_periods=1).mean()

# 2) Volatility z-score (20d vs 126d)
df['Roll_Vol_20d'] = df['Daily_Return'].rolling(window=20, min_periods=1).std()
vol_mean = df['Roll_Vol_20d'].rolling(window=126, min_periods=1).mean()
vol_std  = df['Roll_Vol_20d'].rolling(window=126, min_periods=1).std()
df['Vol_zscore'] = (df['Roll_Vol_20d'] - vol_mean) / vol_std

# 3) Manual Ulcer Index (30-day RMS drawdown)
ui = []
for i in range(len(df)):
    window = df['Close'].iloc[max(0, i-13) : i+1]
    peak   = window.max()
    dd     = (peak - window) / peak
    ui.append(np.sqrt((dd**2).mean()))
df['Ulcer_Index'] = ui

# 4) Return entropy (50-day)
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

# 5) VIX level
vix = yf.Ticker("^VIX").history(start=start_date, end=end_date).reset_index()
vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
vix = vix[['Date','Close']].rename(columns={'Close':'VIX'})
df = df.merge(vix, on='Date', how='left')


# Dip-related original features
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


df['Target'] = 0

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
drop_cols = ['Open', 'High', 'Low', 'Volume', 'Daily_Return', 'Roll_Vol_20d', 'OBV', 'Dividends', 'Stock Splits', 'Capital Gains']
cleaned = df.drop(columns=drop_cols, errors='ignore')

# Launch selector UI
selector = DipSelector(cleaned)
plt.show()

# Mark selected dips
user_dips = selector.get_dips()
cleaned['Target'] = 0
for _, row in user_dips.iterrows():
    dd = pd.Timestamp(row['Date']).tz_localize(None)
    closest = min(cleaned['Date'], key=lambda d: abs(d - dd))
    cleaned.loc[cleaned['Date'] == closest, 'Target'] = 1

# Inspect & save
print(cleaned.head(10))
cleaned.to_csv('training_input.csv', index=False)

# Final plot
plt.figure(figsize=(15,8))
plt.plot(cleaned['Date'], cleaned['Close'], label='Close Price')
plt.scatter(
    cleaned.loc[cleaned['Target']==1, 'Date'],
    cleaned.loc[cleaned['Target']==1, 'Close'],
    color='red', label='User-selected Dips', zorder=5
)
plt.title('Stock with Confirmed User-selected Dips')
plt.xlabel('Date'); plt.ylabel('Price')
plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
plt.show()
