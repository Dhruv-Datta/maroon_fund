import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button
from mpl_interactions import zoom_factory, panhandler
from ta import volume
from scipy.stats import entropy


class DipSelector:
    def __init__(self, data):
        self.data = data.copy()
        self.data['Date'] = pd.to_datetime(self.data['Date']).dt.tz_localize(None)
        self.dips = []
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.scatter = None
        self.done_button = None
        self.undo_button = None
        self.setup_plot()

    def setup_plot(self):
        self.ax.plot(self.data['Date'], self.data['Close'], label='Close Price')
        self.ax.set_title('Select Dips on Stock Chart (Use scroll to zoom, right-click to pan)')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price')
        self.ax.legend()
        plt.xticks(rotation=45)

        # Create and store buttons to prevent garbage collection
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
        plt.close(self.fig)

    def get_dips(self):
        return pd.DataFrame(self.dips, columns=['Date', 'Price'])


def calc_entropy(series):
    arr = series[~np.isnan(series)]
    if arr.size == 0:
        return 0.0
    counts, _ = np.histogram(arr, bins=10)
    return entropy(counts + 1)


def slope(x):
    y = np.asarray(x)
    n = y.size
    if n < 2 or np.allclose(y, y[0]):
        return 0.0
    idx = np.arange(n)
    return np.polyfit(idx, y, 1)[0]


def fetch_and_compute(ticker_symbol, start_date, end_date):
    df = yf.Ticker(ticker_symbol).history(start=start_date, end=end_date).reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Base features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = (df['High'] - df['Low']) / df['Open']
    df['MA_25'] = df['Close'].rolling(25, min_periods=1).mean()
    df['MA_100'] = df['Close'].rolling(100, min_periods=1).mean()
    df['Month_Growth_Rate'] = df['Close'].pct_change(21)

    # Stochastic
    low50 = df['Low'].rolling(50, min_periods=1).min()
    high50 = df['High'].rolling(50, min_periods=1).max()
    df['STOCH_%K'] = (df['Close'] - low50) / (high50 - low50) * 100
    df['STOCH_%D'] = df['STOCH_%K'].rolling(20, min_periods=1).mean()

    # Volatility z-score
    df['Roll_Vol_20d'] = df['Daily_Return'].rolling(20, min_periods=1).std()
    vol_mean = df['Roll_Vol_20d'].rolling(126, min_periods=1).mean()
    vol_std = df['Roll_Vol_20d'].rolling(126, min_periods=1).std()
    df['Vol_zscore'] = (df['Roll_Vol_20d'] - vol_mean) / vol_std

    # Ulcer Index
    ui = []
    for i in range(len(df)):
        window = df['Close'].iloc[max(0, i-13):i+1]
        peak = window.max()
        dd = (peak - window) / peak
        ui.append(np.sqrt((dd**2).mean()))
    df['Ulcer_Index'] = ui

    # Entropy
    df['Return_Entropy_50d'] = df['Daily_Return'].rolling(50, min_periods=1).apply(calc_entropy, raw=False)

    # VIX merge
    vix = yf.Ticker("^VIX").history(start=start_date, end=end_date).reset_index()
    vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
    vix = vix[['Date', 'Close']].rename(columns={'Close': 'VIX'})
    df = df.merge(vix, on='Date', how='left')

    # Dip-related features
    df['Growth_Since_Last_Bottom'] = 0.0
    df['Days_Since_Last_Bottom'] = 0
    df['Week_Growth_Rate'] = df['Close'].pct_change(5)

    last_bot = 0
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] < df.loc[last_bot, 'Close']:
            last_bot = i
        df.at[i, 'Growth_Since_Last_Bottom'] = (df.at[i, 'Close'] - df.at[last_bot, 'Close']) / df.at[last_bot, 'Close']
        df.at[i, 'Days_Since_Last_Bottom'] = i - last_bot

    df.bfill(inplace=True)
    return df



def prepare_data(df):
    drop_cols = ['Open', 'High', 'Low', 'Volume', 'Daily_Return', 'Roll_Vol_20d', 'Dividends', 'Stock Splits', 'Capital Gains']
    return df.drop(columns=drop_cols, errors='ignore').reset_index(drop=True)



if __name__ == '__main__':
    # Parameters
    ticker = "NVDA"
    start_date = "2024-05-24"
    end_date = "2025-09-14"

    # Data preparation
    raw_df = fetch_and_compute(ticker, start_date, end_date)
    cleaned_df = prepare_data(raw_df)

    # Dip labeling
    selector = DipSelector(cleaned_df)
    plt.show()
    dips_df = selector.get_dips()
    cleaned_df['Target'] = 0
    for _, row in dips_df.iterrows():
        date = pd.Timestamp(row['Date']).tz_localize(None)
        closest = min(cleaned_df['Date'], key=lambda d: abs(d - date))
        cleaned_df.loc[cleaned_df['Date'] == closest, 'Target'] = 1

    # Save for training/analysis
    cleaned_df.to_csv('test_input.csv', index=False)
    print('Saved to test_input.csv')
