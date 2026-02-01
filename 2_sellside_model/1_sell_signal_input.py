import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.dates as mdates
from mpl_interactions import zoom_factory, panhandler

import csv
import sys, os

# read what the user wrote
with open(os.path.abspath('./x_utilities/utils.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile)
    reader_indexed = []
    for i in reader:
        reader_indexed.append(i)

start_date = reader_indexed[1][1]
end_date   = reader_indexed[1][2]
t = reader_indexed[1][0]

ticker = t

#Dhrubs code
class PeakSelector:
    def __init__(self, data):
        self.data = data.copy()
        self.data['Date'] = pd.to_datetime(self.data['Date']).dt.tz_localize(None)
        self.peaks = []
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.scatter = None
        self.setup_plot()

    def setup_plot(self):
        self.ax.plot(self.data['Date'], self.data['Close'], label='Close Price')
        self.ax.set_title('Select Peaks on Stock Chart (Use scroll to zoom, right-click to pan)')
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
        if event.inaxes == self.ax and event.button == 1:  # Left click only
            x = event.xdata
            if x is not None:
                date = pd.Timestamp(mdates.num2date(x)).tz_localize(None)
                closest_date = min(self.data['Date'], key=lambda d: abs(d - date))
                price = self.data.loc[self.data['Date'] == closest_date, 'Close'].values[0]
                
                self.peaks.append((closest_date, price))
                if self.scatter:
                    self.scatter.remove()
                self.scatter = self.ax.scatter(*zip(*self.peaks), color='green', zorder=5)
                plt.draw()

    def on_undo(self, event):
        if self.peaks:
            self.peaks.pop()
            if self.scatter:
                self.scatter.remove()
            if self.peaks:
                self.scatter = self.ax.scatter(*zip(*self.peaks), color='green', zorder=5)
            else:
                self.scatter = None
            plt.draw()

    def on_done(self, event):
        plt.close()

    def get_peaks(self):
        return pd.DataFrame(self.peaks, columns=['Date', 'Price'])

# Fetch stock data
stock = yf.Ticker(t)
stock_data = stock.history(start=start_date, end=end_date).reset_index()
stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)

# today's data
today_data = stock.history(period="1d").reset_index()

if not today_data.empty:
    today_data['Date'] = pd.to_datetime(today_data['Date']).dt.tz_localize(None)
    today_data = today_data[['Date', 'Close', 'High', 'Low', 'Open']]

    # Append today's data
    stock_data = pd.concat([stock_data, today_data], ignore_index=True)

# Calculate indicators
stock_data['MA_25'] = stock_data['Close'].rolling(window=25, min_periods=1).mean()
stock_data['MA_100'] = stock_data['Close'].rolling(window=100, min_periods=1).mean()
stock_data['Volatility'] = (stock_data['High'] - stock_data['Low']) / stock_data['Open']

# Calculate Days_Since_Last_Peak
last_peak_index = 0
stock_data['Days_Since_Last_Peak'] = 0  # Initialize column

for i in range(1, len(stock_data)):
    if stock_data.loc[i, 'Close'] > stock_data.loc[last_peak_index, 'Close']:
        last_peak_index = i  # Update peak index
    
    stock_data.loc[i, 'Days_Since_Last_Peak'] = (stock_data.loc[i, 'Date'] - stock_data.loc[last_peak_index, 'Date']).days

# Decline Since Last Peak
last_peak_price = stock_data.loc[last_peak_index, 'Close']
stock_data['Decline_Since_Last_Peak'] = (stock_data['Close'] - last_peak_price) / last_peak_price

# One-Week Growth
stock_data['One_Week_Growth'] = stock_data['Close'].pct_change(periods=5)

# RSI 14-day
delta = stock_data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
stock_data['RSI'] = 100 - (100 / (1 + rs))
stock_data['RSI'].fillna(50, inplace=True)

# Price vs MA Ratios
stock_data['Price_to_MA25'] = stock_data['Close'] / stock_data['MA_25']
stock_data['Price_to_MA100'] = stock_data['Close'] / stock_data['MA_100']

stock_data['Target'] = 0  

# Reorder features
column_order = [
    "Date", "Close", "Volatility", "MA_25", "MA_100",
    "Decline_Since_Last_Peak", "Days_Since_Last_Peak", "One_Week_Growth", 
    "RSI", "Price_to_MA25", "Price_to_MA100", "Target"
]
cleaned_data = stock_data[column_order]

# select peaks
selector = PeakSelector(cleaned_data)
plt.show()

user_peaks = selector.get_peaks()

# Mark Selected Peaks
for _, row in user_peaks.iterrows():
    peak_date = pd.Timestamp(row['Date']).tz_localize(None)
    closest_date = min(cleaned_data['Date'], key=lambda d: abs(d - peak_date))
    cleaned_data.loc[cleaned_data['Date'] == closest_date, 'Target'] = 1


print("Updated Stock Data with User-Selected Peaks (First 10 rows):\n", cleaned_data.head(10))

file_path = f"2_sellside_model/sell_signals/{t}_sell_signals.csv"
cleaned_data.to_csv(file_path, index=False)

print(f"Today's data added and saved to '{file_path}'")

# Plot the selected peaks
plt.figure(figsize=(15, 8))
plt.plot(cleaned_data['Date'], cleaned_data['Close'], label='Close Price')
plt.scatter(cleaned_data[cleaned_data['Target'] == 1]['Date'], 
            cleaned_data[cleaned_data['Target'] == 1]['Close'], 
            color='green', label='User-selected Peaks', zorder=5)
plt.title('Stock with Confirmed User-selected Peaks (Sell Points)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()