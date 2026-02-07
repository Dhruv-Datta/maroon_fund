import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.dates as mdates
from mpl_interactions import zoom_factory, panhandler
from features import calculate_features

class PeakSelector:
    def __init__(self, data):
        self.data = data.copy()
        self.peaks = []
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.scatter = None
        self.setup_plot()

    def setup_plot(self):
        self.ax.plot(self.data['Date'], self.data['Close'], label='Close Price')
        self.ax.set_title('Select Sell Points (Peaks) | Scroll to Zoom, Right-Click to Pan')
        self.ax.set_xlabel('Date'); self.ax.set_ylabel('Price')
        plt.xticks(rotation=45)
        
        # Add UI Buttons
        self.done_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.done_button = Button(self.done_button_ax, 'Done')
        self.done_button.on_clicked(self.on_done)
        self.undo_button_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.undo_button = Button(self.undo_button_ax, 'Undo')
        self.undo_button.on_clicked(self.on_undo)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        zoom_factory(self.ax); panhandler(self.fig)

    def on_click(self, event):
        if event.inaxes == self.ax and event.button == 1:
            date = pd.Timestamp(mdates.num2date(event.xdata)).tz_localize(None)
            closest_date = self.data['Date'].iloc[self.data['Date'].searchsorted(date)]
            price = self.data.loc[self.data['Date'] == closest_date, 'Close'].values[0]
            self.peaks.append((closest_date, price))
            self._update_scatter()
            
    def _update_scatter(self):
        if self.scatter: self.scatter.remove()
        if self.peaks:
            dates, prices = zip(*self.peaks)
            self.scatter = self.ax.scatter(dates, prices, color='green', zorder=5, s=80, label="Selected Sells")
        else: self.scatter = None
        plt.draw()

    def on_undo(self, event):
        if self.peaks: self.peaks.pop(); self._update_scatter()

    def on_done(self, event): plt.close()

    def get_peaks(self): return pd.DataFrame(self.peaks, columns=['Date', 'Price'])

if __name__ == "__main__":
    TICKER = "NVDA"
    START_DATE = "2024-05-24"
    END_DATE = "2025-02-14"
    OUTPUT_CSV = "nvda_sell_signals.csv"

    stock_data = yf.Ticker(TICKER).history(start=START_DATE, end=END_DATE).reset_index()
    featured_data = calculate_features(stock_data)
    
    featured_data['Target'] = 0
    final_data = featured_data.drop(columns=['Open', 'High', 'Low', 'Dividends', 'Stock Splits'], errors='ignore')

    selector = PeakSelector(final_data)
    plt.show()

    user_peaks = selector.get_peaks()
    if not user_peaks.empty:
        peak_dates = user_peaks['Date'].tolist()
        final_data.loc[final_data['Date'].isin(peak_dates), 'Target'] = 1

    final_data.to_csv(OUTPUT_CSV, index=False)
    print(f"\nLabeled training data saved to '{OUTPUT_CSV}'")
    print(f"Total sell points selected: {final_data['Target'].sum()}")