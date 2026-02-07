import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from features import calculate_features
from matplotlib.widgets import Button
import matplotlib.dates as mdates
from mpl_interactions import zoom_factory, panhandler

class PeakSelector:
    def __init__(self, data):
        self.data = data.copy(); self.peaks = []
        self.fig, self.ax = plt.subplots(figsize=(15, 8)); self.scatter = None; self.setup_plot()
    def setup_plot(self):
        self.ax.plot(self.data['Date'], self.data['Close'], label='Close Price')
        self.ax.set_title('Select ACTUAL Sell Points for this TEST Period')
        self.ax.set_xlabel('Date'); self.ax.set_ylabel('Price'); plt.xticks(rotation=45)
        self.done_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075]); self.done_button = Button(self.done_button_ax, 'Done')
        self.done_button.on_clicked(self.on_done)
        self.undo_button_ax = plt.axes([0.7, 0.05, 0.1, 0.075]); self.undo_button = Button(self.undo_button_ax, 'Undo')
        self.undo_button.on_clicked(self.on_undo)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        zoom_factory(self.ax); panhandler(self.fig)
    def on_click(self, event):
        if event.inaxes == self.ax and event.button == 1:
            date = pd.Timestamp(mdates.num2date(event.xdata)).tz_localize(None)
            closest_date = self.data['Date'].iloc[self.data['Date'].searchsorted(date)]
            price = self.data.loc[self.data['Date'] == closest_date, 'Close'].values[0]
            self.peaks.append((closest_date, price)); self._update_scatter()
    def _update_scatter(self):
        if self.scatter: self.scatter.remove()
        if self.peaks:
            dates, prices = zip(*self.peaks)
            self.scatter = self.ax.scatter(dates, prices, color='red', marker='x', s=150, zorder=6)
        else: self.scatter = None
        plt.draw()
    def on_undo(self, event):
        if self.peaks: self.peaks.pop(); self._update_scatter()
    def on_done(self, event): plt.close()
    def get_peaks(self): return pd.DataFrame(self.peaks, columns=['Date', 'Price'])

if __name__ == "__main__":
    TICKER = "NVDA"
    MODEL_PATH = "nvda_sell_model.joblib"
    START_DATE = "2024-05-24"
    END_DATE = "2025-06-13"
    PREDICTION_THRESHOLD = 0.50

    model = joblib.load(MODEL_PATH)
    stock_data = yf.Ticker(TICKER).history(start=START_DATE, end=END_DATE).reset_index()
    
    if stock_data.empty:
        print("No data fetched for the test period.")
    else:
        featured_data = calculate_features(stock_data)
        
        model_features = [
            'Volatility', 'MA_25', 'Price_to_MA25', 'Bollinger_Band_Width',
            'ROC_14', 'Stochastic_K', 'RSI', 'One_Week_Growth',
            'Days_Since_Last_Peak', 'Decline_Since_Last_Peak', 'OBV',
            'Bearish_Divergence'
        ]
        
        X_predict = featured_data[model_features]
        
        featured_data['Sell_Probability'] = model.predict_proba(X_predict)[:, 1]
        featured_data['Sell_Signal'] = featured_data['Sell_Probability'] > PREDICTION_THRESHOLD

        selector = PeakSelector(featured_data)
        print("\nPlease label the actual sell points for the test period on the chart...")
        plt.show()

        user_labels = selector.get_peaks()
        featured_data['Target'] = 0
        if not user_labels.empty:
            peak_dates = user_labels['Date'].tolist()
            featured_data.loc[featured_data['Date'].isin(peak_dates), 'Target'] = 1

        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax1.plot(featured_data['Date'], featured_data['Close'], label='Close Price', zorder=1)
        ax1.set_xlabel('Date'); ax1.set_ylabel('Price (USD)')
        predicted_sells = featured_data[featured_data['Sell_Signal']]
        actual_sells = featured_data[featured_data['Target'] == 1]
        ax1.scatter(predicted_sells['Date'], predicted_sells['Close'], color='green', s=150, marker='v', label='Predicted Sell', zorder=5)
        ax1.scatter(actual_sells['Date'], actual_sells['Close'], color='red', s=150, marker='x', label='Actual Sell', zorder=6)
        ax2 = ax1.twinx()
        ax2.fill_between(featured_data['Date'], 0, featured_data['Sell_Probability'], color='green', alpha=0.2, label='Sell Probability')
        ax2.set_ylabel('Sell Probability'); ax2.set_ylim(0, 1)
        fig.suptitle(f'{TICKER} - Prediction vs. Actual Sell Signals', fontsize=16)
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left')
        plt.xticks(rotation=45); fig.tight_layout(); plt.show()
        
        true_positives = len(featured_data[(featured_data['Sell_Signal'] == True) & (featured_data['Target'] == 1)])
        print("\n--- Comparison Summary ---")
        print(f"Date Range: {featured_data['Date'].min().date()} to {featured_data['Date'].max().date()}")
        print(f"Total Predicted Sells: {len(predicted_sells)}")
        print(f"Total Actual Sells:    {len(actual_sells)}")
        print(f"Correctly Predicted:   {true_positives} (True Positives)")