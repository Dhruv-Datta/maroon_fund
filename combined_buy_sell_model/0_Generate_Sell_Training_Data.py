import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory

from config import load_config
from feature_engineering import compute_sell_features, fetch_stock_data, prepare_sell_labeled_frame


class PeakSelector:
    def __init__(self, data):
        self.data = data.copy()
        self.data["Date"] = pd.to_datetime(self.data["Date"]).dt.tz_localize(None)
        self.peaks = []
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.scatter = None
        self.setup_plot()

    def setup_plot(self):
        self.ax.plot(self.data["Date"], self.data["Close"], label="Close Price")
        self.ax.set_title("Select Peaks on Stock Chart (Use scroll to zoom, right-click to pan)")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.legend()
        plt.xticks(rotation=45)

        self.done_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.done_button = Button(self.done_button_ax, "Done")
        self.done_button.on_clicked(self.on_done)

        self.undo_button_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.undo_button = Button(self.undo_button_ax, "Undo")
        self.undo_button.on_clicked(self.on_undo)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        zoom_factory(self.ax)
        panhandler(self.fig)

    def on_click(self, event):
        if event.inaxes == self.ax and event.button == 1:
            x = event.xdata
            if x is not None:
                date = pd.Timestamp(mdates.num2date(x)).tz_localize(None)
                closest = min(self.data["Date"], key=lambda d: abs(d - date))
                price = self.data.loc[self.data["Date"] == closest, "Close"].values[0]
                self.peaks.append((closest, price))
                if self.scatter:
                    self.scatter.remove()
                self.scatter = self.ax.scatter(*zip(*self.peaks), color="green", zorder=5)
                plt.draw()

    def on_undo(self, _event):
        if self.peaks:
            self.peaks.pop()
            if self.scatter:
                self.scatter.remove()
            if self.peaks:
                self.scatter = self.ax.scatter(*zip(*self.peaks), color="green", zorder=5)
            else:
                self.scatter = None
            plt.draw()

    def on_done(self, _event):
        plt.close()

    def get_peaks(self):
        return pd.DataFrame(self.peaks, columns=["Date", "Price"])


def main():
    cfg = load_config()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    stock_data = fetch_stock_data(cfg.ticker, cfg.train_start_date, cfg.train_end_date)

    cleaned_data = prepare_sell_labeled_frame(compute_sell_features(stock_data))

    selector = PeakSelector(cleaned_data)
    plt.show()

    user_peaks = selector.get_peaks()
    for _, row in user_peaks.iterrows():
        peak_date = pd.Timestamp(row["Date"]).tz_localize(None)
        closest_date = min(cleaned_data["Date"], key=lambda d: abs(d - peak_date))
        cleaned_data.loc[cleaned_data["Date"] == closest_date, "Target"] = 1

    canonical_path = os.path.join(data_dir, "sell_signals.csv")
    per_ticker_path = os.path.join(data_dir, f"{cfg.ticker}_sell_signals.csv")
    cleaned_data.to_csv(canonical_path, index=False)
    cleaned_data.to_csv(per_ticker_path, index=False)
    print(f"Saved sell training data to: {canonical_path}")
    print(f"Saved sell training data to: {per_ticker_path}")

    plt.figure(figsize=(15, 8))
    plt.plot(cleaned_data["Date"], cleaned_data["Close"], label="Close Price")
    plt.scatter(
        cleaned_data[cleaned_data["Target"] == 1]["Date"],
        cleaned_data[cleaned_data["Target"] == 1]["Close"],
        color="green",
        label="User-selected Peaks",
        zorder=5,
    )
    plt.title("Stock with Confirmed User-selected Peaks (Sell Points)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
