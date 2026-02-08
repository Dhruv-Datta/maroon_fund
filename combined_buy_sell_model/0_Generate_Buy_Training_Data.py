import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory

from config import load_config
from feature_engineering import fetch_stock_data, compute_buy_features, prepare_buy_labeled_frame


class DipSelector:
    def __init__(self, data):
        self.data = data.copy()
        self.data["Date"] = pd.to_datetime(self.data["Date"]).dt.tz_localize(None)
        self.dips = []
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.scatter = None
        self.setup_plot()

    def setup_plot(self):
        self.ax.plot(self.data["Date"], self.data["Close"], label="Close Price")
        self.ax.set_title("Select Dips on Stock Chart (Use scroll to zoom, right-click to pan)")
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
                price = self.data.loc[self.data["Date"] == closest, "Close"].iloc[0]
                self.dips.append((closest, price))
                if self.scatter:
                    self.scatter.remove()
                self.scatter = self.ax.scatter(*zip(*self.dips), color="red", zorder=5)
                plt.draw()

    def on_undo(self, _event):
        if self.dips:
            self.dips.pop()
            if self.scatter:
                self.scatter.remove()
            if self.dips:
                self.scatter = self.ax.scatter(*zip(*self.dips), color="red", zorder=5)
            else:
                self.scatter = None
            plt.draw()

    def on_done(self, _event):
        plt.close()

    def get_dips(self):
        return pd.DataFrame(self.dips, columns=["Date", "Price"])


def main():
    cfg = load_config()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = fetch_stock_data(cfg.ticker, cfg.train_start_date, cfg.train_end_date)
    df = compute_buy_features(df, cfg.train_start_date, cfg.train_end_date)
    cleaned = prepare_buy_labeled_frame(df)

    selector = DipSelector(cleaned)
    plt.show()

    user_dips = selector.get_dips()
    cleaned["Target"] = 0
    for _, row in user_dips.iterrows():
        dd = pd.Timestamp(row["Date"]).tz_localize(None)
        closest = min(cleaned["Date"], key=lambda d: abs(d - dd))
        cleaned.loc[cleaned["Date"] == closest, "Target"] = 1

    out_path = os.path.join(data_dir, "training_input.csv")
    cleaned.to_csv(out_path, index=False)
    print(f"Saved buy training data to: {out_path}")

    plt.figure(figsize=(15, 8))
    plt.plot(cleaned["Date"], cleaned["Close"], label="Close Price")
    plt.scatter(
        cleaned.loc[cleaned["Target"] == 1, "Date"],
        cleaned.loc[cleaned["Target"] == 1, "Close"],
        color="red",
        label="User-selected Dips",
        zorder=5,
    )
    plt.title("Stock with Confirmed User-selected Dips")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
