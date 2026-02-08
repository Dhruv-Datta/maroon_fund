import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import entropy
from ta import momentum


BUY_FEATURE_COLUMNS = [
    "Low",
    "Close",
    "Volatility",
    "MA_25",
    "MA_100",
    "RSI_14",
    "Month_Growth_Rate",
    "STOCH_%K",
    "STOCH_%D",
    "Vol_zscore",
    "Ulcer_Index",
    "Return_Entropy_50d",
    "VIX",
    "Growth_Since_Last_Bottom",
    "Days_Since_Last_Bottom",
    "Week_Growth_Rate",
]

SELL_FEATURE_COLUMNS = [
    "Close",
    "Volatility",
    "MA_25",
    "MA_100",
    "Decline_Since_Last_Peak",
    "Days_Since_Last_Peak",
    "One_Week_Growth",
    "RSI",
    "Price_to_MA25",
    "Price_to_MA100",
]


def _compute_growth_rate(i, closes):
    if i < 12:
        return (closes[i] - closes[0]) / closes[0]
    return (closes[i] - closes[i - 12]) / closes[i - 12]


def _compute_week_growth_rate(i, closes):
    if i < 5:
        return (closes[i] - closes[0]) / closes[0]
    return (closes[i] - closes[i - 5]) / closes[i - 5]


def _calc_entropy(series):
    arr = series[~np.isnan(series)]
    if arr.size == 0:
        return 0.0
    counts, _ = np.histogram(arr, bins=10)
    return entropy(counts + 1)


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date).reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def compute_buy_features(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    out = df.copy()

    out["Daily_Return"] = out["Close"].pct_change()
    out["Volatility"] = (out["High"] - out["Low"]) / out["Open"]
    out["MA_25"] = out["Close"].rolling(window=25, min_periods=1).mean()
    out["MA_100"] = out["Close"].rolling(window=100, min_periods=1).mean()
    out["RSI_14"] = momentum.RSIIndicator(close=out["Close"], window=14).rsi()

    out["Month_Growth_Rate"] = out["Close"].rolling(window=len(out), min_periods=1).apply(
        lambda x: _compute_growth_rate(len(x) - 1, x.values), raw=False
    )

    low50 = out["Low"].rolling(window=50, min_periods=1).min()
    high50 = out["High"].rolling(window=50, min_periods=1).max()
    out["STOCH_%K"] = (out["Close"] - low50) / (high50 - low50) * 100
    out["STOCH_%D"] = out["STOCH_%K"].rolling(window=20, min_periods=1).mean()

    out["Roll_Vol_20d"] = out["Daily_Return"].rolling(window=20, min_periods=1).std()
    vol_mean = out["Roll_Vol_20d"].rolling(window=126, min_periods=1).mean()
    vol_std = out["Roll_Vol_20d"].rolling(window=126, min_periods=1).std()
    out["Vol_zscore"] = (out["Roll_Vol_20d"] - vol_mean) / vol_std

    ui = []
    for i in range(len(out)):
        window = out["Close"].iloc[max(0, i - 13): i + 1]
        peak = window.max()
        dd = (peak - window) / peak
        ui.append(np.sqrt((dd ** 2).mean()))
    out["Ulcer_Index"] = ui

    out["Return_Entropy_50d"] = (
        out["Daily_Return"].rolling(window=50, min_periods=1).apply(_calc_entropy, raw=False)
    )

    try:
        vix = yf.Ticker("^VIX").history(start=start_date, end=end_date).reset_index()
        vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None)
        vix = vix[["Date", "Close"]].rename(columns={"Close": "VIX"})
        out = out.merge(vix, on="Date", how="left")
    except Exception:
        out["VIX"] = 20

    out["Growth_Since_Last_Bottom"] = 0.0
    out["Days_Since_Last_Bottom"] = 0
    out["Week_Growth_Rate"] = out["Close"].rolling(window=len(out), min_periods=1).apply(
        lambda x: _compute_week_growth_rate(len(x) - 1, x.values), raw=False
    )

    last_bot = 0
    for i in range(1, len(out)):
        if out.loc[i, "Close"] < out.loc[last_bot, "Close"]:
            last_bot = i
        out.loc[i, "Growth_Since_Last_Bottom"] = (
            out.loc[i, "Close"] - out.loc[last_bot, "Close"]
        ) / out.loc[last_bot, "Close"]
        out.loc[i, "Days_Since_Last_Bottom"] = i - last_bot

    out.bfill(inplace=True)
    out.fillna(0, inplace=True)
    return out


def prepare_buy_labeled_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "Open",
        "High",
        "Volume",
        "Daily_Return",
        "Roll_Vol_20d",
        "OBV",
        "Dividends",
        "Stock Splits",
        "Capital Gains",
    ]
    cleaned = df.drop(columns=drop_cols, errors="ignore")
    if "Target" not in cleaned.columns:
        cleaned["Target"] = 0
    return cleaned


def compute_sell_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["MA_25"] = out["Close"].rolling(window=25, min_periods=1).mean()
    out["MA_100"] = out["Close"].rolling(window=100, min_periods=1).mean()
    out["Volatility"] = (out["High"] - out["Low"]) / out["Open"]

    last_peak_index = 0
    out["Days_Since_Last_Peak"] = 0
    for i in range(1, len(out)):
        if out.loc[i, "Close"] > out.loc[last_peak_index, "Close"]:
            last_peak_index = i
        out.loc[i, "Days_Since_Last_Peak"] = (
            out.loc[i, "Date"] - out.loc[last_peak_index, "Date"]
        ).days

    last_peak_price = out.loc[last_peak_index, "Close"]
    out["Decline_Since_Last_Peak"] = (out["Close"] - last_peak_price) / last_peak_price

    out["One_Week_Growth"] = out["Close"].pct_change(periods=5)

    delta = out["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    out["RSI"] = 100 - (100 / (1 + rs))
    out["RSI"] = out["RSI"].fillna(50)

    out["Price_to_MA25"] = out["Close"] / out["MA_25"]
    out["Price_to_MA100"] = out["Close"] / out["MA_100"]

    out.fillna(0, inplace=True)
    return out


def prepare_sell_labeled_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Target"] = 0
    column_order = [
        "Date",
        "Close",
        "Volatility",
        "MA_25",
        "MA_100",
        "Decline_Since_Last_Peak",
        "Days_Since_Last_Peak",
        "One_Week_Growth",
        "RSI",
        "Price_to_MA25",
        "Price_to_MA100",
        "Target",
    ]
    return out[column_order]
