"""
BacktestEngine — core walk-forward simulation logic.

Trains fresh models on the training period, then simulates trading on the
test period using the buy/sell signal pipeline.  Single position at a time
(fully invested or fully cash).
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from collections import Counter
from ta import momentum
from scipy.stats import entropy

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# ---------------------------------------------------------------------------
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.path.dirname(_parent_dir)


# ---------------------------------------------------------------------------
# Sell-side trainer (self-contained so we don't depend on module-level imports)
# ---------------------------------------------------------------------------
class StockSellSignalTrainer:
    def __init__(self, data_path, model_save_path="stock_sell_model.joblib"):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.model = None
        self.threshold = 0.75

    def load_and_prepare_data(self):
        data = pd.read_csv(self.data_path)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.fillna(0, inplace=True)
        self.features = [c for c in data.columns if c not in ["Date", "Target"]]
        self.X = data[self.features]
        self.y = data["Target"]
        data["Date"] = pd.to_datetime(data["Date"])
        self.data = data
        return self

    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        return self

    def balance_training_data(self):
        counts = self.y_train.value_counts()
        k = min(5, counts.min() - 1)
        if counts.min() > 5:
            sampler = SMOTE(random_state=42, k_neighbors=k)
        else:
            sampler = RandomOverSampler(random_state=42)
        self.X_train_balanced, self.y_train_balanced = sampler.fit_resample(
            self.X_train, self.y_train
        )
        return self

    def train_model(self):
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
        }
        base_model = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc", random_state=42
        )
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0
        )
        grid_search.fit(self.X_train_balanced, self.y_train_balanced)
        self.model = grid_search.best_estimator_
        return self

    def evaluate_model(self):
        return self

    def generate_predictions(self):
        return self

    def save_model(self):
        return self


# ---------------------------------------------------------------------------
# Feature engineering (mirrored from 2_Loaded_Models.py to avoid import
# side-effects — that module reads CSV and loads models at import time)
# ---------------------------------------------------------------------------

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    print(f"\nFetching data for {ticker} from {start_date} to {end_date}")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date).reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def compute_buy_side_features(df):
    """Compute all features needed for buy side model."""
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = (df["High"] - df["Low"]) / df["Open"]
    df["MA_25"] = df["Close"].rolling(window=25, min_periods=1).mean()
    df["MA_100"] = df["Close"].rolling(window=100, min_periods=1).mean()
    df["RSI_14"] = momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    def compute_growth_rate(i, closes):
        if i < 12:
            return (closes[i] - closes[0]) / closes[0]
        return (closes[i] - closes[i - 12]) / closes[i - 12]

    df["Month_Growth_Rate"] = df["Close"].rolling(window=len(df), min_periods=1).apply(
        lambda x: compute_growth_rate(len(x) - 1, x.values), raw=False
    )

    low50 = df["Low"].rolling(window=50, min_periods=1).min()
    high50 = df["High"].rolling(window=50, min_periods=1).max()
    df["STOCH_%K"] = (df["Close"] - low50) / (high50 - low50) * 100
    df["STOCH_%D"] = df["STOCH_%K"].rolling(window=20, min_periods=1).mean()

    df["Roll_Vol_20d"] = df["Daily_Return"].rolling(window=20, min_periods=1).std()
    vol_mean = df["Roll_Vol_20d"].rolling(window=126, min_periods=1).mean()
    vol_std = df["Roll_Vol_20d"].rolling(window=126, min_periods=1).std()
    df["Vol_zscore"] = (df["Roll_Vol_20d"] - vol_mean) / vol_std

    ui = []
    for i in range(len(df)):
        window = df["Close"].iloc[max(0, i - 13): i + 1]
        peak = window.max()
        dd = (peak - window) / peak
        ui.append(np.sqrt((dd ** 2).mean()))
    df["Ulcer_Index"] = ui

    def calc_entropy(series):
        arr = series[~np.isnan(series)]
        if arr.size == 0:
            return 0.0
        counts, _ = np.histogram(arr, bins=10)
        return entropy(counts + 1)

    df["Return_Entropy_50d"] = (
        df["Daily_Return"]
        .rolling(window=50, min_periods=1)
        .apply(calc_entropy, raw=False)
    )

    try:
        vix = yf.Ticker("^VIX").history(start=df["Date"].min(), end=df["Date"].max()).reset_index()
        vix["Date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None)
        vix = vix[["Date", "Close"]].rename(columns={"Close": "VIX"})
        df = df.merge(vix, on="Date", how="left")
    except Exception:
        df["VIX"] = 20

    df["Growth_Since_Last_Bottom"] = 0.0
    df["Days_Since_Last_Bottom"] = 0

    def compute_week_growth_rate(i, closes):
        if i < 5:
            return (closes[i] - closes[0]) / closes[0]
        return (closes[i] - closes[i - 5]) / closes[i - 5]

    df["Week_Growth_Rate"] = df["Close"].rolling(window=len(df), min_periods=1).apply(
        lambda x: compute_week_growth_rate(len(x) - 1, x.values), raw=False
    )

    last_bot = 0
    for i in range(1, len(df)):
        if df.loc[i, "Close"] < df.loc[last_bot, "Close"]:
            last_bot = i
        df.loc[i, "Growth_Since_Last_Bottom"] = (
            df.loc[i, "Close"] - df.loc[last_bot, "Close"]
        ) / df.loc[last_bot, "Close"]
        df.loc[i, "Days_Since_Last_Bottom"] = i - last_bot

    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    return df


def compute_sell_side_features(df):
    """Compute all features needed for sell side model."""
    df["MA_25"] = df["Close"].rolling(25, min_periods=1).mean()
    df["MA_100"] = df["Close"].rolling(100, min_periods=1).mean()
    df["Volatility"] = (df["High"] - df["Low"]) / df["Open"]

    last_peak_idx = 0
    df["Days_Since_Last_Peak"] = 0
    for i in range(1, len(df)):
        if df.loc[i, "Close"] > df.loc[last_peak_idx, "Close"]:
            last_peak_idx = i
        df.loc[i, "Days_Since_Last_Peak"] = (df.loc[i, "Date"] - df.loc[last_peak_idx, "Date"]).days

    last_peak_price = df.loc[last_peak_idx, "Close"]
    df["Decline_Since_Last_Peak"] = (df["Close"] - last_peak_price) / last_peak_price

    df["One_Week_Growth"] = df["Close"].pct_change(periods=5)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + rs)
    df["RSI"] = df["RSI"].fillna(50)

    df["Price_to_MA25"] = df["Close"] / df["MA_25"]
    df["Price_to_MA100"] = df["Close"] / df["MA_100"]

    df.fillna(0, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: datetime
    entry_price: float
    exit_date: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0


@dataclass
class BacktestResult:
    ticker: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature names (must match model training order)
# ---------------------------------------------------------------------------
BUY_FEATURES = [
    "Low", "Close", "Volatility", "MA_25", "MA_100", "RSI_14",
    "Month_Growth_Rate", "STOCH_%K", "STOCH_%D", "Vol_zscore",
    "Ulcer_Index", "Return_Entropy_50d", "VIX",
    "Growth_Since_Last_Bottom", "Days_Since_Last_Bottom", "Week_Growth_Rate",
]

SELL_FEATURES = [
    "Close", "Volatility", "MA_25", "MA_100",
    "Decline_Since_Last_Peak", "Days_Since_Last_Peak",
    "One_Week_Growth", "RSI", "Price_to_MA25", "Price_to_MA100",
]


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------
class BacktestEngine:
    def __init__(
        self,
        ticker: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        initial_capital: float = 100_000,
        buy_threshold: float = 0.9,
        sell_threshold: float = 0.75,
    ):
        self.ticker = ticker
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.initial_capital = initial_capital
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        self.buy_model = None
        self.sell_model = None

    # ----- training --------------------------------------------------------

    def _train_buy_model(self, buy_training_path: str):
        """Train buy-side model from CSV."""
        buy_data = pd.read_csv(buy_training_path)
        buy_features = [c for c in buy_data.columns if c not in ["Date", "Target"]]
        X = buy_data[buy_features]
        y = buy_data["Target"]

        neg, pos = Counter(y).values()
        scale = neg / pos

        pipeline = ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=scale,
                max_depth=3,
                n_estimators=100,
                min_child_weight=5,
                subsample=0.7,
                colsample_bytree=0.7,
            )),
        ])

        param_grid = {"xgb__n_estimators": [100, 200], "xgb__max_depth": [3, 5]}
        tscv = TimeSeriesSplit(n_splits=5)
        gs = GridSearchCV(pipeline, param_grid, cv=tscv, scoring="f1", n_jobs=-1)
        gs.fit(X, y)
        self.buy_model = gs.best_estimator_
        print(f"  Buy model trained — best params: {gs.best_params_}")

    def _train_sell_model(self, sell_training_path: str):
        """Train sell-side model from CSV."""
        trainer = StockSellSignalTrainer(data_path=sell_training_path)
        (
            trainer
            .load_and_prepare_data()
            .split_data()
            .balance_training_data()
            .train_model()
        )
        self.sell_model = trainer.model
        print("  Sell model trained")

    def train(self):
        """Train both models using existing training CSVs."""
        print(f"\nTraining models for {self.ticker} ...")

        buy_path = os.path.join(_repo_root, "1_buyside_model", "training_input.csv")
        sell_path = os.path.join(
            _repo_root, "2_sellside_model", "sell_signals", "sell_signals.csv"
        )

        if not os.path.exists(buy_path):
            raise FileNotFoundError(f"Buy training data not found: {buy_path}")
        if not os.path.exists(sell_path):
            raise FileNotFoundError(f"Sell training data not found: {sell_path}")

        self._train_buy_model(buy_path)
        self._train_sell_model(sell_path)

    # ----- simulation ------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute walk-forward backtest on the test period."""
        if self.buy_model is None or self.sell_model is None:
            self.train()

        # Fetch data covering both train and test period for feature warm-up
        # We need extra history before test_start so rolling features are valid
        warmup_start = pd.Timestamp(self.test_start) - pd.Timedelta(days=200)
        warmup_start_str = warmup_start.strftime("%Y-%m-%d")

        print(f"\nFetching {self.ticker} data ({warmup_start_str} to {self.test_end}) ...")
        df = fetch_stock_data(self.ticker, warmup_start_str, self.test_end)

        if df.empty:
            raise ValueError(f"No data returned for {self.ticker}")

        # Compute features on the full dataframe
        df = df.reset_index(drop=True)
        df = compute_buy_side_features(df)
        df = compute_sell_side_features(df)

        for feat in BUY_FEATURES + SELL_FEATURES:
            if feat not in df.columns:
                df[feat] = 0

        # Restrict to test period
        test_mask = df["Date"] >= pd.Timestamp(self.test_start)
        test_df = df[test_mask].copy().reset_index(drop=True)

        if test_df.empty:
            raise ValueError(f"No test data for {self.ticker} in [{self.test_start}, {self.test_end}]")

        # Generate probabilities for entire test set
        X_buy = test_df[BUY_FEATURES].fillna(0)
        X_sell = test_df[SELL_FEATURES].fillna(0)
        test_df["buy_prob"] = self.buy_model.predict_proba(X_buy)[:, 1]
        test_df["sell_prob"] = self.sell_model.predict_proba(X_sell)[:, 1]

        # Walk-forward simulation
        cash = self.initial_capital
        shares = 0.0
        position_open = False
        trades: list[Trade] = []
        current_trade: Trade | None = None

        equity_rows = []

        for i, row in test_df.iterrows():
            price = row["Close"]
            date = row["Date"]
            buy_prob = row["buy_prob"]
            sell_prob = row["sell_prob"]

            # Entry
            if not position_open and buy_prob >= self.buy_threshold:
                shares = cash / price
                cash = 0.0
                position_open = True
                current_trade = Trade(entry_date=date, entry_price=price)

            # Exit
            elif position_open and sell_prob >= self.sell_threshold:
                cash = shares * price
                pnl = cash - (current_trade.entry_price * shares)
                pnl_pct = (price - current_trade.entry_price) / current_trade.entry_price
                holding = (date - current_trade.entry_date).days
                current_trade.exit_date = date
                current_trade.exit_price = price
                current_trade.pnl = pnl
                current_trade.pnl_pct = pnl_pct
                current_trade.holding_days = holding
                trades.append(current_trade)
                current_trade = None
                shares = 0.0
                position_open = False

            portfolio_value = cash + shares * price
            equity_rows.append({
                "Date": date,
                "Close": price,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "shares": shares,
                "in_position": position_open,
                "buy_prob": buy_prob,
                "sell_prob": sell_prob,
            })

        # Close any open position at end of test
        if position_open and current_trade is not None:
            last = test_df.iloc[-1]
            price = last["Close"]
            date = last["Date"]
            cash = shares * price
            pnl = cash - (current_trade.entry_price * shares)
            pnl_pct = (price - current_trade.entry_price) / current_trade.entry_price
            holding = (date - current_trade.entry_date).days
            current_trade.exit_date = date
            current_trade.exit_price = price
            current_trade.pnl = pnl
            current_trade.pnl_pct = pnl_pct
            current_trade.holding_days = holding
            trades.append(current_trade)

        equity_df = pd.DataFrame(equity_rows)

        # Buy-and-hold benchmark
        if not equity_df.empty:
            first_price = equity_df.iloc[0]["Close"]
            equity_df["buy_hold_value"] = (
                equity_df["Close"] / first_price * self.initial_capital
            )

        result = BacktestResult(
            ticker=self.ticker,
            trades=trades,
            equity_curve=equity_df,
            daily_positions=equity_df[["Date", "in_position"]],
            config={
                "train_start": self.train_start,
                "train_end": self.train_end,
                "test_start": self.test_start,
                "test_end": self.test_end,
                "initial_capital": self.initial_capital,
                "buy_threshold": self.buy_threshold,
                "sell_threshold": self.sell_threshold,
            },
        )
        return result
