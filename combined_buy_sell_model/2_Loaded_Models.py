"""
Load trained models and run combined analysis from local combined pipeline config.
"""

import os
from datetime import datetime

import joblib
import pandas as pd
import plotly.graph_objects as go

from config import load_config
from feature_engineering import (
    BUY_FEATURE_COLUMNS,
    SELL_FEATURE_COLUMNS,
    compute_buy_features,
    compute_sell_features,
    fetch_stock_data,
)

cfg = load_config()
model_dir = os.path.dirname(os.path.abspath(__file__))
buy_model_path = os.path.join(model_dir, f"{cfg.ticker}_buy_model.joblib")
sell_model_path = os.path.join(model_dir, f"{cfg.ticker}_sell_model.joblib")

buy_model = None
sell_model = None


def load_models():
    global buy_model, sell_model
    if buy_model is not None and sell_model is not None:
        return

    if not os.path.exists(buy_model_path):
        raise FileNotFoundError(f"Buy model not found: {buy_model_path}. Run 1_Train_Models.py first.")
    if not os.path.exists(sell_model_path):
        raise FileNotFoundError(f"Sell model not found: {sell_model_path}. Run 1_Train_Models.py first.")

    buy_model = joblib.load(buy_model_path)
    sell_model = joblib.load(sell_model_path)


load_models()


def _load_analysis_frame_from_test_csv(ticker: str, start_date: str, end_date: str):
    data_dir = os.path.join(model_dir, "data")
    test_path = os.path.join(data_dir, "test_input.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test input CSV not found: {test_path}. Run make generate-buy-test first.")

    df = pd.read_csv(test_path)
    if "Date" not in df.columns:
        raise ValueError("test_input.csv must include a Date column.")
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))].copy()
    if df.empty:
        raise ValueError(
            f"test_input.csv has no rows in range {start_date} to {end_date}. "
            "Update config test dates or regenerate test_input.csv."
        )

    # Buy features are expected in test_input.csv; fill if missing.
    for feature in BUY_FEATURE_COLUMNS:
        if feature not in df.columns:
            df[feature] = 0

    # Sell features may not exist in buy test CSV; derive by date from market data.
    missing_sell = [feature for feature in SELL_FEATURE_COLUMNS if feature not in df.columns]
    if missing_sell:
        raw = fetch_stock_data(ticker, start_date, end_date)
        raw = compute_sell_features(raw)
        sell_df = raw[["Date"] + SELL_FEATURE_COLUMNS].copy()
        sell_df["Date"] = pd.to_datetime(sell_df["Date"]).dt.tz_localize(None)
        df = df.merge(sell_df, on="Date", how="left", suffixes=("", "_sell"))
        for feature in SELL_FEATURE_COLUMNS:
            alt = f"{feature}_sell"
            if feature not in df.columns and alt in df.columns:
                df[feature] = df[alt]
            if alt in df.columns:
                df.drop(columns=[alt], inplace=True)

    for feature in SELL_FEATURE_COLUMNS:
        if feature not in df.columns:
            df[feature] = 0

    return df


def run_combined_analysis(ticker=None, start_date=None, end_date=None):
    ticker = ticker or cfg.ticker
    start_date = start_date or cfg.test_start_date
    end_date = end_date or cfg.test_end_date

    print("=" * 60)
    print("COMBINED BUY & SELL ANALYSIS")
    print("=" * 60)

    df = _load_analysis_frame_from_test_csv(ticker, start_date, end_date)

    X_buy = df[BUY_FEATURE_COLUMNS].fillna(0)
    X_sell = df[SELL_FEATURE_COLUMNS].fillna(0)

    df["Dip_Probability"] = buy_model.predict_proba(X_buy)[:, 1]
    df["Predicted_Dip"] = df["Dip_Probability"] > cfg.buy_threshold

    df["Sell_Probability"] = sell_model.predict_proba(X_sell)[:, 1]
    df["Sell_Signal"] = df["Sell_Probability"] > cfg.sell_threshold

    buy_signals = df[df["Predicted_Dip"]]
    sell_signals = df[df["Sell_Signal"]]

    print("\nCombined analysis results:")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Trading days: {len(df)}")
    print(f"  Buy signals: {len(buy_signals)}")
    print(f"  Sell signals: {len(sell_signals)}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="blue"),
            hoverinfo="skip",
        )
    )

    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_signals["Date"],
                y=buy_signals["Close"],
                mode="markers+text",
                name="Predicted Dip",
                marker=dict(color="green", size=8),
                text=[f"Prob {p:.2f}" for p in buy_signals["Dip_Probability"]],
                textposition="top center",
                hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>Dip Probability: %{text}<extra></extra>",
            )
        )

    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_signals["Date"],
                y=sell_signals["Close"],
                mode="markers+text",
                name="Predicted Sell",
                marker=dict(color="red", size=8),
                text=[f"Prob {p:.2f}" for p in sell_signals["Sell_Probability"]],
                textposition="bottom center",
                hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>Sell Probability: %{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Combined Buy & Sell Analysis - {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="closest",
        template="plotly_white",
        autosize=True,
        width=None,
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.show()

    output_file = os.path.join(
        model_dir,
        f"combined_analysis_{ticker}_{df['Date'].min().strftime('%Y%m%d')}_{df['Date'].max().strftime('%Y%m%d')}.csv",
    )
    df.to_csv(output_file, index=False)
    print(f"Saved analysis CSV: {output_file}")
    return df


if __name__ == "__main__":
    run_combined_analysis(cfg.ticker, cfg.test_start_date, cfg.test_end_date)
    print("\nCombined analysis complete")
