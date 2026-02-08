"""
Future predictions from local combined config.
"""

import importlib.util
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go

from config import load_config
from feature_engineering import BUY_FEATURE_COLUMNS, SELL_FEATURE_COLUMNS

model_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "loaded_models", os.path.join(model_dir, "2_Loaded_Models.py")
)
loaded_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loaded_models)
fetch_stock_data = loaded_models.fetch_stock_data
compute_buy_features = loaded_models.compute_buy_features
compute_sell_features = loaded_models.compute_sell_features
buy_model = loaded_models.buy_model
sell_model = loaded_models.sell_model


cfg = load_config()


def run_future_predictions(ticker=None, start_date=None, end_date=None):
    ticker = ticker or cfg.ticker
    start_date = start_date or cfg.test_start_date
    end_date = end_date or cfg.test_end_date

    print("=" * 60)
    print("FUTURE PREDICTIONS")
    print("=" * 60)

    # Pull extra lookback for rolling indicators, then filter to test window.
    warmup_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=100)).strftime("%Y-%m-%d")

    df = fetch_stock_data(ticker, warmup_start, end_date)
    df = compute_buy_features(df, warmup_start, end_date)
    df = compute_sell_features(df)

    for feature in BUY_FEATURE_COLUMNS:
        if feature not in df.columns:
            df[feature] = 0
    for feature in SELL_FEATURE_COLUMNS:
        if feature not in df.columns:
            df[feature] = 0

    X_buy = df[BUY_FEATURE_COLUMNS].fillna(0)
    X_sell = df[SELL_FEATURE_COLUMNS].fillna(0)

    df["Dip_Probability"] = buy_model.predict_proba(X_buy)[:, 1]
    df["Predicted_Dip"] = df["Dip_Probability"] > cfg.buy_threshold

    df["Sell_Probability"] = sell_model.predict_proba(X_sell)[:, 1]
    df["Sell_Signal"] = df["Sell_Probability"] > cfg.sell_threshold

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    period_df = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()
    buy_signals = period_df[period_df["Predicted_Dip"]]
    sell_signals = period_df[period_df["Sell_Signal"]]

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

    highlight_start = pd.Timestamp(cfg.train_end_date)
    fig.add_vrect(
        x0=highlight_start,
        x1=end_ts,
        fillcolor="yellow",
        opacity=0.1,
        layer="below",
        line_width=0,
        annotation_text="Prediction Period",
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
                textposition="top center",
                hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>Sell Probability: %{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Future Predictions - {ticker.upper()} ({start_date} to {end_date})",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="closest",
        template="plotly_white",
        height=600,
    )
    fig.show()

    output_file = os.path.join(model_dir, f"future_predictions_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv")
    period_df.to_csv(output_file, index=False)
    print(f"Saved future predictions to: {output_file}")
    return period_df


if __name__ == "__main__":
    run_future_predictions()
