"""
Individual model visualizations from local combined config.
"""

import importlib.util
import os

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


def visualize_buy_model(ticker=None, start_date=None, end_date=None):
    ticker = ticker or cfg.ticker
    start_date = start_date or cfg.train_start_date
    end_date = end_date or cfg.train_end_date

    df = fetch_stock_data(ticker, start_date, end_date)
    df = compute_buy_features(df, start_date, end_date)

    for feature in BUY_FEATURE_COLUMNS:
        if feature not in df.columns:
            df[feature] = 0

    X_buy = df[BUY_FEATURE_COLUMNS].fillna(0)
    df["Dip_Probability"] = buy_model.predict_proba(X_buy)[:, 1]
    df["Predicted_Dip"] = df["Dip_Probability"] > cfg.buy_threshold

    predicted_dips = df[df["Predicted_Dip"]]

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

    if len(predicted_dips) > 0:
        fig.add_trace(
            go.Scatter(
                x=predicted_dips["Date"],
                y=predicted_dips["Close"],
                mode="markers+text",
                name="Predicted Dip",
                marker=dict(color="red", size=8),
                text=[f"Prob {p:.2f}" for p in predicted_dips["Dip_Probability"]],
                textposition="top center",
                hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>Dip Probability: %{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Buy Side Predictions - {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="closest",
        template="plotly_white",
        height=600,
    )
    fig.show()
    return df


def visualize_sell_model(ticker=None, start_date=None, end_date=None):
    ticker = ticker or cfg.ticker
    start_date = start_date or cfg.train_start_date
    end_date = end_date or cfg.train_end_date

    df = fetch_stock_data(ticker, start_date, end_date)
    df = compute_sell_features(df)

    for feature in SELL_FEATURE_COLUMNS:
        if feature not in df.columns:
            df[feature] = 0

    X_sell = df[SELL_FEATURE_COLUMNS].fillna(0)
    df["Sell_Probability"] = sell_model.predict_proba(X_sell)[:, 1]
    df["Sell_Signal"] = df["Sell_Probability"] > cfg.sell_threshold

    sell_signals = df[df["Sell_Signal"]]

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
        title=f"Sell Side Predictions - {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="closest",
        template="plotly_white",
        height=600,
    )
    fig.show()
    return df


if __name__ == "__main__":
    visualize_buy_model()
    visualize_sell_model()
