"""
Backtest chart generation — all output is standalone Plotly HTML.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_all_charts(result, output_dir: str):
    """Generate and save all backtest charts for a single ticker."""
    _ensure_dir(output_dir)
    eq = result.equity_curve
    trades = result.trades
    ticker = result.ticker

    if eq.empty:
        print(f"  No data to chart for {ticker}")
        return

    _equity_curve(eq, trades, ticker, output_dir)
    _drawdown_chart(eq, ticker, output_dir)
    _trade_scatter(eq, trades, ticker, output_dir)
    _monthly_returns_heatmap(eq, ticker, output_dir)
    _rolling_sharpe(eq, ticker, output_dir)
    _signal_distribution(eq, ticker, output_dir)

    print(f"  Charts saved to {output_dir}")


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------

def _equity_curve(eq, trades, ticker, out):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["Date"], y=eq["portfolio_value"],
        mode="lines", name="Strategy",
        line=dict(color="blue"),
    ))
    fig.add_trace(go.Scatter(
        x=eq["Date"], y=eq["buy_hold_value"],
        mode="lines", name="Buy & Hold",
        line=dict(color="gray", dash="dash"),
    ))

    # Mark trades
    for t in trades:
        fig.add_trace(go.Scatter(
            x=[t.entry_date], y=[eq.loc[eq["Date"] == t.entry_date, "portfolio_value"].values[0]]
            if len(eq.loc[eq["Date"] == t.entry_date]) else [None],
            mode="markers", marker=dict(symbol="triangle-up", color="green", size=10),
            showlegend=False,
        ))
        if t.exit_date is not None:
            exit_vals = eq.loc[eq["Date"] == t.exit_date, "portfolio_value"]
            fig.add_trace(go.Scatter(
                x=[t.exit_date],
                y=[exit_vals.values[0]] if len(exit_vals) else [None],
                mode="markers", marker=dict(symbol="triangle-down", color="red", size=10),
                showlegend=False,
            ))

    fig.update_layout(
        title=f"Equity Curve — {ticker}",
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        template="plotly_white", height=500,
    )
    fig.write_html(os.path.join(out, "equity_curve.html"))


def _drawdown_chart(eq, ticker, out):
    cummax = eq["portfolio_value"].cummax()
    dd = (eq["portfolio_value"] - cummax) / cummax * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["Date"], y=dd,
        fill="tozeroy", mode="lines",
        line=dict(color="crimson"),
        name="Drawdown %",
    ))
    fig.update_layout(
        title=f"Underwater / Drawdown — {ticker}",
        xaxis_title="Date", yaxis_title="Drawdown %",
        template="plotly_white", height=400,
    )
    fig.write_html(os.path.join(out, "drawdown.html"))


def _trade_scatter(eq, trades, ticker, out):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["Date"], y=eq["Close"],
        mode="lines", name="Close Price",
        line=dict(color="steelblue"),
    ))

    buy_dates = [t.entry_date for t in trades]
    buy_prices = [t.entry_price for t in trades]
    sell_dates = [t.exit_date for t in trades if t.exit_date]
    sell_prices = [t.exit_price for t in trades if t.exit_price]

    fig.add_trace(go.Scatter(
        x=buy_dates, y=buy_prices,
        mode="markers", name="Buy",
        marker=dict(symbol="triangle-up", color="green", size=12),
    ))
    fig.add_trace(go.Scatter(
        x=sell_dates, y=sell_prices,
        mode="markers", name="Sell",
        marker=dict(symbol="triangle-down", color="red", size=12),
    ))
    fig.update_layout(
        title=f"Trade Entry/Exit — {ticker}",
        xaxis_title="Date", yaxis_title="Price ($)",
        template="plotly_white", height=500,
    )
    fig.write_html(os.path.join(out, "trade_scatter.html"))


def _monthly_returns_heatmap(eq, ticker, out):
    pv = eq.set_index("Date")["portfolio_value"]
    monthly = pv.resample("ME").last().pct_change().dropna()

    if monthly.empty:
        return

    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values * 100,
    })
    pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index],
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
    ))
    fig.update_layout(
        title=f"Monthly Returns (%) — {ticker}",
        template="plotly_white", height=300 + 40 * len(pivot),
    )
    fig.write_html(os.path.join(out, "monthly_returns.html"))


def _rolling_sharpe(eq, ticker, out, window=60, rf=0.04):
    pv = eq["portfolio_value"].values
    rets = np.diff(pv) / pv[:-1]
    if len(rets) < window:
        return

    rolling = pd.Series(rets)
    excess = rolling - rf / 252
    roll_mean = excess.rolling(window).mean()
    roll_std = rolling.rolling(window).std()
    rs = (roll_mean / roll_std * np.sqrt(252)).dropna()

    dates = eq["Date"].iloc[1:].reset_index(drop=True)
    dates = dates.iloc[window - 1 : window - 1 + len(rs)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates.values, y=rs.values,
        mode="lines", name=f"{window}-day Rolling Sharpe",
        line=dict(color="purple"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Rolling Sharpe Ratio ({window}d) — {ticker}",
        xaxis_title="Date", yaxis_title="Sharpe Ratio",
        template="plotly_white", height=400,
    )
    fig.write_html(os.path.join(out, "rolling_sharpe.html"))


def _signal_distribution(eq, ticker, out):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Buy Probabilities", "Sell Probabilities"])

    fig.add_trace(go.Histogram(
        x=eq["buy_prob"], nbinsx=50, name="Buy Prob",
        marker_color="green", opacity=0.7,
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=eq["sell_prob"], nbinsx=50, name="Sell Prob",
        marker_color="red", opacity=0.7,
    ), row=1, col=2)

    fig.update_layout(
        title=f"Signal Probability Distribution — {ticker}",
        template="plotly_white", height=400, showlegend=False,
    )
    fig.write_html(os.path.join(out, "signal_distribution.html"))
