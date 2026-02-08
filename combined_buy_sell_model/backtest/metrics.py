"""
Performance metrics computed from a BacktestResult.
"""

import numpy as np
import pandas as pd
from tabulate import tabulate


def compute_metrics(result, risk_free_rate: float = 0.04) -> dict:
    """Return a dict of performance metrics from a BacktestResult."""
    eq = result.equity_curve
    trades = result.trades
    capital = result.config["initial_capital"]

    if eq.empty:
        return {"error": "No equity curve data"}

    # --- returns -----------------------------------------------------------
    total_days = len(eq)
    final_value = eq["portfolio_value"].iloc[-1]
    total_return = (final_value - capital) / capital

    first_date = eq["Date"].iloc[0]
    last_date = eq["Date"].iloc[-1]
    years = max((last_date - first_date).days / 365.25, 1e-6)
    ann_return = (1 + total_return) ** (1 / years) - 1

    # Buy & hold
    bh_final = eq["buy_hold_value"].iloc[-1]
    bh_return = (bh_final - capital) / capital

    # --- daily returns -----------------------------------------------------
    pv = eq["portfolio_value"].values
    daily_returns = np.diff(pv) / pv[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    # --- Sharpe ------------------------------------------------------------
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        excess = daily_returns - risk_free_rate / 252
        sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252)
    else:
        sharpe = 0.0

    # --- Sortino -----------------------------------------------------------
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        excess_mean = np.mean(daily_returns) - risk_free_rate / 252
        sortino = excess_mean / np.std(downside) * np.sqrt(252)
    else:
        sortino = 0.0

    # --- Drawdown ----------------------------------------------------------
    cummax = eq["portfolio_value"].cummax()
    drawdown = (eq["portfolio_value"] - cummax) / cummax
    max_dd = drawdown.min()

    # Max drawdown duration
    dd_dur = 0
    max_dd_dur = 0
    for dd in drawdown:
        if dd < 0:
            dd_dur += 1
            max_dd_dur = max(max_dd_dur, dd_dur)
        else:
            dd_dur = 0

    # --- Calmar ------------------------------------------------------------
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    # --- Trade stats -------------------------------------------------------
    n_trades = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades else 0.0
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
    gross_profit = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_holding = np.mean([t.holding_days for t in trades]) if trades else 0.0

    # Exposure
    in_market = eq["in_position"].sum()
    exposure = in_market / total_days * 100 if total_days else 0.0

    metrics = {
        "Total Return %": total_return * 100,
        "Annualized Return %": ann_return * 100,
        "Buy & Hold Return %": bh_return * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown %": max_dd * 100,
        "Max DD Duration (days)": max_dd_dur,
        "Calmar Ratio": calmar,
        "Win Rate %": win_rate,
        "Avg Win %": avg_win * 100,
        "Avg Loss %": avg_loss * 100,
        "Profit Factor": profit_factor,
        "Number of Trades": n_trades,
        "Avg Holding (days)": avg_holding,
        "Exposure %": exposure,
    }
    return metrics


def format_metrics_table(metrics: dict, ticker: str = "") -> str:
    """Return a pretty-printed table string."""
    title = f"  Backtest Metrics — {ticker}  " if ticker else "  Backtest Metrics  "
    rows = []
    for k, v in metrics.items():
        if isinstance(v, float):
            rows.append([k, f"{v:.2f}"])
        else:
            rows.append([k, str(v)])
    table = tabulate(rows, headers=["Metric", "Value"], tablefmt="grid")
    header = f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}"
    return header + "\n" + table


def metrics_comparison_table(all_metrics: dict[str, dict]) -> str:
    """Multi-ticker comparison table.  all_metrics is {ticker: metrics_dict}."""
    if not all_metrics:
        return ""

    keys = list(next(iter(all_metrics.values())).keys())
    headers = ["Metric"] + list(all_metrics.keys())
    rows = []
    for k in keys:
        row = [k]
        for ticker in all_metrics:
            v = all_metrics[ticker].get(k, "")
            row.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        rows.append(row)

    return "\n" + tabulate(rows, headers=headers, tablefmt="grid")
