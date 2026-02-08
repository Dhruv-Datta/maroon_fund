#!/usr/bin/env python3
"""
CLI entry point for running the backtest pipeline.

Usage examples:
    # Use defaults from x_utilities/utils.csv
    python combined_buy_sell_model/backtest/run_backtest.py

    # Override everything
    python combined_buy_sell_model/backtest/run_backtest.py \
        --tickers NVDA,AAPL \
        --train-start 2020-01-01 --train-end 2023-12-31 \
        --test-start 2024-01-01 --test-end 2025-01-01 \
        --capital 100000 --buy-threshold 0.9 --sell-threshold 0.75
"""

import argparse
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Resolve paths relative to this file
_this_dir = os.path.dirname(os.path.abspath(__file__))
_combined_dir = os.path.dirname(_this_dir)
_repo_root = os.path.dirname(_combined_dir)

sys.path.insert(0, _combined_dir)

from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, format_metrics_table, metrics_comparison_table
from backtest.visualize import save_all_charts


def _read_utils_csv() -> dict:
    """Read defaults from x_utilities/utils.csv."""
    utils_path = os.path.join(_repo_root, "x_utilities", "utils.csv")
    if not os.path.exists(utils_path):
        return {}
    with open(utils_path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {}
    return {
        "ticker": rows[1][0],
        "start": rows[1][1],
        "end": rows[1][2],
    }


def main():
    defaults = _read_utils_csv()

    parser = argparse.ArgumentParser(description="Run backtest on buy/sell signal model")
    parser.add_argument(
        "--tickers", default=defaults.get("ticker", "NVDA"),
        help="Comma-separated ticker(s), e.g. NVDA,AAPL (default: from utils.csv)",
    )
    parser.add_argument("--train-start", default=defaults.get("start", "2020-01-01"))
    parser.add_argument("--train-end", default=defaults.get("end", "2023-12-31"))
    parser.add_argument("--test-start", default=defaults.get("end", "2024-01-01"))
    parser.add_argument("--test-end", default="2025-01-01")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--buy-threshold", type=float, default=0.9)
    parser.add_argument("--sell-threshold", type=float, default=0.75)
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]

    print("=" * 60)
    print("BACKTEST PIPELINE")
    print("=" * 60)
    print(f"  Tickers      : {', '.join(tickers)}")
    print(f"  Train period : {args.train_start} to {args.train_end}")
    print(f"  Test period  : {args.test_start} to {args.test_end}")
    print(f"  Capital      : ${args.capital:,.0f}")
    print(f"  Buy thresh   : {args.buy_threshold}")
    print(f"  Sell thresh  : {args.sell_threshold}")
    print("=" * 60)

    all_metrics = {}

    for ticker in tickers:
        print(f"\n{'—' * 60}")
        print(f"  Running backtest for {ticker}")
        print(f"{'—' * 60}")

        engine = BacktestEngine(
            ticker=ticker,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            initial_capital=args.capital,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )

        result = engine.run()
        metrics = compute_metrics(result)
        all_metrics[ticker] = metrics

        print(format_metrics_table(metrics, ticker))
        print(f"\n  Trades: {len(result.trades)}")
        for i, t in enumerate(result.trades, 1):
            direction = "+" if t.pnl > 0 else ""
            print(
                f"    #{i}: {t.entry_date.strftime('%Y-%m-%d')} @ ${t.entry_price:.2f} -> "
                f"{t.exit_date.strftime('%Y-%m-%d') if t.exit_date else 'OPEN'} "
                f"@ ${t.exit_price:.2f if t.exit_price else 0:.2f}  "
                f"P&L: {direction}${t.pnl:.2f} ({direction}{t.pnl_pct*100:.1f}%)  "
                f"Hold: {t.holding_days}d"
            )

        # Save outputs
        out_dir = os.path.join(_this_dir, "output", ticker)
        save_all_charts(result, out_dir)

        # Save trades CSV
        if result.trades:
            import pandas as pd
            trades_df = pd.DataFrame([
                {
                    "entry_date": t.entry_date,
                    "entry_price": t.entry_price,
                    "exit_date": t.exit_date,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "holding_days": t.holding_days,
                }
                for t in result.trades
            ])
            trades_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)

        # Save metrics JSON
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    # Multi-ticker comparison
    if len(tickers) > 1:
        print(metrics_comparison_table(all_metrics))

    print("\nBacktest complete!")


if __name__ == "__main__":
    main()
