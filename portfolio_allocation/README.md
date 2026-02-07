# Portfolio allocation

Portfolio optimization using a mean–variance style approach with factor-based risk. This is not an ML model: it uses Monte Carlo simulation over portfolio weights subject to constraints, then selects efficient or maximum-Sharpe portfolios.

## Inputs

Configured at the top of `portfolio_allocation.py`:

- **Symbols** – List of tickers (e.g. GOOGL, AMZN, CASH).
- **Expected returns** – Per-asset expected return array.
- **Risk-free rate** – For Sharpe ratio.
- **Min / max weights** – Per-asset and for cash (e.g. `min_weight`, `max_weight`, `cash_min_weight`, `cash_max_weight`).
- **Risk factors** – Factor names (e.g. Volatility, Regulatory, Disruption, Valuation, Earnings Quality).
- **Risk factor weights** – Diagonal weights for the factor covariance.
- **Factor exposures** – Per-ticker exposure to each risk factor.
- **Num of portfolios** – Number of Monte Carlo portfolio samples.

## Outputs

- **Optimal / efficient portfolios** – Weights, return, volatility, Sharpe ratio; visualizations (e.g. efficient frontier).
- **Rebalancing** – `rebalancing.py` turns target weights into a step-by-step execution plan (buy/sell dollars, transaction costs).

## Files

| File | Purpose |
|------|----------|
| **portfolio_allocation.py** | Main script: builds factor exposure matrix, runs Monte Carlo over weights, computes risk/return/Sharpe, and produces results and plots. |
| **rebalancing.py** | Rebalance logic: given current positions and target weights, returns an execution plan (steps, buy/sell dollars, final values). |
