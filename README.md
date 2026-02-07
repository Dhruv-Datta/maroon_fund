# Maroon Fund

Quantitative research and trading models for the Maroon Fund quant team. This repository contains scripts for training, evaluating, and visualizing ML and optimization models.

## Models

| Model | Location | Purpose |
|-------|----------|---------|
| **Buyside (dip)** | [1_buyside_model/](1_buyside_model/) | XGBoost buy-the-dip classifier; manual labeling, SMOTE, GridSearchCV |
| **Sellside (peak)** | [2_sellside_model/](2_sellside_model/) | XGBoost sell-signal classifier; per-ticker peak labeling |
| **Combined buy + sell** | [combined_buy_sell_model/](combined_buy_sell_model/) | Unified pipeline; trains and runs both models together |
| **LSTM quantile** | [lstm_model/](lstm_model/) | LSTM + quantile regression for price percentiles (pinball loss, non-crossing quantiles) |
| **Portfolio allocation** | [portfolio_allocation/](portfolio_allocation/) | Mean-variance optimization with factor risk; Monte Carlo, rebalancing |
| **Stock-specific artifacts** | [ddm_stock_specific_models/](ddm_stock_specific_models/) | Archived per-ticker buy/sell joblibs; not used by the main pipeline |

## Configuration

Ticker and date range for the main pipelines are set in **x_utilities/utils.csv**.

## Getting Started

Create a Python virtual environment and install dependencies from the `requirements.txt` in the model directory you intend to run (e.g. `1_buyside_model/requirements.txt`, `combined_buy_sell_model/requirements.txt`). See each model's README for workflow and usage instructions.

```bash
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
cd <model_directory>
pip install -r requirements.txt
```

## For More Detail

Each model folder has its own README with script-by-script workflow, inputs/outputs, and usage. Start with [1_buyside_model/README.md](1_buyside_model/README.md), [2_sellside_model/README.md](2_sellside_model/README.md), or [combined_buy_sell_model/README.md](combined_buy_sell_model/README.md) depending on your workflow.
