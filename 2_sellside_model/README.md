# Sellside model (peak / sell-signal classifier)

XGBoost classifier for sell signals (peak detection) per ticker. Ticker and date range come from **x_utilities/utils.csv**.

## Inputs

- **User-selected peaks** – Via `1_sell_signal_input.py`, which saves to `sell_signals/{ticker}_sell_signals.csv`.
- **Existing CSVs** in `sell_signals/` – Used by `2_Model.py` to train per-ticker sell models.

## Outputs

- **sell_signals/{ticker}_sell_signals.csv** – Labeled sell-signal data per ticker.
- **sell_signals/{ticker}_sell_model.joblib** – Trained sell model per ticker.
- **Analysis CSVs** – From `3_Loaded_model.py` (e.g. `{ticker}_sell_analysis_*.csv`).

## Scripts

| File | Purpose |
|------|----------|
| **1_sell_signal_input.py** | Interactive chart for human peak-selecting; saves labeled data as `sell_signals/{ticker}_sell_signals.csv`. |
| **2_Model.py** | Trains XGBoost sell-signal model per ticker; reads from `sell_signals/`, saves `{ticker}_sell_model.joblib`. |
| **3_Loaded_model.py** | Loads trained sell model(s), runs predictions, writes analysis CSV. |
| **4_Visualize Prediction.py** | Displays charts with sell-signal predictions. |

## Note

The combined pipeline in **combined_buy_sell_model** uses sell signals and models from this folder. Multiple tickers are supported via the ticker list in `x_utilities/utils.csv`.
