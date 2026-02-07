# Buyside model (dip classifier)

XGBoost classifier for buy-the-dip signals. Ticker and date range come from **x_utilities/utils.csv** (or interactive fallback in `1_manual_input.py`).

## Inputs

- **training_input.csv** – Created by `1_manual_input.py` (manual dip labeling). Required for training and visualization steps.
- **xgboost_model.joblib** – Optional for steps 5–6; created by `4_model.py` after training.

## Outputs

- **training_input.csv** – Labeled dataset with engineered features and dip targets.
- **xgboost_model.joblib** – Trained XGBoost model (from `4_model.py`).
- **test_input.csv** – Hold-out dataset for validation (from `5_test_input.py`).

## Scripts

| File | Purpose |
|------|----------|
| **1_manual_input.py** | Interactive chart for human dip-selecting; saves cleaned, labeled data as `training_input.csv`. |
| **2_visualizing_input.py** | Reads `training_input.csv` and overlays dips on the price chart; verifies dip selections. |
| **3_visualizing_features.py** | Explores engineered features (moving averages, volatility, growth since last bottom, etc.). |
| **4_model.py** | Loads `training_input.csv`, balances with SMOTE, tunes XGBoost via GridSearchCV, saves `xgboost_model.joblib`. |
| **5_test_input.py** | Generates `test_input.csv` for a new date range not seen during training. |
| **6_loaded_model.py** | Loads the trained model and applies it to `test_input.csv`; outputs predictions and metrics. |

## Workflow

1. Label data – Run `1_manual_input.py` → creates `training_input.csv`.
2. Check labeling – Run `2_visualizing_input.py`.
3. Explore features – Run `3_visualizing_features.py`.
4. Train model – Run `4_model.py` → saves `xgboost_model.joblib`.
5. Generate test data – Run `5_test_input.py` → creates `test_input.csv`.
6. Validate on new data – Run `6_loaded_model.py`.

## Note

The combined pipeline in **combined_buy_sell_model** reads `1_buyside_model/training_input.csv` for buy-side training.
