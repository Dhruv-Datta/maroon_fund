# Combined Buy + Sell Pipeline (Self-Contained)

This folder now contains a full, local pipeline for data generation, training, prediction, and visualization.

## Configuration

All ticker and date ranges are set in:

- `combined_buy_sell_model/config.yaml`

Example keys:

- `ticker`
- `train_start_date`
- `train_end_date`
- `test_start_date`
- `test_end_date`
- `buy_threshold`
- `sell_threshold`

## Data Files (Local)

Generated under `combined_buy_sell_model/data/`:

- `training_input.csv` (buy-side labeled training data)
- `sell_signals.csv` (sell-side labeled training data)
- `test_input.csv` (buy-side labeled test data)

## Scripts

- `0_Generate_Buy_Training_Data.py` : Interactive dip labeling (same feature logic as original buyside manual input flow).
- `0_Generate_Sell_Training_Data.py` : Interactive peak labeling (same feature logic as original sellside input flow).
- `0_Generate_Buy_Test_Data.py` : Interactive test data labeling for buy-side holdout data.
- `1_Train_Models.py` : Train both buy/sell models from local `data/*.csv`.
- `2_Loaded_Models.py` : Run combined predictions and write combined analysis CSV.
- `3_Visualize.py` : Separate buy/sell visualizations.
- `4_Future_Predictions.py` : Prediction run over the configured test date window.

## Makefile Commands

Run from `combined_buy_sell_model/`.

- `make setup` : install/sync dependencies.
- `make generate-buy-train` : create `data/training_input.csv` (interactive).
- `make generate-sell-train` : create `data/sell_signals.csv` (interactive).
- `make generate-buy-test` : create `data/test_input.csv` (interactive).
- `make generate-data` : run all generation steps.
- `make train` : train and save models.
- `make predict` : run combined analysis.
- `make visualize` : show individual model visualizations.
- `make future` : run predictions for `test_start_date` to `test_end_date`.
- `make all` : generate data, train, and run combined prediction.

## Notes

- Data generation steps are interactive and require manually selecting dips/peaks in charts.
- Models are saved in this folder as `{ticker}_buy_model.joblib` and `{ticker}_sell_model.joblib`.
