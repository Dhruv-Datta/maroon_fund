.PHONY: setup train predict visualize future backtest all clean

DAYS ?= 60
TICKERS ?=
TRAIN_START ?=
TRAIN_END ?=
TEST_START ?=
TEST_END ?=
CAPITAL ?= 100000
BUY_THRESHOLD ?= 0.9
SELL_THRESHOLD ?= 0.75

setup:
	uv sync

train:
	uv run python combined_buy_sell_model/1_Train_Models.py

predict:
	uv run python combined_buy_sell_model/2_Loaded_Models.py

visualize:
	uv run python combined_buy_sell_model/3_Visualize.py

future:
	DAYS_AHEAD=$(DAYS) uv run python combined_buy_sell_model/4_Future_Predictions.py

backtest:
	uv run python combined_buy_sell_model/backtest/run_backtest.py \
		$(if $(TICKERS),--tickers $(TICKERS)) \
		$(if $(TRAIN_START),--train-start $(TRAIN_START)) \
		$(if $(TRAIN_END),--train-end $(TRAIN_END)) \
		$(if $(TEST_START),--test-start $(TEST_START)) \
		$(if $(TEST_END),--test-end $(TEST_END)) \
		$(if $(filter-out 100000,$(CAPITAL)),--capital $(CAPITAL)) \
		$(if $(filter-out 0.9,$(BUY_THRESHOLD)),--buy-threshold $(BUY_THRESHOLD)) \
		$(if $(filter-out 0.75,$(SELL_THRESHOLD)),--sell-threshold $(SELL_THRESHOLD))

all: train predict backtest

clean:
	find combined_buy_sell_model -name "*.joblib" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f combined_buy_sell_model/combined_analysis_*.csv
	rm -f combined_buy_sell_model/future_predictions_*.csv
	rm -rf combined_buy_sell_model/backtest/output
