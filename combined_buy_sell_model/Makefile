.PHONY: setup generate-buy-train generate-buy-test generate-sell-train generate-data train predict visualize future all clean

UV_RUN = uv run --with-requirements requirements.txt

setup:
	uv sync
	uv pip install -r requirements.txt

generate-buy-train:
	$(UV_RUN) python 0_Generate_Buy_Training_Data.py

generate-buy-test:
	$(UV_RUN) python 0_Generate_Buy_Test_Data.py

generate-sell-train:
	$(UV_RUN) python 0_Generate_Sell_Training_Data.py

generate-data: generate-buy-train generate-sell-train generate-buy-test

train:
	$(UV_RUN) python 1_Train_Models.py

predict:
	$(UV_RUN) python 2_Loaded_Models.py

visualize:
	$(UV_RUN) python 3_Visualize.py

future:
	$(UV_RUN) python 4_Future_Predictions.py

all: generate-data train predict

clean:
	find . -name "*.joblib" -delete
	rm -f combined_analysis_*.csv future_predictions_*.csv
	rm -f data/*.csv
