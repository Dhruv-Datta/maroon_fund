# Gradient Boosting Decision Tree Quant Model Dev

This is the quant team development repo. Includes scripts for training, evaluating, and visualising an XGBoost-based quant model.

## Getting Started

```bash
# clone and enter the repo
git clone https://github.com/BDSQR/QR.git
cd QR

# create / activate a Python 3 virtual environment
python3 -m venv env
source env/bin/activate        # Windows + WSL2 users: run inside Ubuntu

# install Python dependencies
pip install -r requirements.txt

# one-time system packages (for Graphviz + compilers)
sudo apt update
sudo apt install python3-tk 
sudo apt install build-essential graphviz
```

You could also skip the venv and pip install all of the directories to your global pip path, but debugging tends to be more difficult that way. Using venv will likely be a simpler route to modularize your packages.

If you are adding any repositories, make sure to add them to the requirements.txt so everyone can install everything with one command.

```bash
# still inside the active env
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Pin updated dependencies"
```

## "Run and done" script: 0_run_and_done.py
This script combines the power of the buyside and sellside models by running all the necessary scripts and outputting a complete chart with buy and sell signals. You can select **custom stock and date ranges** when running. The scripts run include:

| File                            | Purpose                                                                                                                                                         |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`1_manual_input.py`**       | Interactive chart for **human dip-selecting**. Lets you click dips on historical stock charts and saves the cleaned & labelled dataset as `training_input.csv`. |
| **`1_sell_signal_input.py`**  | Interactive chart for **human peak-selecting**. Lets you click peaks on historical stock charts and saves the cleaned & labelled dataset as `sell_signals.csv`. |
| **`1_Train_Models.py`**       | Combines both buy and sell training models and saves the result as a .joblib                                                                                    |
| **`2_Loaded_Models.py`**      | Displays a chart complete with **buy and sell signals**. You can click the each symbol and see the probability and the date.                                    |
| **`4_Future_Predictions.py`** | (Work in progress). Specify a number of days in advance and the script will return a chart with probable future sell or buy signals.                            |

## Buyside Code Overview

| File                           | Purpose                                                                                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`1_manual_input.py`**        | Interactive chart for **human dip-selecting**. Lets you click dips on historical stock charts and saves the cleaned & labelled dataset as `training_input.csv`. |
| **`2_visualizing_input.py`**   | Reads a labelled CSV (e.g., `training_input.csv`) and visualises it with dips overlaid on the price chart. Helps verify your dip selections.          |
| **`3_visualizing_features.py`**| Explores engineered features (e.g., moving averages, volatility, growth since last bottom). Plots correlations and distributions before modeling.    |
| **`4_model.py`**               | Core training pipeline: loads `training_input.csv`, balances with SMOTE, tunes XGBoost via GridSearchCV, evaluates metrics, and saves the model.     |
| **`5_test_input.py`**          | Generates a `test_input.csv` for a new date range not seen during training. Lets you prepare hold-out datasets for model validation.                  |
| **`6_loaded_model.py`**        | Loads a pre-trained model (from `4_model.py`) and applies it to `test_input.csv`. Outputs predictions and evaluates generalization on unseen data.   |
| **`requirements.txt`**         | Pinned dependencies for reproducibility.                                                                                                             |
| **`training_input.csv`**       | Labeled training dataset created from `1_manual_input.py`.                                                                                            |
| **`test_input.csv`**           | Fresh dataset created from `5_test_input.py`, used for validation with `6_loaded_model.py`.                                                           |


## 1_manual_input.py

This script lets you visually select dips on a stock chart. By default, it fetches NVDA data from yfinance, engineers a large set of features (volatility, moving averages, stochastic oscillators, entropy, ulcer index, VIX levels, etc.), then launches a matplotlib UI for manual dip-marking.

### Key Features
- Interactive Dip Selector UI
    - Scroll to zoom, right-click to pan.
    - Left-click marks dips.
    - Undo button removes the last dip.
    - Done button closes the window and saves selections.
- Engineered Features
    - Daily return, volatility, moving averages (25 & 100), stochastic %K/%D, volatility z-score, ulcer index, entropy, VIX.
- Targets
    - Each clicked dip is assigned Target = 1.
    - All other points default to Target = 0.
- Output
    - Cleaned dataset saved as training_input.csv.
    - Includes all engineered features + user-selected dips.


```

At the end, the script saves to 'training_input.py'.

The following features are in the codebase:

```
Date, Close, Volatility, MA_25, MA_100,
Month_Growth_Rate, STOCH_%K, STOCH_%D,
Vol_zscore, Ulcer_Index, Return_Entropy_50d, VIX,
Growth_Since_Last_Bottom, Days_Since_Last_Bottom,
Week_Growth_Rate, Target
```

## 2_visualizing_input.py
Reads a CSV (e.g., `training_input.csv`) and overlays dips on the price chart.
Useful for validating dip selections. Prints summary stats such as count and % of dips.


## 3_visualizing_features.py
Helps you visualize and understand the features.


## 4_model.py
What does it do:
1. Loads historical_input.csv.
2. Splits data into train / test.
3. Balances rare dip events with SMOTE.
4. Performs GridSearchCV over XGBoost hyper-params.
5. Prints metrics, confusion matrix, feature importance.
6. Plots the price curve with red dots for ≥ 75 % predicted-dip probability
7. Shows a horizontal bar chart of feature importances
8. Charts the first boosted tree (graphviz required)

Some key things of note:
#### Training / Testing Data
The code reads the historical_input.csv file and splits the features into the collumn as follows:
```python
features = [col for col in data.columns if col not in ['Date', 'Target']]
X = data[features]
y = data['Target']
```
The X variable from here will be used to distinguish all of the features such as Close price, Volatility, etc., whereas y will only be used as a dip quantifier.

The data is then split into a training set (which will be used to train the model) and a testing set (which is a portion of data that is reserved and not to be used until the training is complete and the model is validating its model).

#### Handling Class Imbalance with SMOTE
Dip events are rare, so the raw dataset is heavily skewed toward the **non-dip (0)** class.  
To prevent the model from always predicting “no dip,” we oversample the minority class with **SMOTE**:

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```
This synthetically generates new dip samples by interpolating between nearest-neighbor points in feature space, giving the classifier a balanced view during training. 

#### Training the Model with Hyper-parameter Tuning
Instead of hand-picking XGBoost settings, the script runs a grid search across several key knobs:
```python
param_grid = {
    "n_estimators":   [200, 300, 400],
    "learning_rate":  [0.01, 0.05, 0.10],
    "max_depth":      [4, 5, 6],
    "min_child_weight":[1, 2, 3],
    "subsample":      [0.7, 0.8, 0.9],
    "colsample_bytree":[0.7, 0.8, 0.9],
}
```
Each parameter sequence is scored on accuracy and runs in parallel. The best estimator is stored in best_model and used for all subsequent predictions.

We then specify our base model and use XGBClassifier as the model used in grid search.
```python
# Create the base model
base_model = XGBClassifier()

# Perform Grid Search
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)
```
The training actually happens in the grid_search.fit() line. 

Subsequently the best model is given through the code:
```python
best_model = grid_search.best_estimator_
```

#### Prediction Threshold & Probability
After training, the model outputs a probability that each row is a dip:
```python
data["Dip_Probability"] = model.predict_proba(X)[:, 1]
data["Predicted_Dip"]   = data["Dip_Probability"] > 0.75
```
If the probability is above the threshold (which is 75% confidence as a default) red dot in the chart and it will be classified as a dip.

Adjust this value in model_2.0.py to trade off sensitivity vs precision.

#### Evaluation Metrics
On the held-out test set (20 % of the data):
- Accuracy – overall correct predictions
- Precision – of predicted dips, how many were true dips
- Recall – of all true dips, how many were found
- F1-score – harmonic mean of precision & recall
- ROC-AUC – probability the classifier ranks a random dip higher than a random non-dip
- Confusion matrix – True Positive / False Positive / True Negative / False Negative counts
- These metrics print to the console after training.


## 5_test_input.py
Generates a new CSV ('test_input.csv') for a date range not seen during training. However, it is important to note that the timeframe has to 

This ensures the evaluation measures true out-of-sample generalization.

## 6_loaded_model.py
Outputs:
- Predicted dip probabilities
- Binary dip predictions (default threshold = 0.75)
- Evaluation metrics on test data
- Plot of predictions vs price chart

## Workflow Summary
1. Label data – Run `1_manual_input.py` → creates training_input.csv.
2. Check labeling – Run `2_visualizing_input.py`.
3. Explore features – Run `3_visualizing_features.py`.
4. Train model – Run `4_model.py` → saves trained model.
5. Generate test data – Run `5_test_input.py` → creates test_input.csv.
6. Validate on new data – Run `6_loaded_model.py`.

