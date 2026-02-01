"""
Train Both Buy and Sell Side Models
===================================

This script trains both models using the existing training CSV files:
- Buy side: 1_buyside_model/training_input.csv
- Sell side: 2_sellside_model/nvda_sell_signals.csv
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from collections import Counter
import shutil
from pathlib import Path
import csv

# read what the user wrote
with open(os.path.abspath('./x_utilities/utils.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile)
    reader_indexed = []
    for i in reader:
        reader_indexed.append(i)

start_date = reader_indexed[1][1]
end_date   = reader_indexed[1][2]
t = reader_indexed[1][0]

ticker = t


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sell side trainer class - need to import differently
sell_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '2_sellside_model', '2_Model.py')
if os.path.exists(sell_model_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("sell_model_module", sell_model_path)
    sell_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sell_model_module)
    StockSellSignalTrainer = sell_model_module.StockSellSignalTrainer
else:
    # Fallback: define the class here if import fails
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    
    class StockSellSignalTrainer:
        def __init__(self, data_path, model_save_path='stock_sell_model.joblib'):
            self.data_path = data_path
            self.model_save_path = model_save_path
            self.model = None
            self.threshold = 0.75
            
        def load_and_prepare_data(self):
            data = pd.read_csv(self.data_path)
            data.ffill(inplace=True)
            data.bfill(inplace=True)
            data.fillna(0, inplace=True)
            self.features = [c for c in data.columns if c not in ['Date', 'Target']]
            self.X = data[self.features]
            self.y = data['Target']
            data['Date'] = pd.to_datetime(data['Date'])
            self.data = data
            return self
        
        def split_data(self, test_size=0.2):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
            )
            return self
        
        def balance_training_data(self):
            counts = self.y_train.value_counts()
            k = min(5, counts.min() - 1)
            if counts.min() > 5:
                sampler = SMOTE(random_state=42, k_neighbors=k)
            else:
                sampler = RandomOverSampler(random_state=42)
            self.X_train_balanced, self.y_train_balanced = sampler.fit_resample(
                self.X_train, self.y_train
            )
            return self
        
        def train_model(self):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }
            base_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42)
            grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
            grid_search.fit(self.X_train_balanced, self.y_train_balanced)
            self.model = grid_search.best_estimator_
            return self
        
        def evaluate_model(self):
            proba = self.model.predict_proba(self.X_test)[:, 1]
            preds = proba > self.threshold
            print(f"Test Accuracy: {accuracy_score(self.y_test, preds):.4f}")
            print(f"Test ROC AUC: {roc_auc_score(self.y_test, proba):.4f}")
            return self
        
        def generate_predictions(self):
            return self
        
        def save_model(self):
            joblib.dump(self.model, self.model_save_path)
            return self

print("="*60)
print("TRAINING BUY SIDE MODEL")
print("="*60)

# Load buy side training data
##### This data comes after running 1_manual_input.py #####
buy_training_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  '1_buyside_model', 'training_input.csv')
print(f"Loading buy side training data from: {buy_training_path}")

buy_data = pd.read_csv(buy_training_path)
buy_features = [col for col in buy_data.columns if col not in ['Date', 'Target']]
X_buy = buy_data[buy_features]
y_buy = buy_data['Target']
buy_data['Date'] = pd.to_datetime(buy_data['Date'])

print(f"Buy side data shape: {buy_data.shape}")
print(f"Buy side features: {len(buy_features)}")
print(f"Buy side target distribution: {Counter(y_buy)}")

# Compute class imbalance weight
neg, pos = Counter(y_buy).values()
scale = neg / pos
print(f"Class imbalance scale: {scale:.2f}")

# Build buy side pipeline
buy_pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale,
        max_depth=3,
        n_estimators=100,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.7
    ))
])

# Grid search for buy model
buy_param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5]
}

buy_tscv = TimeSeriesSplit(n_splits=5)
buy_grid_search = GridSearchCV(
    estimator=buy_pipeline,
    param_grid=buy_param_grid,
    cv=buy_tscv,
    scoring='f1',
    n_jobs=-1
)

print("\nTraining buy side model...")
buy_grid_search.fit(X_buy, y_buy)
buy_model = buy_grid_search.best_estimator_

print(f"✅ Buy model best parameters: {buy_grid_search.best_params_}")

# Evaluate buy model
buy_data['Dip_Probability'] = buy_model.predict_proba(X_buy)[:, 1]
buy_data['Predicted_Dip'] = buy_data['Dip_Probability'] > 0.9

# Metrics
buy_predicted_dips = buy_data[buy_data['Predicted_Dip']]
print(f"\n📊 BUY MODEL RESULTS:")
print(f"  Total data points: {len(buy_data)}")
print(f"  Actual dips: {y_buy.sum()}")
print(f"  Predicted dips: {len(buy_predicted_dips)}")
print(f"  Prediction rate: {len(buy_predicted_dips)/len(buy_data)*100:.2f}%")

# Evaluate on test set
buy_last_train_idx, buy_last_test_idx = list(buy_tscv.split(X_buy))[-1]
buy_X_test = X_buy.iloc[buy_last_test_idx]
buy_y_test = y_buy.iloc[buy_last_test_idx]
buy_y_pred_proba = buy_model.predict_proba(buy_X_test)[:, 1]
buy_y_pred = buy_y_pred_proba > 0.9

buy_accuracy = accuracy_score(buy_y_test, buy_y_pred)
buy_roc_auc = roc_auc_score(buy_y_test, buy_y_pred_proba)
print(f"  Test Accuracy: {buy_accuracy:.4f}")
print(f"  Test ROC AUC: {buy_roc_auc:.4f}")

# Save buy model
buy_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{t}_buy_model.joblib')
joblib.dump(buy_model, buy_model_path)
print(f"✅ Buy model saved to: {buy_model_path}")

print("\n" + "="*60)
print("TRAINING SELL SIDE MODEL")
print("="*60)

# Save utils.csv for visualizing later
script_dir = os.path.dirname(__file__)
source = os.path.join(script_dir, '..', 'x_utilities', 'utils.csv')
destination = os.path.join(script_dir, f'utils.csv')
source = os.path.normpath(source)
destination = os.path.normpath(destination)

shutil.copy2(source, destination)

# Load sell side training data
##### This data is obtained after running 1_input_stock_picker
sell_training_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   '2_sellside_model/sell_signals', 'sell_signals.csv')
print(f"Loading sell side training data from: {sell_training_path}")

# Use the sell side trainer class
sell_trainer = StockSellSignalTrainer(
    data_path=sell_training_path,
    model_save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{t}_sell_model.joblib')
)

(sell_trainer
    .load_and_prepare_data()
    .split_data()
    .balance_training_data()
    .train_model()
    .evaluate_model()
    .generate_predictions()
    .save_model()
)

sell_model = sell_trainer.model
sell_features = sell_trainer.features

# Get predictions for evaluation
sell_data = sell_trainer.data
sell_data['Sell_Probability'] = sell_model.predict_proba(sell_trainer.X)[:, 1]
sell_data['Sell_Signal'] = sell_data['Sell_Probability'] > 0.9

sell_predicted_signals = sell_data[sell_data['Sell_Signal']]
print(f"\n📊 SELL MODEL RESULTS:")
print(f"  Total data points: {len(sell_data)}")
print(f"  Actual sell opportunities: {sell_trainer.y.sum()}")
print(f"  Predicted sell signals: {len(sell_predicted_signals)}")
print(f"  Signal rate: {len(sell_predicted_signals)/len(sell_data)*100:.2f}%")

print(f"\n✅ Sell model saved to: {sell_trainer.model_save_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Buy model: {buy_model_path}")
print(f"Sell model: {sell_trainer.model_save_path}")
print("You can now use these models with 2_Loaded_Models.py")

# NOT WORKING
'''
# Copy all files to stock_specific_models
# Buy model
source_file = Path(__file__).parent / f'{t}_buy_model.joblib'
destination_file = Path(__file__).parent.parent / 'stock_specific_model' / f'{t}_model' / f'{t}_buy_model.joblib'
shutil.copy2(source_file, destination_file)
# Sell model
source_file = Path(__file__).parent / f'{t}_sell_model.joblib'
destination_file = Path(__file__).parent.parent / 'stock_specific_model' / f'{t}_model' / f'{t}_sell_model.joblib'
shutil.copy2(source_file, destination_file)
# CSV file
source_file = Path(__file__).parent / f'{t}_utils.csv'
destination_file = Path(__file__).parent.parent / 'stock_specific_model' / f'{t}_model' / f'{t}_utils.csv'
shutil.copy2(source_file, destination_file)
'''