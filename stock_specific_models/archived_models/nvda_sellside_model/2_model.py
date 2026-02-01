import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from collections import Counter
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from imblearn.pipeline import Pipeline as ImbPipeline

class StockSellSignalTrainer:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.features = []

    def load_and_prepare_data(self):
        data = pd.read_csv(self.data_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.fillna(0, inplace=True)
        
        self.features = [
            'Volatility', 'MA_25', 'Price_to_MA25', 'Bollinger_Band_Width',
            'ROC_14', 'Stochastic_K', 'RSI', 'One_Week_Growth',
            'Days_Since_Last_Peak', 'Decline_Since_Last_Peak', 'OBV',
            'Bearish_Divergence'
        ]
        
        self.X = data[self.features]
        self.y = data['Target']
        self.data = data
        
        balance = self.y.value_counts()
        print(f"Class balance: {dict(balance)}")
        self.scale_pos_weight = balance.loc[0] / balance.loc[1] if 1 in balance else 1
        print(f"Calculated scale_pos_weight: {self.scale_pos_weight:.2f}")
        return self

    def train(self):
        tscv = TimeSeriesSplit(n_splits=5)
        
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                scale_pos_weight=self.scale_pos_weight,
                random_state=42
            ))
        ])

        param_grid = {
            'xgb__n_estimators': [100, 200],
            'xgb__max_depth': [3, 4, 5],
            'xgb__learning_rate': [0.05],
            'xgb__subsample': [0.7, 0.8],
            'xgb__colsample_bytree': [0.7, 0.8],
            'xgb__min_child_weight': [3, 5],
            'xgb__gamma': [0.5, 1]
        }
        
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=tscv, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(self.X, self.y)
        self.model = grid_search.best_estimator_
        
        print(f"\nBest parameters found: {grid_search.best_params_}")
        
        _, test_indices = list(tscv.split(self.X))[-1]
        self.X_test = self.X.iloc[test_indices]
        self.y_test = self.y.iloc[test_indices]
        self.evaluate()
        return self

    def evaluate(self):
        proba = self.model.predict_proba(self.X_test)[:, 1]
        print("\n--- Model Evaluation ---")
        print(f"ROC AUC Score: {roc_auc_score(self.y_test, proba):.4f}\n")
        for threshold in [0.50, 0.60, 0.70, 0.80]:
            preds = proba > threshold
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, preds, average='binary', zero_division=0)
            print(f"Threshold: {threshold:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1-Score: {f1:.2f}")

    def save(self):
        joblib.dump(self.model, self.model_path)
        print(f"\nModel saved to '{self.model_path}'")
        return self

    def visualize_training_performance(self):
        self.data['Sell_Probability'] = self.model.predict_proba(self.X)[:, 1]
        self.data['Predicted_Signal'] = self.data['Sell_Probability'] > 0.70

        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax1.plot(self.data['Date'], self.data['Close'], label='Close Price', zorder=1)
        
        actual = self.data[self.data['Target'] == 1]
        ax1.scatter(actual['Date'], actual['Close'], marker='x', color='red', s=150, label='Actual Sell', zorder=6)
        
        predicted = self.data[self.data['Predicted_Signal']]
        ax1.scatter(predicted['Date'], predicted['Close'], marker='v', color='green', s=150, label='Predicted Sell (>{0.70*100:.0f}%)', zorder=5)
        
        ax1.set_title('Model Performance on Full Training Dataset'); ax1.set_xlabel('Date'); ax1.set_ylabel('Price')
        ax1.legend(); plt.xticks(rotation=45); fig.tight_layout(); plt.show()

if __name__ == "__main__":
    TRAINING_DATA = "nvda_sell_signals.csv"
    MODEL_OUTPUT = "nvda_sell_model.joblib"
    
    trainer = StockSellSignalTrainer(TRAINING_DATA, MODEL_OUTPUT)
    trainer.load_and_prepare_data().train().save().visualize_training_performance()