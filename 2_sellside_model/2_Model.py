import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import os, csv

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

class StockSellSignalTrainer:
    def __init__(self, data_path, model_save_path='stock_sell_model.joblib'):
        # file paths and decision threshold
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.model = None
        self.threshold = 0.75
        
    def load_and_prepare_data(self):
        # load CSV and fill any missing entries
        data = pd.read_csv(self.data_path)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.fillna(0, inplace=True)
        
        # define feature matrix X and target vector y
        self.features = [c for c in data.columns if c not in ['Date', 'Target']]
        self.X = data[self.features]
        self.y = data['Target']
        
        # keep full dataset with parsed dates
        data['Date'] = pd.to_datetime(data['Date'])
        self.data = data
        
        # basic sanity checks
        self._validate_data()
        return self
    
    def _validate_data(self):
        # warn if any NaNs or severe class imbalance
        nan_x = self.X.isna().sum().sum()
        nan_y = self.y.isna().sum()
        if nan_x or nan_y:
            print(f"Warning: {nan_x} feature NaNs, {nan_y} target NaNs")
        
        balance = self.y.value_counts(normalize=True)
        print(f"Class counts: {dict(self.y.value_counts())}")
        if balance.min() < 0.05:
            print("Warning: severe class imbalance")
    
    def split_data(self, test_size=0.2):
        # stratified train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size,
            random_state=42, stratify=self.y
        )
        return self
    
    def balance_training_data(self):
        # choose SMOTE or oversampling based on sample count
        counts = self.y_train.value_counts()
        k = min(5, counts.min() - 1)
        if counts.min() > 5:
            sampler = SMOTE(random_state=42, k_neighbors=k)
            print(f"Applied SMOTE (k={k})")
        else:
            sampler = RandomOverSampler(random_state=42)
            print("Applied RandomOverSampler")
        
        self.X_train_balanced, self.y_train_balanced = sampler.fit_resample(
            self.X_train, self.y_train
        )
        print(f"Balanced class counts: {dict(pd.Series(self.y_train_balanced).value_counts())}")
        return self
    
    def train_model(self):
        # hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }

        # XGBClassifier without deprecated params
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',  # binary classification
            eval_metric='auc',            # use AUC for early stopping
            random_state=42
        )

        # grid-search over hyperparameters
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        print("Starting hyperparameter search...")
        grid_search.fit(self.X_train_balanced, self.y_train_balanced)

        # save best estimator
        self.model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        return self
    
    def evaluate_model(self):
        # performance metrics on hold‑out set
        if self.model is None:
            raise ValueError("Call train_model() first")
        
        proba = self.model.predict_proba(self.X_test)[:, 1]
        preds = proba > self.threshold
        
        print("\n=== EVALUATION ===")
        print(f"Accuracy: {accuracy_score(self.y_test, preds):.4f}")
        print(f"ROC AUC: {roc_auc_score(self.y_test, proba):.4f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, preds))
        print("\nClassification Report:\n", classification_report(self.y_test, preds))
        
        # display which features mattered most
        self._analyze_feature_importance()
        return self
    
    def _analyze_feature_importance(self):
        # list features by descending importance
        imp = self.model.feature_importances_
        print("\n=== FEATURE IMPORTANCE ===")
        for feat, score in sorted(zip(self.features, imp), key=lambda x: x[1], reverse=True):
            print(f"{feat}: {score:.4f}")
    
    def generate_predictions(self):
        # annotate full dataset with sell probabilities and signals
        if self.model is None:
            raise ValueError("Call train_model() first")
        
        self.data['Sell_Probability'] = self.model.predict_proba(self.X)[:, 1]
        self.data['Sell_Signal'] = self.data['Sell_Probability'] > self.threshold
        
        # show the latest recommendation
        latest = self.data.iloc[-1]
        print("\n=== LATEST PREDICTION ===")
        print(f"Date: {latest['Date'].date()} | Close: ${latest['Close']:.2f}")
        print(f"Prob: {latest['Sell_Probability']:.4f} → {'SELL' if latest['Sell_Signal'] else 'HOLD'}")
        return self
    
    def visualize_results(self):
        # plot price with predicted and actual sell points
        fig, ax1 = plt.subplots(figsize=(15, 8))
        ax1.plot(self.data['Date'], self.data['Close'], label='Close Price')
        
        sig = self.data[self.data['Sell_Signal']]
        ax1.scatter(sig['Date'], sig['Close'], marker='v', s=100, label='Predicted Sell')
        
        if 'Target' in self.data:
            actual = self.data[self.data['Target'] == 1]
            ax1.scatter(actual['Date'], actual['Close'], marker='o', alpha=0.5, label='Actual Sell')
        
        ax2 = ax1.twinx()
        ax2.fill_between(self.data['Date'], self.data['Sell_Probability'], alpha=0.2, label='Sell Prob')
        ax2.set_ylim(0, 1)
        
        ax1.set_xlabel('Date'); ax1.set_ylabel('Price'); ax2.set_ylabel('Probability')
        ax1.grid(alpha=0.3)
        ax1.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return self
    
    def visualize_feature_importance(self):
        # horizontal bar chart of feature importance
        if self.model is None:
            raise ValueError("Call train_model() first")
        
        imp = self.model.feature_importances_
        idx = np.argsort(imp)
        feats = [self.features[i] for i in idx]
        vals = imp[idx]
        
        plt.figure(figsize=(10, max(4, len(feats)*0.3)))
        plt.barh(range(len(feats)), vals, align='center')
        plt.yticks(range(len(feats)), feats)
        plt.xlabel('Importance'); plt.title('Feature Importances')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        return self
    
    def save_model(self):
        # write trained model to disk
        if self.model is None:
            raise ValueError("Call train_model() first")
        
        joblib.dump(self.model, self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        return self


if __name__ == "__main__":
    trainer = StockSellSignalTrainer(f"2_sellside_model/sell_signals/{t}_sell_signals.csv", f"2_sellside_model/sell_signals/{t}_sell_model.joblib")
    (trainer
        .load_and_prepare_data()
        .split_data()
        .balance_training_data()
        .train_model()
        .evaluate_model()
        .generate_predictions()
        .visualize_results()
        .visualize_feature_importance()
        .save_model()
    )
    print("\nTraining and evaluation complete.")