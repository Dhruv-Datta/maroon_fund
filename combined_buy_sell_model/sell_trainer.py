import joblib
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split


class StockSellSignalTrainer:
    def __init__(self, data_path, model_save_path="stock_sell_model.joblib", threshold=0.75):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.model = None
        self.threshold = threshold

    def load_and_prepare_data(self):
        data = pd.read_csv(self.data_path)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.fillna(0, inplace=True)

        self.features = [c for c in data.columns if c not in ["Date", "Target"]]
        self.X = data[self.features]
        self.y = data["Target"]

        data["Date"] = pd.to_datetime(data["Date"])
        self.data = data
        self._validate_data()
        return self

    def _validate_data(self):
        balance = self.y.value_counts(normalize=True)
        print(f"Class counts: {dict(self.y.value_counts())}")
        if balance.min() < 0.05:
            print("Warning: severe class imbalance")

    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=42,
            stratify=self.y,
        )
        return self

    def balance_training_data(self):
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
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2],
        }

        base_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        print("Starting hyperparameter search...")
        grid_search.fit(self.X_train_balanced, self.y_train_balanced)
        self.model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        return self

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Call train_model() first")

        proba = self.model.predict_proba(self.X_test)[:, 1]
        preds = proba > self.threshold

        print("\n=== EVALUATION ===")
        print(f"Accuracy: {accuracy_score(self.y_test, preds):.4f}")
        print(f"ROC AUC: {roc_auc_score(self.y_test, proba):.4f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, preds))
        print("\nClassification Report:\n", classification_report(self.y_test, preds))
        return self

    def generate_predictions(self):
        if self.model is None:
            raise ValueError("Call train_model() first")

        self.data["Sell_Probability"] = self.model.predict_proba(self.X)[:, 1]
        self.data["Sell_Signal"] = self.data["Sell_Probability"] > self.threshold

        latest = self.data.iloc[-1]
        print("\n=== LATEST PREDICTION ===")
        print(f"Date: {latest['Date'].date()} | Close: ${latest['Close']:.2f}")
        print(f"Prob: {latest['Sell_Probability']:.4f} -> {'SELL' if latest['Sell_Signal'] else 'HOLD'}")
        return self

    def save_model(self):
        if self.model is None:
            raise ValueError("Call train_model() first")
        joblib.dump(self.model, self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        return self
