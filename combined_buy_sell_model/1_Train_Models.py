"""
Train Both Buy and Sell Side Models from local combined data files.
"""

import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import load_config
from sell_trainer import StockSellSignalTrainer


def _require_file(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _safe_f1_scorer(estimator, X, y_true):
    """Robust scorer for folds where model may predict a single class."""
    y_pred = estimator.predict(X)
    return f1_score(y_true, y_pred, zero_division=0)


def _valid_time_series_splits(y: pd.Series, n_splits: int = 5):
    """Keep only folds where train/test each contain both classes."""
    splits = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(len(y))
    for train_idx, test_idx in splitter.split(indices):
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]
        if y_train_fold.nunique() < 2:
            continue
        if y_test_fold.nunique() < 2:
            continue
        splits.append((train_idx, test_idx))
    return splits


def main():
    cfg = load_config()
    model_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(model_dir, "data")

    buy_training_path = os.path.join(data_dir, "training_input.csv")
    sell_training_path = os.path.join(data_dir, "sell_signals.csv")

    _require_file(buy_training_path, "Buy training CSV")
    _require_file(sell_training_path, "Sell training CSV")

    print("=" * 60)
    print("TRAINING BUY SIDE MODEL")
    print("=" * 60)
    print(f"Loading buy side training data from: {buy_training_path}")

    buy_data = pd.read_csv(buy_training_path)
    buy_features = [col for col in buy_data.columns if col not in ["Date", "Target"]]
    X_buy = buy_data[buy_features]
    y_buy = buy_data["Target"]

    print(f"Buy side data shape: {buy_data.shape}")
    print(f"Buy side features: {len(buy_features)}")
    print(f"Buy side target distribution: {Counter(y_buy)}")

    class_counts = Counter(y_buy)
    neg = class_counts.get(0, 0)
    pos = class_counts.get(1, 0)
    if pos == 0:
        raise ValueError("Buy training CSV has no positive class (Target=1). Label at least one dip.")
    scale = neg / pos
    print(f"Class imbalance scale: {scale:.2f}")

    buy_pipeline = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "xgb",
                XGBClassifier(
                    eval_metric="logloss",
                    scale_pos_weight=scale,
                    max_depth=3,
                    n_estimators=100,
                    min_child_weight=5,
                    subsample=0.7,
                    colsample_bytree=0.7,
                ),
            ),
        ]
    )

    buy_param_grid = {
        "xgb__n_estimators": [100, 200],
        "xgb__max_depth": [3, 5],
    }

    buy_cv_splits = _valid_time_series_splits(y_buy, n_splits=5)
    if not buy_cv_splits:
        raise ValueError(
            "Unable to build valid time-series CV folds with both classes. "
            "Add more positive dip labels or widen the training date range."
        )
    buy_grid_search = GridSearchCV(
        estimator=buy_pipeline,
        param_grid=buy_param_grid,
        cv=buy_cv_splits,
        scoring=_safe_f1_scorer,
        n_jobs=-1,
        error_score="raise",
    )

    print("\nTraining buy side model...")
    buy_grid_search.fit(X_buy, y_buy)
    buy_model = buy_grid_search.best_estimator_
    print(f"Best buy params: {buy_grid_search.best_params_}")

    buy_last_train_idx, buy_last_test_idx = buy_cv_splits[-1]
    buy_X_test = X_buy.iloc[buy_last_test_idx]
    buy_y_test = y_buy.iloc[buy_last_test_idx]
    buy_y_pred_proba = buy_model.predict_proba(buy_X_test)[:, 1]
    buy_y_pred = buy_y_pred_proba > cfg.buy_threshold

    buy_accuracy = accuracy_score(buy_y_test, buy_y_pred)
    buy_roc_auc = roc_auc_score(buy_y_test, buy_y_pred_proba)
    print(f"Buy test accuracy: {buy_accuracy:.4f}")
    print(f"Buy test ROC AUC: {buy_roc_auc:.4f}")

    buy_model_path = os.path.join(model_dir, f"{cfg.ticker}_buy_model.joblib")
    joblib.dump(buy_model, buy_model_path)
    print(f"Saved buy model to: {buy_model_path}")

    print("\n" + "=" * 60)
    print("TRAINING SELL SIDE MODEL")
    print("=" * 60)
    print(f"Loading sell side training data from: {sell_training_path}")

    sell_trainer = StockSellSignalTrainer(
        data_path=sell_training_path,
        model_save_path=os.path.join(model_dir, f"{cfg.ticker}_sell_model.joblib"),
        threshold=cfg.sell_threshold,
    )

    (
        sell_trainer.load_and_prepare_data()
        .split_data()
        .balance_training_data()
        .train_model()
        .evaluate_model()
        .generate_predictions()
        .save_model()
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Buy model:  {buy_model_path}")
    print(f"Sell model: {sell_trainer.model_save_path}")


if __name__ == "__main__":
    main()
