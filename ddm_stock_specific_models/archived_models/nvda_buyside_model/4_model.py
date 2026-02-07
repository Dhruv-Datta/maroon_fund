import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory, panhandler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import joblib
from collections import Counter

# ------------------ Load and Prepare Data ------------------
data = pd.read_csv("training_input.csv")
features = [col for col in data.columns if col not in ['Date', 'Target']]
X = data[features]
y = data['Target']
data['Date'] = pd.to_datetime(data['Date'])  # Ensure date is datetime

# ------------------ Compute Class Imbalance Weight ------------------
neg, pos = Counter(y).values()
scale = neg / pos

# ------------------ Build Pipeline (No SMOTE) ------------------
pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale,
        max_depth=3,
        n_estimators=100,
        min_child_weight=5,        # less likely to split on small noisy patterns
        subsample=0.7,             # makes trees less sensitive to outliers
        colsample_bytree=0.7       # uses only part of features per tree
    ))
])


param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5]
}

tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring='f1',
    n_jobs=-1
)

# ------------------ Fit Model ------------------
grid_search.fit(X, y)
model = grid_search.best_estimator_

# ------------------ Predict on Full Dataset ------------------
data['Dip_Probability'] = model.predict_proba(X)[:, 1]
data['Predicted_Dip'] = data['Dip_Probability'] > 0.75

# ------------------ Plot Dips ------------------
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(data['Date'], data['Close'], label='Close Price', color='blue')
predicted_dips = data[data['Predicted_Dip']]
ax.scatter(predicted_dips['Date'], predicted_dips['Close'], color='red', label='Predicted Dip (>75%)', zorder=5)
ax.set_title('Stock Price with Predicted Dips')
ax.set_xlabel('Date'); ax.set_ylabel('Price'); ax.legend()
plt.xticks(rotation=45); plt.tight_layout()
zoom_factory(ax); panhandler(fig)
plt.show()

# ------------------ Evaluate Model on Last Fold ------------------
last_train_idx, last_test_idx = list(tscv.split(X))[-1]
X_test = X.iloc[last_test_idx]
y_test = y.iloc[last_test_idx]
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = y_pred_proba > 0.75

# ------------------ Metrics ------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nRecent Data Points:")
print(data.tail(5)[['Date', 'Close'] + features + ['Dip_Probability']])
print("\nNote: Model considers it a dip if probability > 0.75")

# ------------------ Extra Metrics ------------------
print("\n" + "="*50)
print("Comprehensive Model Performance Report")
print("="*50)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nPrecision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# ------------------ Feature Importance ------------------
feature_importance = model.named_steps['xgb'].feature_importances_
sorted_fi = sorted(zip(features, feature_importance), key=lambda x: x[1], reverse=True)
print("\nFeature Importance:")
for f, imp in sorted_fi:
    print(f"{f}: {imp:.4f}")

# ------------------ Prediction for Most Recent Day ------------------
most_recent = data.iloc[-1]
recent_features = most_recent[features].values.reshape(1, -1)
recent_prob = model.predict_proba(recent_features)[0, 1]
is_dip = recent_prob > 0.75

print("\n" + "="*50)
print("Prediction for the Most Recent Day")
print("="*50)
print(f"Date: {most_recent['Date']}")
print(f"Close Price: {most_recent['Close']:.2f}")
print(f"Dip Probability: {recent_prob:.4f}")
print(f"Is it a dip? {'Yes' if is_dip else 'No'}")
print(f"Confidence: {'High' if abs(recent_prob - 0.5) > 0.3 else 'Low'}")

# ------------------ Save Model and Visualize ------------------
joblib.dump(model, 'nvda_model.joblib')
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.title('Feature Importance')
plt.show()

booster = model.named_steps['xgb'].get_booster()
booster.feature_names = features
plt.figure(figsize=(20, 12))
xgb.plot_tree(booster, num_trees=0, rankdir='LR')
plt.title('XGBoost Tree (With Feature Names)')
plt.show()
