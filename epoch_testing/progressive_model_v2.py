import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory, panhandler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import glob

# Load and prepare the datasets
csv_files = ['input1.csv', 'input2.csv', 'input3.csv']  # Assumes your CSVs are named input1.csv, input2.csv, etc.
n_epochs = len(csv_files)

# Initialize a booster variable
booster = None

# Progressive Training Loop
for epoch, csv_file in enumerate(csv_files, 1):
    print(f"\nEpoch {epoch}/{n_epochs}")
    print(f"Training on file: {csv_file}")
    
    # Load the current dataset
    data = pd.read_csv(csv_file)
    
    # Prepare features and target
    features = [col for col in data.columns if col not in ['Date', 'Target']]
    X = data[features]
    y = data['Target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE to balance the dataset
    smote = SMOTE()
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Convert data into DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_balanced, label=y_train_balanced)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set up parameters for XGBoost
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.1 * (0.9 ** (epoch - 1)),  # Decay learning rate
        "max_depth": 5,
        "reg_alpha": 1,
        "reg_lambda": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    
    # Train the model
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dtest, "test")],
        early_stopping_rounds=10,
        xgb_model=booster,  # Load the previous booster if not the first epoch
        verbose_eval=True,
    )
    
    # Evaluate the model for this epoch
    y_pred_proba = booster.predict(dtest)
    y_pred = y_pred_proba > 0.75  # Custom threshold for 'dip'
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nEpoch {epoch} Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    # Final evaluation and visualization (using the last dataset)
    data['Dip_Probability'] = booster.predict(xgb.DMatrix(X))  # Get probability predictions
    data['Predicted_Dip'] = data['Dip_Probability'] > 0.75  # Custom threshold for 'dip'


    # Convert 'Date' column to datetime for plotting
    data['Date'] = pd.to_datetime(data['Date'])

    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot the closing price line
    ax.plot(data['Date'], data['Close'], label='Close Price', color='blue')

    # Add red dots for predicted dips
    predicted_dips = data[data['Predicted_Dip']]
    ax.scatter(predicted_dips['Date'], predicted_dips['Close'], color='red', label='Predicted Dip (>75% confidence)', zorder=5)

    # Customize the plot
    ax.set_title('Stock Price with Predicted Dips')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add zooming and panning functionality using mpl_interactions
    zoom_factory(ax)
    panhandler(fig)

    # Show the plot
    plt.show()

    # Evaluate the model on test data
    y_pred_proba = booster.predict(dtest)
    y_pred = y_pred_proba > 0.75

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Comprehensive Model Performance Report
    print("\n" + "="*50)
    print("Comprehensive Model Performance Report")
    print("="*50)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Additional Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Save the model
    booster.save_model('xgboost_model.json')
    print("Model saved successfully.")

