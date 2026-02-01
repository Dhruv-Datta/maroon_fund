# Combined Buy & Sell Side Model

This folder contains the combined trading system that integrates both buy and sell side models.

## 📁 File Structure

```
combined_buy_sell_model/
├── 1_Train_Models.py        # Train both models from existing CSV files
├── 2_Loaded_Models.py        # Load trained models and run combined analysis
├── 3_Visualize.py            # Individual model visualizations
├── 4_Future_Predictions.py   # Future prediction capabilities
├── README.md                 # This file
├── buy_model.joblib          # Trained buy model (created after training)
└── sell_model.joblib         # Trained sell model (created after training)
```

## 🚀 Quick Start

### Step 1: Train the Models
```bash
cd combined_buy_sell_model
python 1_Train_Models.py
```

This will:
- Load buy side training data from `../1_buyside_model/training_input.csv`
- Load sell side training data from `../2_sellside_model/nvda_sell_signals.csv`
- Train both models with proper hyperparameter tuning
- Save models as `buy_model.joblib` and `sell_model.joblib`

### Step 2: Run Combined Analysis
```bash
python 2_Loaded_Models.py
```

This will:
- Load the trained models
- Run predictions on test data
- Show combined visualization with buy and sell signals
- Save results to CSV

### Step 3: Visualize Individual Models
```bash
python 3_Visualize.py
```

This will:
- Show buy side model visualization separately
- Show sell side model visualization separately
- Display predictions and probabilities for each model

### Step 4: Run Future Predictions
```bash
python 4_Future_Predictions.py
```

This will:
- Predict buy/sell signals for the next 30 days
- Show visualization with highlighted prediction period
- Save future predictions to CSV

## 📊 Training Data

The models use the existing training CSV files:

- **Buy Side**: `../1_buyside_model/training_input.csv`
- **Sell Side**: `../2_sellside_model/nvda_sell_signals.csv`

These files contain pre-computed features and labeled targets, ensuring accurate model training.

## 🎯 Features

### 1. Training (`1_Train_Models.py`)
- Uses actual training CSV files (not auto-generated)
- Proper hyperparameter tuning with grid search
- Time series cross-validation
- Class imbalance handling
- Model evaluation metrics

### 2. Combined Analysis (`2_Loaded_Models.py`)
- Loads trained models
- Runs both buy and sell predictions
- Creates unified visualization
- Shows both signals on same chart
- Saves results to CSV

### 3. Individual Visualization (`3_Visualize.py`)
- Separate charts for buy side model
- Separate charts for sell side model
- Detailed prediction information
- Probability overlays

### 4. Future Predictions (`4_Future_Predictions.py`)
- Predicts next N days (default: 30)
- Uses most recent market data
- Highlights prediction period
- Shows confidence levels

## 📈 Output

Each script generates:
- **Interactive Plotly Charts**: Zoom, pan, hover for details
- **CSV Files**: Complete prediction data with probabilities
- **Console Output**: Statistics and summary information

## ⚙️ Customization

### Modify Parameters

Edit the default parameters in each script:

**2_Loaded_Models.py:**
```python
ticker = "NVDA"
start_date = "2024-01-02"
end_date = "2025-07-25"
```

**3_Visualize.py:**
```python
ticker = "NVDA"
start_date = "2024-01-02"
end_date = "2025-07-25"
```

**4_Future_Predictions.py:**
```python
ticker = "NVDA"
days_ahead = 30
```

### Thresholds

- **Buy Signal Threshold**: 0.9 (90% confidence) - in prediction code
- **Sell Signal Threshold**: 0.75 (75% confidence) - in prediction code

## 🔧 Requirements

Same requirements as the individual models:
- pandas
- numpy
- plotly
- matplotlib
- yfinance
- scikit-learn
- xgboost
- imbalanced-learn
- ta
- scipy

## 📝 Notes

1. **Always train first**: Run `1_Train_Models.py` before using other scripts
2. **Models are saved locally**: Trained models are saved in this folder
3. **Uses actual training data**: Models use the pre-labeled CSV files, not auto-generated targets
4. **Organized structure**: Follows same numbering pattern as individual model folders

## 🎉 Benefits

✅ Uses actual training data files  
✅ Proper model training with validation  
✅ Individual and combined visualizations  
✅ Future prediction capabilities  
✅ Organized folder structure  
✅ Consistent with existing codebase  

This combined system maintains the same logic and accuracy as the separate models while providing a unified interface for all operations!
