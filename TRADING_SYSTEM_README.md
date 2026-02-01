# Comprehensive Trading System

A complete trading system that combines buy and sell side models with training, visualization, and prediction capabilities.

## 🚀 Features

### 1. **Model Training**
- **Buy Side Model**: Identifies potential buying opportunities (dips)
- **Sell Side Model**: Identifies potential selling opportunities (peaks)
- **Automatic Feature Engineering**: Computes all necessary technical indicators
- **Class Imbalance Handling**: Uses appropriate weighting for imbalanced datasets

### 2. **Individual Model Visualization**
- **Buy Side Charts**: Shows stock price with predicted buy signals
- **Sell Side Charts**: Shows stock price with predicted sell signals
- **Probability Overlays**: Displays model confidence levels
- **Interactive Plotly Charts**: Zoom, pan, and hover for detailed analysis

### 3. **Future Predictions**
- **Next N Days**: Predicts buy/sell signals for upcoming days
- **Confidence Levels**: Shows probability scores for each prediction
- **Recent Data Analysis**: Uses latest market data for predictions

### 4. **Combined Analysis**
- **Unified View**: Shows both buy and sell signals on the same chart
- **Signal Overlap Detection**: Identifies conflicting signals
- **Comprehensive Metrics**: Detailed statistics and performance analysis

## 📁 Files Overview

### Core System Files
- `comprehensive_trading_system.py` - Main system class with all functionality
- `trading_system_orchestrator.py` - Command-line interface for all operations
- `run_trading_system.py` - Interactive menu-driven interface
- `demo_trading_system.py` - Complete demonstration of all features

### Original Model Files
- `1_buyside_model/` - Original buy side model files
- `2_sellside_model/` - Original sell side model files
- `combined_buy_sell_model.py` - Simple combined model (original request)

## 🛠️ Usage

### Option 1: Interactive Menu
```bash
python run_trading_system.py
```
Choose from:
1. Train both models
2. Visualize individual models
3. Run future predictions
4. Run combined analysis
5. Run everything

### Option 2: Command Line
```bash
# Train models
python trading_system_orchestrator.py --mode train --ticker NVDA

# Visualize models
python trading_system_orchestrator.py --mode visualize --ticker NVDA

# Future predictions
python trading_system_orchestrator.py --mode predict --ticker NVDA --days_ahead 30

# Combined analysis
python trading_system_orchestrator.py --mode combined --ticker NVDA

# Run everything
python trading_system_orchestrator.py --mode all --ticker NVDA
```

### Option 3: Demo
```bash
python demo_trading_system.py
```

### Option 4: Direct Python Usage
```python
from comprehensive_trading_system import ComprehensiveTradingSystem

system = ComprehensiveTradingSystem()

# Train models
buy_data = system.train_buy_model("NVDA", "2023-01-01", "2024-01-01")
sell_data = system.train_sell_model("NVDA", "2023-01-01", "2024-01-01")

# Visualize
system.visualize_buy_model(buy_data)
system.visualize_sell_model(sell_data)

# Future predictions
future_data = system.run_future_predictions("NVDA", days_ahead=30)

# Combined analysis
combined_data = system.run_combined_analysis("NVDA", "2024-01-01", "2024-12-31")
```

## 📊 Model Features

### Buy Side Features
- **Technical Indicators**: RSI, Moving Averages, Stochastic Oscillator
- **Volatility Metrics**: Volatility z-score, Ulcer Index
- **Market Conditions**: VIX, Growth rates, Return entropy
- **Dip Detection**: Growth since last bottom, Days since last bottom

### Sell Side Features
- **Peak Detection**: Days since last peak, Decline since last peak
- **Momentum Indicators**: One week growth, RSI
- **Price Ratios**: Price to moving average ratios
- **Volatility**: High-low volatility measures

## 🎯 Output Examples

### Training Results
```
📊 BUY MODEL RESULTS:
  Total data points: 252
  Actual dips: 45
  Predicted dips: 12
  Prediction rate: 4.76%

📊 SELL MODEL RESULTS:
  Total data points: 252
  Actual sell opportunities: 38
  Predicted sell signals: 8
  Signal rate: 3.17%
```

### Future Predictions
```
🔮 FUTURE PREDICTIONS FOR NVDA
Analysis period: 2024-12-01 to 2024-12-31

📈 BUY SIGNALS:
  2024-12-15: Price $485.23, Prob: 0.923
  2024-12-22: Price $472.18, Prob: 0.891

📉 SELL SIGNALS:
  2024-12-08: Price $512.45, Prob: 0.856
  2024-12-18: Price $498.67, Prob: 0.823
```

## 🔧 Customization

### Parameters
- **Ticker Symbol**: Any valid Yahoo Finance ticker
- **Date Ranges**: Custom training and test periods
- **Prediction Thresholds**: Adjustable buy/sell signal thresholds
- **Lookforward Days**: Configurable future prediction horizon

### Model Settings
- **Buy Threshold**: Default 0.9 (90% confidence)
- **Sell Threshold**: Default 0.75 (75% confidence)
- **Feature Engineering**: Automatic computation of all indicators
- **Data Validation**: Automatic handling of missing values

## 📈 Visualization Features

### Interactive Charts
- **Zoom and Pan**: Navigate through time periods
- **Hover Details**: See exact values and probabilities
- **Multiple Overlays**: Price, signals, and probabilities
- **Export Options**: Save charts as images or HTML

### Chart Types
- **Price Charts**: Stock price with buy/sell signals
- **Probability Charts**: Model confidence over time
- **Combined Charts**: Both signals on single chart
- **Feature Importance**: Model decision factors

## 🚨 Requirements

### Python Packages
```
pandas
numpy
plotly
matplotlib
yfinance
scikit-learn
xgboost
imbalanced-learn
ta
scipy
```

### Installation
```bash
pip install pandas numpy plotly matplotlib yfinance scikit-learn xgboost imbalanced-learn ta scipy
```

## 🎉 Benefits

1. **Complete Workflow**: From training to prediction in one system
2. **Flexible Usage**: Multiple interfaces for different needs
3. **Professional Visualizations**: Interactive charts with detailed information
4. **Future-Ready**: Predicts upcoming market opportunities
5. **Combined Intelligence**: Uses both buy and sell signals for comprehensive analysis

## 📝 Notes

- Models are automatically saved after training
- All visualizations are interactive Plotly charts
- Future predictions use the most recent market data
- The system handles missing data and edge cases automatically
- All operations can be run independently or together

This comprehensive system gives you everything needed for sophisticated trading analysis with both individual model insights and combined intelligence.



