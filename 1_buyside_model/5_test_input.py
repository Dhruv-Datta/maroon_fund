import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button
from mpl_interactions import zoom_factory, panhandler
from ta import volume, momentum
from scipy.stats import entropy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class DipSelector:
    def __init__(self, data):
        self.data = data.copy()
        self.data['Date'] = pd.to_datetime(self.data['Date']).dt.tz_localize(None)
        self.dips = []
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.scatter = None
        self.done_button = None
        self.undo_button = None
        self.setup_plot()

    def setup_plot(self):
        self.ax.plot(self.data['Date'], self.data['Close'], label='Close Price')
        self.ax.set_title('Select Dips on Stock Chart (Use scroll to zoom, right-click to pan)')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price')
        self.ax.legend()
        plt.xticks(rotation=45)

        # Create and store buttons to prevent garbage collection
        self.done_button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.done_button = Button(self.done_button_ax, 'Done')
        self.done_button.on_clicked(self.on_done)

        self.undo_button_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.undo_button = Button(self.undo_button_ax, 'Undo')
        self.undo_button.on_clicked(self.on_undo)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        zoom_factory(self.ax)
        panhandler(self.fig)

    def on_click(self, event):
        if event.inaxes == self.ax and event.button == 1:
            x = event.xdata
            if x is not None:
                date = pd.Timestamp(mdates.num2date(x)).tz_localize(None)
                closest = min(self.data['Date'], key=lambda d: abs(d - date))
                price = self.data.loc[self.data['Date'] == closest, 'Close'].iloc[0]
                self.dips.append((closest, price))
                if self.scatter:
                    self.scatter.remove()
                self.scatter = self.ax.scatter(*zip(*self.dips), color='red', zorder=5)
                plt.draw()

    def on_undo(self, event):
        if self.dips:
            self.dips.pop()
            if self.scatter:
                self.scatter.remove()
            if self.dips:
                self.scatter = self.ax.scatter(*zip(*self.dips), color='red', zorder=5)
            else:
                self.scatter = None
            plt.draw()

    def on_done(self, event):
        plt.close(self.fig)

    def get_dips(self):
        return pd.DataFrame(self.dips, columns=['Date', 'Price'])


def calc_entropy(series):
    arr = series[~np.isnan(series)]
    if arr.size == 0:
        return 0.0
    counts, _ = np.histogram(arr, bins=10)
    return entropy(counts + 1)


def slope(x):
    y = np.asarray(x)
    n = y.size
    if n < 2 or np.allclose(y, y[0]):
        return 0.0
    idx = np.arange(n)
    return np.polyfit(idx, y, 1)[0]


def fetch_and_compute(ticker_symbol, start_date, end_date):
    df = yf.Ticker(ticker_symbol).history(start=start_date, end=end_date).reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Base features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = (df['High'] - df['Low']) / df['Open']
    df['MA_25'] = df['Close'].rolling(25, min_periods=1).mean()
    df['MA_100'] = df['Close'].rolling(100, min_periods=1).mean()
    df['RSI_14'] = momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['Month_Growth_Rate'] = df['Close'].pct_change(21)

    # Stochastic
    low50 = df['Low'].rolling(50, min_periods=1).min()
    high50 = df['High'].rolling(50, min_periods=1).max()
    df['STOCH_%K'] = (df['Close'] - low50) / (high50 - low50) * 100
    df['STOCH_%D'] = df['STOCH_%K'].rolling(20, min_periods=1).mean()

    # Volatility z-score
    df['Roll_Vol_20d'] = df['Daily_Return'].rolling(20, min_periods=1).std()
    vol_mean = df['Roll_Vol_20d'].rolling(126, min_periods=1).mean()
    vol_std = df['Roll_Vol_20d'].rolling(126, min_periods=1).std()
    df['Vol_zscore'] = (df['Roll_Vol_20d'] - vol_mean) / vol_std

    # Ulcer Index
    ui = []
    for i in range(len(df)):
        window = df['Close'].iloc[max(0, i-13):i+1]
        peak = window.max()
        dd = (peak - window) / peak
        ui.append(np.sqrt((dd**2).mean()))
    df['Ulcer_Index'] = ui

    # Entropy
    df['Return_Entropy_50d'] = df['Daily_Return'].rolling(50, min_periods=1).apply(calc_entropy, raw=False)

    # VIX merge
    vix = yf.Ticker("^VIX").history(start=start_date, end=end_date).reset_index()
    vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
    vix = vix[['Date', 'Close']].rename(columns={'Close': 'VIX'})
    df = df.merge(vix, on='Date', how='left')


    # Dip-related features
    df['Growth_Since_Last_Bottom'] = 0.0
    df['Days_Since_Last_Bottom'] = 0
    df['Week_Growth_Rate'] = df['Close'].pct_change(5)

    last_bot = 0
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] < df.loc[last_bot, 'Close']:
            last_bot = i
        df.at[i, 'Growth_Since_Last_Bottom'] = (df.at[i, 'Close'] - df.at[last_bot, 'Close']) / df.at[last_bot, 'Close']
        df.at[i, 'Days_Since_Last_Bottom'] = i - last_bot

    df.bfill(inplace=True)
    return df



def prepare_data(df):
    drop_cols = ['Open', 'High', 'Volume', 'Daily_Return', 'Roll_Vol_20d', 'Dividends', 'Stock Splits', 'Capital Gains']
    return df.drop(columns=drop_cols, errors='ignore').reset_index(drop=True)



if __name__ == '__main__':
    # Parameters
    ticker = "NVDA"
    start_date = "2024-01-01"
    end_date = "2025-07-28"

    # Data preparation
    raw_df = fetch_and_compute(ticker, start_date, end_date)
    cleaned_df = prepare_data(raw_df)

    # Dip labeling
    selector = DipSelector(cleaned_df)
    plt.show()
    dips_df = selector.get_dips()
    cleaned_df['Target'] = 0
    for _, row in dips_df.iterrows():
        date = pd.Timestamp(row['Date']).tz_localize(None)
        closest = min(cleaned_df['Date'], key=lambda d: abs(d - date))
        cleaned_df.loc[cleaned_df['Date'] == closest, 'Target'] = 1

    # Save for training/analysis
    cleaned_df.to_csv('test_input.csv', index=False)
    print('Saved to test_input.csv')

# ==================== MODEL TESTING ACROSS PERIODS ====================
print(f"\n" + "="*60)
print("MODEL TESTING ACROSS DIFFERENT TIME PERIODS")
print("="*60)

def fetch_and_compute_features(ticker_symbol, start_date, end_date):
    """Fetch stock data and compute all features used in the model"""
    df = yf.Ticker(ticker_symbol).history(start=start_date, end=end_date).reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Base features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = (df['High'] - df['Low']) / df['Open']
    df['MA_25'] = df['Close'].rolling(window=25, min_periods=1).mean()
    df['MA_100'] = df['Close'].rolling(window=100, min_periods=1).mean()
    df['RSI_14'] = momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # Month growth rate
    def compute_growth_rate(i, closes):
        if i < 12:
            return (closes[i] - closes[0]) / closes[0]
        else:
            return (closes[i] - closes[i - 12]) / closes[i - 12]
    
    df['Month_Growth_Rate'] = df['Close'].rolling(window=len(df), min_periods=1).apply(
        lambda x: compute_growth_rate(len(x)-1, x.values), raw=False
    )

    # Stochastic %K and %D
    low50 = df['Low'].rolling(window=50, min_periods=1).min()
    high50 = df['High'].rolling(window=50, min_periods=1).max()
    df['STOCH_%K'] = (df['Close'] - low50) / (high50 - low50) * 100
    df['STOCH_%D'] = df['STOCH_%K'].rolling(window=20, min_periods=1).mean()

    # Volatility z-score
    df['Roll_Vol_20d'] = df['Daily_Return'].rolling(window=20, min_periods=1).std()
    vol_mean = df['Roll_Vol_20d'].rolling(window=126, min_periods=1).mean()
    vol_std = df['Roll_Vol_20d'].rolling(window=126, min_periods=1).std()
    df['Vol_zscore'] = (df['Roll_Vol_20d'] - vol_mean) / vol_std

    # Ulcer Index
    ui = []
    for i in range(len(df)):
        window = df['Close'].iloc[max(0, i-13) : i+1]
        peak = window.max()
        dd = (peak - window) / peak
        ui.append(np.sqrt((dd**2).mean()))
    df['Ulcer_Index'] = ui

    # Return entropy
    def calc_entropy(series):
        arr = series[~np.isnan(series)]
        if arr.size == 0:
            return 0.0
        counts, _ = np.histogram(arr, bins=10)
        return entropy(counts + 1)
    
    df['Return_Entropy_50d'] = (
        df['Daily_Return']
        .rolling(window=50, min_periods=1)
        .apply(calc_entropy, raw=False)
    )

    # VIX
    vix = yf.Ticker("^VIX").history(start=start_date, end=end_date).reset_index()
    vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None)
    vix = vix[['Date','Close']].rename(columns={'Close':'VIX'})
    df = df.merge(vix, on='Date', how='left')

    # Dip-related features
    df['Growth_Since_Last_Bottom'] = 0.0
    df['Days_Since_Last_Bottom'] = 0
    
    def compute_week_growth_rate(i, closes):
        if i < 5:
            return (closes[i] - closes[0]) / closes[0]
        else:
            return (closes[i] - closes[i - 5]) / closes[i - 5]
    
    df['Week_Growth_Rate'] = df['Close'].rolling(window=len(df), min_periods=1).apply(
        lambda x: compute_week_growth_rate(len(x)-1, x.values), raw=False
    )

    # Compute Growth/Days since last bottom
    last_bot = 0
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] < df.loc[last_bot, 'Close']:
            last_bot = i
        df.loc[i, 'Growth_Since_Last_Bottom'] = (
            df.loc[i, 'Close'] - df.loc[last_bot, 'Close']
        ) / df.loc[last_bot, 'Close']
        df.loc[i, 'Days_Since_Last_Bottom'] = i - last_bot

    # Backfill missing values
    df.bfill(inplace=True)

    # Drop raw columns
    drop_cols = ['Open', 'High', 'Volume', 'Daily_Return', 'Roll_Vol_20d', 'Dividends', 'Stock Splits', 'Capital Gains']
    cleaned = df.drop(columns=drop_cols, errors='ignore')
    
    return cleaned

def test_model_period(ticker, start_date, end_date, period_name, model, threshold=0.9):
    """Test model on a specific time period"""
    print(f"\n{'='*50}")
    print(f"TESTING: {period_name}")
    print(f"Ticker: {ticker} | Period: {start_date} to {end_date}")
    print(f"{'='*50}")
    
    try:
        # Fetch and prepare data
        test_data = fetch_and_compute_features(ticker, start_date, end_date)
        
        if len(test_data) < 50:  # Need minimum data for features
            print(f"❌ Insufficient data: {len(test_data)} days")
            return None
            
        # Prepare features
        test_features = [col for col in test_data.columns if col not in ['Date', 'Target']]
        X_test = test_data[test_features]
        
        # Make predictions
        test_data['Dip_Probability'] = model.predict_proba(X_test)[:, 1]
        test_data['Predicted_Dip'] = test_data['Dip_Probability'] > threshold
        
        # Calculate metrics
        predicted_dips = test_data[test_data['Predicted_Dip']]
        prediction_rate = len(predicted_dips) / len(test_data) * 100
        avg_prob = predicted_dips['Dip_Probability'].mean() if len(predicted_dips) > 0 else 0
        
        # Price performance analysis
        price_start = test_data['Close'].iloc[0]
        price_end = test_data['Close'].iloc[-1]
        total_return = (price_end - price_start) / price_start * 100
        
        # Volatility analysis
        daily_returns = test_data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized
        
        print(f"📊 RESULTS:")
        print(f"  Data points: {len(test_data)}")
        print(f"  Predictions: {len(predicted_dips)} ({prediction_rate:.2f}%)")
        print(f"  Avg probability: {avg_prob:.4f}")
        print(f"  Total return: {total_return:+.2f}%")
        print(f"  Volatility: {volatility:.2f}%")
        
        # Analyze prediction timing
        if len(predicted_dips) > 0:
            print(f"  Prediction dates: {predicted_dips['Date'].min().strftime('%Y-%m-%d')} to {predicted_dips['Date'].max().strftime('%Y-%m-%d')}")
            
            # Check if predictions align with actual dips (local minima)
            predicted_dates = predicted_dips['Date'].tolist()
            actual_dips = []
            
            for i in range(1, len(test_data)-1):
                if (test_data.iloc[i]['Close'] < test_data.iloc[i-1]['Close'] and 
                    test_data.iloc[i]['Close'] < test_data.iloc[i+1]['Close']):
                    actual_dips.append(test_data.iloc[i]['Date'])
            
            # Simple alignment check
            aligned_predictions = 0
            for pred_date in predicted_dates:
                for actual_date in actual_dips:
                    if abs((pred_date - actual_date).days) <= 3:  # Within 3 days
                        aligned_predictions += 1
                        break
            
            alignment_rate = aligned_predictions / len(predicted_dates) * 100 if len(predicted_dates) > 0 else 0
            print(f"  Alignment with actual dips: {alignment_rate:.1f}%")
        
        return {
            'period_name': period_name,
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'data_points': len(test_data),
            'predictions': len(predicted_dips),
            'prediction_rate': prediction_rate,
            'avg_probability': avg_prob,
            'total_return': total_return,
            'volatility': volatility,
            'data': test_data
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

# Check if model exists and run tests
try:
    import joblib
    model = joblib.load('xgboost_model.joblib')
    print("✅ Model loaded successfully - running comprehensive tests...")
    
    # Define test periods
    test_periods = [
        # Historical periods
        ("NVDA", "2020-01-01", "2020-12-31", "2020 (COVID Crash & Recovery)"),
        ("NVDA", "2021-01-01", "2021-12-31", "2021 (AI Boom)"),
        ("NVDA", "2022-01-01", "2022-12-31", "2022 (Tech Crash)"),
        ("NVDA", "2023-01-01", "2023-12-31", "2023 (AI Surge)"),
        
        # Recent periods
        ("NVDA", "2024-01-01", "2024-06-30", "2024 H1"),
        ("NVDA", "2024-07-01", "2024-12-31", "2024 H2"),
        
        # Future prediction (if data available)
        ("NVDA", "2025-01-01", "2025-06-30", "2025 H1 (Future Prediction)"),
        
        # Different tickers for robustness
        ("AAPL", "2023-01-01", "2024-12-31", "AAPL 2023-2024"),
        ("TSLA", "2023-01-01", "2024-12-31", "TSLA 2023-2024"),
        ("MSFT", "2023-01-01", "2024-12-31", "MSFT 2023-2024"),
    ]

    # Run tests
    results = []
    for ticker, start_date, end_date, period_name in test_periods:
        result = test_model_period(ticker, start_date, end_date, period_name, model)
        if result:
            results.append(result)

    # Create summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*60}")

    if results:
        df_summary = pd.DataFrame([
            {
                'Period': r['period_name'],
                'Ticker': r['ticker'],
                'Predictions': r['predictions'],
                'Prediction Rate (%)': f"{r['prediction_rate']:.2f}",
                'Avg Probability': f"{r['avg_probability']:.3f}",
                'Total Return (%)': f"{r['total_return']:+.2f}",
                'Volatility (%)': f"{r['volatility']:.2f}"
            }
            for r in results
        ])
        
        print(df_summary.to_string(index=False))
        
        # Future prediction analysis
        future_results = [r for r in results if 'Future' in r['period_name']]
        if future_results:
            print(f"\n🔮 FUTURE PREDICTION ANALYSIS:")
            print(f"{'='*40}")
            for result in future_results:
                print(f"Period: {result['period_name']}")
                print(f"Predictions: {result['predictions']} ({result['prediction_rate']:.2f}%)")
                if result['predictions'] > 0:
                    print(f"Average confidence: {result['avg_probability']:.3f}")
                    print(f"Market conditions: {result['volatility']:.1f}% volatility")

    print(f"\n✅ Testing complete! Your model has been tested across multiple periods and tickers.")
    
except FileNotFoundError:
    print("⚠️  Model not found. Please run 4_model.py first to train the model.")
    print("   Then re-run this script to see comprehensive testing results.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print("   Please ensure 4_model.py has been run successfully first.")
