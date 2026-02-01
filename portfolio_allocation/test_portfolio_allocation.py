"""
Test Portfolio Allocation Strategy

Backtests the 3% per trade allocation strategy using signals from combined_buy_sell_model
on historical test data (test_input.csv)
"""
import sys
import os, csv
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

# read what the user wrote
with open(os.path.abspath('./x_utilities/utils.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile)
    reader_indexed = []
    for i in reader:
        reader_indexed.append(i)

start_date = reader_indexed[1][1]
end_date   = "2025-11-11"
t = reader_indexed[1][0]

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both portfolio allocation strategies
import importlib.util

# Import sell 100% strategy
spec_100 = importlib.util.spec_from_file_location(
    "portfolio_allocation_sell_100",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_allocation_sell_100%.py")
)
module_100 = importlib.util.module_from_spec(spec_100)
spec_100.loader.exec_module(module_100)
PortfolioAllocatorSell100 = module_100.PortfolioAllocator

# Import sell 3% strategy
spec_3 = importlib.util.spec_from_file_location(
    "portfolio_allocation_sell_3",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_allocation_sell_3%.py")
)
module_3 = importlib.util.module_from_spec(spec_3)
spec_3.loader.exec_module(module_3)
PortfolioAllocatorSell3 = module_3.PortfolioAllocator

# Import proportional sell strategy
spec_prop = importlib.util.spec_from_file_location(
    "portfolio_allocation_sell_proportional",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_allocation_sell_proportional.py")
)
module_prop = importlib.util.module_from_spec(spec_prop)
spec_prop.loader.exec_module(module_prop)
PortfolioAllocatorProportional = module_prop.PortfolioAllocator

# Import optimized strategy
spec_opt = importlib.util.spec_from_file_location(
    "portfolio_allocation_optimized",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_allocation_optimized.py")
)
module_opt = importlib.util.module_from_spec(spec_opt)
spec_opt.loader.exec_module(module_opt)
OptimizedPortfolioAllocator = module_opt.OptimizedPortfolioAllocator

# Try to load ML models
USE_ML_MODELS = False
buy_model = None
sell_model = None

def get_spy_data(start_date, end_date):
    """Fetch SPY data for benchmark comparison"""
    try:
        import yfinance as yf
        print(f"Fetching SPY data from {start_date} to {end_date}...")
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        spy.reset_index(inplace=True)
        print(f"   Fetched {len(spy)} SPY data points")
        return spy
    except Exception as e:
        print(f"Warning: Could not fetch SPY data: {e}")
        return None

try:
    import joblib
    import importlib.util
    
    # Import the loaded models module
    combined_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'combined_buy_sell_model'
    )
    
    # Import models module
    spec = importlib.util.spec_from_file_location(
        "loaded_models", 
        os.path.join(combined_model_path, '2_Loaded_Models.py')
    )
    loaded_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_models)
    
    # Models are loaded at module level
    buy_model = loaded_models.buy_model
    sell_model = loaded_models.sell_model
    
    if buy_model is not None and sell_model is not None:
        USE_ML_MODELS = True
        print("ML models loaded successfully")
except Exception as e:
    USE_ML_MODELS = False
    print(f"Warning: ML models not available: {e}")
    print("Exiting - ML models required for this strategy")
    sys.exit(1)


def get_signals_from_combined_model(ticker=t, start_date=start_date, end_date="2025-11-11"):
    """
    Get buy/sell signals using EXACTLY the same method as combined_buy_sell_model/3_Visualize.py
    
    This ensures we get the exact same signals by:
    1. Using the same data fetching method
    2. Using the same feature computation functions
    3. Using the same models
    4. Using the same thresholds
    """
    print("\nGetting predictions using EXACT same method as combined_buy_sell_model...")
    
    # Import the functions from combined_buy_sell_model exactly like 3_Visualize.py does
    import importlib.util
    combined_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'combined_buy_sell_model'
    )
    
    spec = importlib.util.spec_from_file_location(
        "loaded_models", 
        os.path.join(combined_model_path, '2_Loaded_Models.py')
    )
    loaded_models_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_models_module)
    
    # Get the exact same functions used in 3_Visualize.py
    fetch_stock_data = loaded_models_module.fetch_stock_data
    compute_buy_side_features = loaded_models_module.compute_buy_side_features
    compute_sell_side_features = loaded_models_module.compute_sell_side_features
    
    # Step 1: Fetch data exactly like 3_Visualize.py does
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = fetch_stock_data(ticker, start_date, end_date)
    
    # Step 2: Compute buy side features exactly like 3_Visualize.py
    print("Computing buy side features...")
    df = compute_buy_side_features(df)
    
    # Step 3: Compute sell side features exactly like 3_Visualize.py
    print("Computing sell side features...")
    df = compute_sell_side_features(df)
    
    # Step 4: Prepare buy features exactly like 3_Visualize.py
    buy_features = [
        'Low', 'Close', 'Volatility', 'MA_25', 'MA_100', 'RSI_14', 'Month_Growth_Rate',
        'STOCH_%K', 'STOCH_%D', 'Vol_zscore', 'Ulcer_Index', 'Return_Entropy_50d',
        'VIX', 'Growth_Since_Last_Bottom', 'Days_Since_Last_Bottom', 'Week_Growth_Rate'
    ]
    
    # Ensure all buy features exist (exactly like 3_Visualize.py)
    for feature in buy_features:
        if feature not in df.columns:
            df[feature] = 0
    
    X_buy = df[buy_features].fillna(0)
    
    # Step 5: Get buy predictions (exactly like 3_Visualize.py)
    df['Dip_Probability'] = buy_model.predict_proba(X_buy)[:, 1]
    df['Predicted_Dip'] = df['Dip_Probability'] > 0.9
    
    # Step 6: Prepare sell features exactly like 3_Visualize.py
    sell_features = [
        'Close', 'Volatility', 'MA_25', 'MA_100',
        'Decline_Since_Last_Peak', 'Days_Since_Last_Peak',
        'One_Week_Growth', 'RSI', 'Price_to_MA25', 'Price_to_MA100'
    ]
    
    # Ensure all sell features exist (exactly like 3_Visualize.py)
    for feature in sell_features:
        if feature not in df.columns:
            df[feature] = 0
    
    X_sell = df[sell_features].fillna(0)
    
    # Step 7: Get sell predictions (exactly like 3_Visualize.py)
    df['Sell_Probability'] = sell_model.predict_proba(X_sell)[:, 1]
    df['Sell_Signal'] = df['Sell_Probability'] > 0.9
    
    buy_count = df['Predicted_Dip'].sum()
    sell_count = df['Sell_Signal'].sum()
    
    print(f"\nSignals Generated:")
    print(f"   Buy signals (Predicted_Dip=True): {buy_count}")
    print(f"   Sell signals (Sell_Signal=True): {sell_count}")
    
    return df


def run_backtest(strategy_type='optimized'):
    """
    Run the portfolio allocation backtest
    
    Args:
        strategy_type: 'sell_100', 'sell_3', 'sell_proportional', or 'optimized'
            - 'sell_100': Sell 100% of position on sell signal
            - 'sell_3': Sell 3% of portfolio on sell signal
            - 'sell_proportional': Sell (100% / buy_count) of position on sell signal
            - 'optimized': Best performing strategy (5% buy, 150% base, capped 50% max sell)
    """
    print("="*80)
    print("PORTFOLIO ALLOCATION STRATEGY - BACKTEST")
    print("="*80)
    
    # Get signals using EXACTLY the same method as combined_buy_sell_model/3_Visualize.py
    ticker = t
    
    df = get_signals_from_combined_model(ticker, start_date, end_date)
    
    # Fetch SPY data for benchmark comparison
    spy_data = get_spy_data(start_date, end_date)   

    print(f"\nData loaded: {len(df)} rows")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    
    # Choose strategy
    if strategy_type == 'sell_100':
        print("\nStrategy: Buy 3% per signal, Sell 100% of position on sell signal")
        allocator = PortfolioAllocatorSell100(
            initial_capital=100000.0,
            position_size_pct=0.03,
            transaction_cost_pct=0.001,
            commission=1.0
        )
        strategy_name = "Sell 100%"
    elif strategy_type == 'sell_3':
        print("\nStrategy: Buy 3% per signal, Sell 3% of portfolio on sell signal")
        allocator = PortfolioAllocatorSell3(
            initial_capital=100000.0,
            position_size_pct=0.03,
            transaction_cost_pct=0.001,
            commission=1.0
        )
        strategy_name = "Sell 3%"
    elif strategy_type == 'sell_proportional':
        print("\nStrategy: Buy 3% per signal, Sell (100% / buy_count) of position on sell signal")
        allocator = PortfolioAllocatorProportional(
            initial_capital=100000.0,
            position_size_pct=0.03,
            transaction_cost_pct=0.001,
            commission=1.0
        )
        strategy_name = "Sell Proportional"
    elif strategy_type == 'optimized':
        print("\nStrategy: OPTIMIZED - Best Performing Strategy")
        allocator = OptimizedPortfolioAllocator(
            initial_capital=100000.0,
            transaction_cost_pct=0.001,
            commission=1.0
        )
        strategy_name = "Optimized"
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")
    
    allocator.strategy_name = strategy_name
    
    # Process signals
    print("\n" + "="*80)
    print("PROCESSING SIGNALS AND EXECUTING TRADES...")
    print("="*80)
    
    allocator.process_signals(df)
    
    # Get final portfolio value
    final_price = float(df.iloc[-1]['Close'])
    summary = allocator.get_summary(final_price)
    
    # Print results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    
    print(f"\nCapital Summary:")
    print(f"  Initial Capital: ${summary['initial_capital']:,.2f}")
    print(f"  Final Capital:   ${summary['final_capital']:,.2f}")
    print(f"  Total Return:    {summary['total_return']:.2%}")
    
    print(f"\nPortfolio Breakdown:")
    print(f"  Cash:            ${summary['cash']:,.2f} ({summary['cash_pct']:.1f}%)")
    print(f"  Positions:       ${summary['position_value']:,.2f} ({summary['position_pct']:.1f}%)")
    
    if allocator.state.positions:
        print(f"\nPositions:")
        for ticker, position in allocator.state.positions.items():
            current_value = position.shares * final_price
            pnl = current_value - (position.shares * position.avg_cost)
            pnl_pct = (pnl / (position.shares * position.avg_cost)) * 100 if position.shares > 0 else 0
            print(f"  {ticker}: {position.shares:.4f} shares @ ${position.avg_cost:.2f} avg")
            print(f"    Current Value: ${current_value:,.2f}")
            print(f"    P&L: ${pnl:,.2f} ({pnl_pct:.2f}%)")
    
    print(f"\nTrade Statistics:")
    print(f"  Total Trades:    {summary['num_trades']}")
    print(f"  Buy Trades:      {summary['buy_trades']}")
    print(f"  Sell Trades:     {summary['sell_trades']}")
    
    # Build equity curve
    equity_curve = []
    dates = []
    cash_history = []
    position_history = []
    cash = allocator.initial_capital
    shares = 0
    
    for _, row in df.iterrows():
        date = row['Date']
        price = float(row['Close'])
        
        # Process trades that occurred on this date
        date_str = str(date)
        day_trades = [t for t in allocator.state.trades if t.date == date_str]
        
        for trade in day_trades:
            if trade.action == 'BUY':
                cost = trade.shares * trade.price * 1.001 + 1.0
                cash -= cost
                shares += trade.shares
            else:  # SELL
                proceeds = trade.shares * trade.price * 0.999 - 1.0
                cash += proceeds
                shares -= trade.shares
        
        # Calculate portfolio value
        portfolio_value = cash + shares * price
        position_value = shares * price
        
        equity_curve.append(portfolio_value)
        cash_history.append(cash)
        position_history.append(position_value)
        dates.append(date)
    
    # Create visualization
    create_visualization(df, allocator, equity_curve, dates, summary, cash_history, position_history, strategy_name, spy_data)
    
    # Save results
    save_results(df, allocator, equity_curve, dates, summary, cash_history, position_history, strategy_name)
    
    return allocator, summary


def create_visualization(df, allocator, equity_curve, dates, summary, cash_history, position_history, strategy_name, spy_data):
    """Create visualization of backtest results"""
    print("\nCreating visualization...")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Signals', 'Portfolio Value', 'Cash & Positions'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Plot 1: Price with buy/sell signals
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add buy signals
    buy_df = df[df['Predicted_Dip']]
    if len(buy_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_df['Date'],
                y=buy_df['Close'],
                mode='markers+text',
                name='Predicted Dip',
                marker=dict(color='red', size=8),
                text=[f"Prob {p:.2f}" for p in buy_df['Dip_Probability']],
                textposition='top center',
                hovertemplate='Date: %{x}<br>Price: %{y:.2f}<br>Dip Probability: %{customdata:.3f}<extra></extra>',
                customdata=buy_df['Dip_Probability']
            ),
            row=1, col=1
        )
    
    # Add sell signals
    sell_df = df[df['Sell_Signal']]
    if len(sell_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_df['Date'],
                y=sell_df['Close'],
                mode='markers+text',
                name='Predicted Sell',
                marker=dict(color='green', size=8),
                text=[f"Prob {p:.2f}" for p in sell_df['Sell_Probability']],
                textposition='top center',
                hovertemplate='Date: %{x}<br>Price: %{y:.2f}<br>Sell Probability: %{customdata:.3f}<extra></extra>',
                customdata=sell_df['Sell_Probability']
            ),
            row=1, col=1
        )
    
    # Plot 2: Equity curve
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity_curve,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)'
        ),
        row=2, col=1
    )
    
    # Add SPY buy-and-hold comparison
    if spy_data is not None and len(spy_data) > 0:
        spy_start_price = float(spy_data.iloc[0]['Close'])
        spy_shares = allocator.initial_capital / spy_start_price
        
        spy_capital_line = []
        for date in dates:
            spy_row = spy_data[spy_data['Date'] <= date]
            if len(spy_row) > 0:
                spy_price = float(spy_row.iloc[-1]['Close'])
                spy_capital_line.append(spy_shares * spy_price)
            else:
                spy_capital_line.append(allocator.initial_capital)
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=spy_capital_line,
                mode='lines',
                name='SPY Buy & Hold',
                line=dict(color='purple', width=2, dash='dot'),
                hovertemplate='Date: %{x}<br>SPY Value: $%{y:,.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        spy_final_value = spy_capital_line[-1]
        spy_return = (spy_final_value - allocator.initial_capital) / allocator.initial_capital
        print(f"\nSPY Buy & Hold Benchmark:")
        print(f"  Initial Value: ${allocator.initial_capital:,.2f}")
        print(f"  Final Value:   ${spy_final_value:,.2f}")
        print(f"  Total Return:  {spy_return:.2%}")
        print(f"\nStrategy vs SPY:")
        print(f"  Outperformance: {(summary['total_return'] - spy_return):.2%}")

    # Add initial capital line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[allocator.initial_capital] * len(dates),
            mode='lines',
            name='Initial Capital',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Plot 3: Cash and position values
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cash_history,
            mode='lines',
            name='Cash',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,0,255,0.1)',
            hovertemplate='Date: %{x}<br>Cash: $%{y:,.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=position_history,
            mode='lines',
            name='Position Value',
            line=dict(color='orange', width=2),
            fill='tonexty',
            fillcolor='rgba(255,165,0,0.1)',
            hovertemplate='Date: %{x}<br>Position Value: $%{y:,.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'Portfolio Allocation Strategy ({strategy_name}) - Backtest Results',
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=3, col=1)
    
    # Save and open
    filename = f"portfolio_allocation_backtest_{strategy_name.replace(' ', '_').replace('%', '')}.html"
    fig.write_html(filename)
    print(f"Visualization saved to: {filename}")
    print("Opening in browser...\n")
    webbrowser.open('file://' + os.path.realpath(filename))


def save_results(df, allocator, equity_curve, dates, summary, cash_history, position_history, strategy_name):
    """Save backtest results to files"""
    print("Saving results...")
    
    suffix = strategy_name.replace(' ', '_').replace('%', '')
    
    # Save equity curve
    equity_df = pd.DataFrame({
        'Date': dates,
        'Portfolio_Value': equity_curve,
        'Cash': cash_history,
        'Position_Value': position_history
    })
    filename = f'portfolio_equity_curve_{suffix}.csv'
    equity_df.to_csv(filename, index=False)
    print(f"   Equity curve saved to: {filename}")
    
    # Save trades
    if allocator.state.trades:
        trades_df = pd.DataFrame([{
            'Date': t.date,
            'Ticker': t.ticker,
            'Action': t.action,
            'Price': t.price,
            'Shares': t.shares,
            'Value': t.value,
            'Signal_Probability': t.signal_probability
        } for t in allocator.state.trades])
        
        filename = f'portfolio_trades_{suffix}.csv'
        trades_df.to_csv(filename, index=False)
        print(f"   Trades saved to: {filename}")
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df['Strategy'] = strategy_name
    filename = f'portfolio_summary_{suffix}.csv'
    summary_df.to_csv(filename, index=False)
    print(f"   Summary saved to: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test portfolio allocation strategy')
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['sell_100', 'sell_3', 'sell_proportional', 'optimized'],
        default='sell_100',
        help='Strategy to use'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PORTFOLIO ALLOCATION BACKTEST")
    print("="*80)
    print(f"Selected Strategy: {args.strategy}")
    print("="*80)
    
    allocator, summary = run_backtest(strategy_type=args.strategy)
    print("\nBacktest complete!")