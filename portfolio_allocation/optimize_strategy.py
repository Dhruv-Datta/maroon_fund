"""
Optimize Portfolio Allocation Strategy

Tests various parameter combinations to find the best performing strategy
using the proportional selling technique.
"""
import sys
import os
import pandas as pd
import numpy as np
from itertools import product
import importlib.util

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the optimizable allocator
optimizable_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_allocation_optimizable.py")
spec = importlib.util.spec_from_file_location("optimizable", optimizable_path)
optimizable_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimizable_module)
OptimizablePortfolioAllocator = optimizable_module.OptimizablePortfolioAllocator

# Import signal generation function from test_portfolio_allocation
test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_portfolio_allocation.py")
spec_test = importlib.util.spec_from_file_location("test_portfolio", test_path)
test_module = importlib.util.module_from_spec(spec_test)
spec_test.loader.exec_module(test_module)
get_signals_from_combined_model = test_module.get_signals_from_combined_model


def run_backtest_with_params(
    df: pd.DataFrame,
    buy_size_pct: float,
    sell_base_pct: float,
    sell_formula: str = 'div',
    max_sell_pct: float = 1.0,
    min_sell_pct: float = 0.0,
    power_factor: float = 1.0,
    max_buys: int = 10
):
    """
    Run backtest with specific parameters
    
    Args:
        df: DataFrame with signals (already computed)
        buy_size_pct: Buy size percentage
        sell_base_pct: Sell base percentage
        sell_formula: Sell formula type
        max_sell_pct: Maximum sell percentage
        min_sell_pct: Minimum sell percentage
        power_factor: Power factor for pow formula
        max_buys: Maximum buys for mult formula
    
    Returns:
        Dictionary with summary statistics
    """
    
    # Initialize allocator with parameters
    allocator = OptimizablePortfolioAllocator(
        initial_capital=100000.0,
        buy_size_pct=buy_size_pct,
        sell_base_pct=sell_base_pct,
        sell_formula=sell_formula,
        max_sell_pct=max_sell_pct,
        min_sell_pct=min_sell_pct,
        power_factor=power_factor,
        max_buys=max_buys,
        transaction_cost_pct=0.001,
        commission=1.0
    )
    
    # Process signals
    allocator.process_signals(df)
    
    # Get final price
    final_price = float(df.iloc[-1]['Close'])
    summary = allocator.get_summary(final_price)
    
    # Calculate additional metrics
    summary['sharpe_ratio'] = calculate_sharpe_ratio(df, allocator)
    summary['max_drawdown'] = calculate_max_drawdown(df, allocator)
    
    return summary


def calculate_sharpe_ratio(df, allocator):
    """Calculate Sharpe ratio (simplified)"""
    # Reconstruct equity curve
    equity_curve = []
    cash = allocator.initial_capital
    shares = 0
    
    for _, row in df.iterrows():
        date = str(row['Date'])
        price = float(row['Close'])
        
        # Process trades that occurred on this date
        day_trades = [t for t in allocator.state.trades if t.date == date]
        
        for trade in day_trades:
            if trade.action == 'BUY':
                cost = trade.shares * trade.price * 1.001 + 1.0
                cash -= cost
                shares += trade.shares
            else:  # SELL
                proceeds = trade.shares * trade.price * 0.999 - 1.0
                cash += proceeds
                shares -= trade.shares
        
        portfolio_value = cash + shares * price
        equity_curve.append(portfolio_value)
    
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate returns
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    if np.std(returns) == 0:
        return 0.0
    
    # Annualized Sharpe (assuming 252 trading days)
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    return sharpe


def calculate_max_drawdown(df, allocator):
    """Calculate maximum drawdown"""
    # Reconstruct equity curve
    equity_curve = []
    cash = allocator.initial_capital
    shares = 0
    
    for _, row in df.iterrows():
        date = str(row['Date'])
        price = float(row['Close'])
        
        # Process trades that occurred on this date
        day_trades = [t for t in allocator.state.trades if t.date == date]
        
        for trade in day_trades:
            if trade.action == 'BUY':
                cost = trade.shares * trade.price * 1.001 + 1.0
                cash -= cost
                shares += trade.shares
            else:  # SELL
                proceeds = trade.shares * trade.price * 0.999 - 1.0
                cash += proceeds
                shares -= trade.shares
        
        portfolio_value = cash + shares * price
        equity_curve.append(portfolio_value)
    
    if len(equity_curve) < 2:
        return 0.0
    
    equity_curve = np.array(equity_curve)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    return abs(np.min(drawdown))


def optimize_strategy():
    """
    Test various parameter combinations to find optimal strategy
    """
    print("="*80)
    print("OPTIMIZING PORTFOLIO ALLOCATION STRATEGY")
    print("="*80)
    
    # Fetch data once (this is expensive, so we do it once)
    print("\nFetching data and computing signals (this may take a moment)...")
    ticker = "NVDA"
    start_date = "2024-01-02"
    end_date = "2025-07-25"
    df = get_signals_from_combined_model(ticker, start_date, end_date)
    print(f"Data loaded: {len(df)} rows\n")
    
    # Define parameter ranges to test
    buy_size_pcts = [0.02, 0.03, 0.04, 0.05]  # 2%, 3%, 4%, 5%
    sell_base_pcts = [50.0, 75.0, 100.0, 125.0, 150.0]  # Different base percentages
    sell_formulas = ['div', 'cap']  # Test division and capped versions
    max_sell_pcts = [0.5, 0.75, 1.0]  # Max 50%, 75%, 100%
    
    results = []
    total_combinations = len(buy_size_pcts) * len(sell_base_pcts) * len(sell_formulas) * len(max_sell_pcts)
    current = 0
    
    print(f"Testing {total_combinations} parameter combinations...")
    print("This may take a few minutes...\n")
    
    # Test all combinations
    for buy_size_pct, sell_base_pct, sell_formula, max_sell_pct in product(
        buy_size_pcts, sell_base_pcts, sell_formulas, max_sell_pcts
    ):
        current += 1
        
        try:
            summary = run_backtest_with_params(
                df=df,  # Pass pre-computed dataframe
                buy_size_pct=buy_size_pct,
                sell_base_pct=sell_base_pct,
                sell_formula=sell_formula,
                max_sell_pct=max_sell_pct,
                min_sell_pct=0.0,
                power_factor=1.0,
                max_buys=10
            )
            
            results.append({
                'buy_size_pct': buy_size_pct,
                'sell_base_pct': sell_base_pct,
                'sell_formula': sell_formula,
                'max_sell_pct': max_sell_pct,
                'total_return': summary['total_return'],
                'final_capital': summary['final_capital'],
                'sharpe_ratio': summary['sharpe_ratio'],
                'max_drawdown': summary['max_drawdown'],
                'num_trades': summary['num_trades'],
                'buy_trades': summary['buy_trades'],
                'sell_trades': summary['sell_trades'],
                'position_pct': summary['position_pct']
            })
            
            if current % 10 == 0:
                print(f"  Progress: {current}/{total_combinations} ({current/total_combinations*100:.1f}%)")
        
        except Exception as e:
            print(f"  Error with params (buy={buy_size_pct}, base={sell_base_pct}, formula={sell_formula}, max={max_sell_pct}): {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\nNo results to analyze!")
        return None
    
    # Sort by total return
    results_df = results_df.sort_values('total_return', ascending=False)
    
    # Display top 10
    print("\n" + "="*80)
    print("TOP 10 STRATEGIES BY TOTAL RETURN")
    print("="*80)
    print(results_df.head(10).to_string(index=False))
    
    # Save all results
    results_df.to_csv('optimization_results.csv', index=False)
    print(f"\nAll results saved to: optimization_results.csv")
    
    # Get best strategy
    best = results_df.iloc[0]
    
    print("\n" + "="*80)
    print("BEST STRATEGY")
    print("="*80)
    print(f"Buy Size: {best['buy_size_pct']*100:.1f}% of capital")
    print(f"Sell Base: {best['sell_base_pct']:.1f}%")
    print(f"Sell Formula: {best['sell_formula']}")
    print(f"Max Sell: {best['max_sell_pct']*100:.1f}%")
    print(f"\nPerformance:")
    print(f"  Total Return: {best['total_return']*100:.2f}%")
    print(f"  Final Capital: ${best['final_capital']:,.2f}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {best['max_drawdown']*100:.2f}%")
    print(f"  Total Trades: {best['num_trades']:.0f}")
    print(f"  Final Position: {best['position_pct']:.1f}%")
    
    return best


if __name__ == "__main__":
    best_strategy = optimize_strategy()
    
    if best_strategy is not None:
        print("\nOptimization complete!")
        print("\nTo use the best strategy, update your parameters:")
        print(f"  buy_size_pct={best_strategy['buy_size_pct']}")
        print(f"  sell_base_pct={best_strategy['sell_base_pct']}")
        print(f"  sell_formula='{best_strategy['sell_formula']}'")
        print(f"  max_sell_pct={best_strategy['max_sell_pct']}")

