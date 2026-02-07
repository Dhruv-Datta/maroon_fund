"""
Portfolio Allocation Strategy - Optimized Best Performer

This is the optimized strategy found through parameter testing.
Best parameters:
- Buy Size: 5.0% of capital per buy
- Sell Base: 150.0%
- Sell Formula: cap (capped division)
- Max Sell: 50.0% of position

Performance (on test data):
- Total Return: 66.63%
- Sharpe Ratio: 1.186
- Max Drawdown: 28.38%
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import the optimizable allocator base class
import sys
import os
import importlib.util

optimizable_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_allocation_optimizable.py")
spec = importlib.util.spec_from_file_location("optimizable", optimizable_path)
optimizable_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimizable_module)
OptimizablePortfolioAllocator = optimizable_module.OptimizablePortfolioAllocator


class OptimizedPortfolioAllocator(OptimizablePortfolioAllocator):
    """
    Optimized portfolio allocator with best-performing parameters
    
    This strategy uses the parameters that achieved the best performance
    during optimization testing:
    - 5% buy size per signal
    - 150% sell base with capped division formula
    - Maximum 50% sell per signal
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_pct: float = 0.001,  # 0.1% transaction cost
        commission: float = 1.0  # $1 per trade
    ):
        # Use optimized parameters
        super().__init__(
            initial_capital=initial_capital,
            buy_size_pct=0.05,  # 5% per buy (optimized)
            sell_base_pct=150.0,  # 150% base (optimized)
            sell_formula='cap',  # Capped division (optimized)
            max_sell_pct=0.5,  # Max 50% sell (optimized)
            min_sell_pct=0.0,
            power_factor=1.0,
            max_buys=10,
            transaction_cost_pct=transaction_cost_pct,
            commission=commission
        )

