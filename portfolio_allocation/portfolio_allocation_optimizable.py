"""
Portfolio Allocation Strategy - Optimizable Proportional Sell

Uses buy/sell signals from combined_buy_sell_model to allocate portfolio.
Strategy: 
- Configurable % of capital per buy signal
- Configurable sell percentage calculation based on buy count
- Allows optimization of parameters for best performance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class Position:
    """Represents a position in a stock"""
    ticker: str
    shares: float
    avg_cost: float
    
    @property
    def value(self) -> float:
        return self.shares * self.avg_cost


@dataclass
class Trade:
    """Represents a single trade"""
    date: str
    ticker: str
    action: str  # 'BUY' or 'SELL'
    price: float
    shares: float
    value: float
    signal_probability: float


@dataclass
class PortfolioState:
    """Current state of the portfolio"""
    cash: float
    positions: Dict[str, Position]  # ticker -> Position
    total_value: float
    trades: List[Trade]


class OptimizablePortfolioAllocator:
    """
    Portfolio allocator with optimizable proportional selling
    
    Strategy:
    - Configurable % of capital per buy trade
    - Configurable sell percentage calculation based on buy count
    - Tracks buy count since last sell
    - On sell: calculates sell percentage using customizable formula
    
    Parameters:
    - buy_size_pct: Percentage of capital to use per buy (default: 0.03 = 3%)
    - sell_base_pct: Base percentage for sell calculation (default: 100.0)
    - sell_formula: Formula type for calculating sell percentage
        - 'div': sell_base_pct / buy_count (default)
        - 'mult': sell_base_pct * buy_count / max_buys
        - 'pow': sell_base_pct / (buy_count ** power_factor)
        - 'cap': min(sell_base_pct / buy_count, max_sell_pct)
    - max_sell_pct: Maximum sell percentage (default: 1.0 = 100%)
    - min_sell_pct: Minimum sell percentage (default: 0.0 = 0%)
    - power_factor: Power factor for pow formula (default: 1.0)
    - max_buys: Maximum buy count for mult formula (default: 10)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        buy_size_pct: float = 0.03,  # 3% per buy
        sell_base_pct: float = 100.0,  # Base percentage for sell calculation
        sell_formula: str = 'div',  # 'div', 'mult', 'pow', 'cap'
        max_sell_pct: float = 1.0,  # Maximum sell percentage (100%)
        min_sell_pct: float = 0.0,  # Minimum sell percentage (0%)
        power_factor: float = 1.0,  # Power factor for pow formula
        max_buys: int = 10,  # Max buys for mult formula
        transaction_cost_pct: float = 0.001,  # 0.1% transaction cost
        commission: float = 1.0  # $1 per trade
    ):
        self.initial_capital = initial_capital
        self.buy_size_pct = buy_size_pct
        self.sell_base_pct = sell_base_pct
        self.sell_formula = sell_formula
        self.max_sell_pct = max_sell_pct
        self.min_sell_pct = min_sell_pct
        self.power_factor = power_factor
        self.max_buys = max_buys
        self.transaction_cost_pct = transaction_cost_pct
        self.commission = commission
        
        # Initialize portfolio state
        self.state = PortfolioState(
            cash=initial_capital,
            positions={},
            total_value=initial_capital,
            trades=[]
        )
        
        # Track buy count since last sell
        self.buy_count_since_last_sell = 0
    
    def get_total_capital(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        total = self.state.cash
        
        # Add value of all positions at current price
        for position in self.state.positions.values():
            total += position.shares * current_price
        
        return total
    
    def calculate_sell_percentage(self) -> float:
        """
        Calculate sell percentage based on buy count and formula
        
        Returns:
            Sell percentage (0.0 to 1.0)
        """
        if self.buy_count_since_last_sell == 0:
            return 1.0  # Sell 100% if no buys recorded (edge case)
        
        if self.sell_formula == 'div':
            # Divide: base / buy_count
            sell_pct = (self.sell_base_pct / 100.0) / self.buy_count_since_last_sell
        elif self.sell_formula == 'mult':
            # Multiply: base * buy_count / max_buys
            sell_pct = (self.sell_base_pct / 100.0) * (self.buy_count_since_last_sell / self.max_buys)
        elif self.sell_formula == 'pow':
            # Power: base / (buy_count ** power_factor)
            sell_pct = (self.sell_base_pct / 100.0) / (self.buy_count_since_last_sell ** self.power_factor)
        elif self.sell_formula == 'cap':
            # Capped: min(base / buy_count, max_sell_pct)
            sell_pct = min((self.sell_base_pct / 100.0) / self.buy_count_since_last_sell, self.max_sell_pct)
        else:
            # Default to div
            sell_pct = (self.sell_base_pct / 100.0) / self.buy_count_since_last_sell
        
        # Apply min/max constraints
        sell_pct = max(self.min_sell_pct, min(sell_pct, self.max_sell_pct))
        
        return sell_pct
    
    def execute_buy_signal(self, date: str, ticker: str, price: float, signal_prob: float):
        """
        Execute a buy signal
        
        Args:
            date: Trading date
            ticker: Stock ticker
            price: Current price
            signal_prob: Signal probability (for record keeping)
        """
        # Increment buy counter
        self.buy_count_since_last_sell += 1
        
        # Calculate target position size (buy_size_pct of total capital)
        total_capital = self.get_total_capital(price)
        target_trade_value = total_capital * self.buy_size_pct
        
        # Check if we have enough cash
        # Account for transaction costs: cost = value * (1 + fee) + commission
        cost_multiplier = 1 + self.transaction_cost_pct
        total_cost_needed = target_trade_value * cost_multiplier + self.commission
        
        # Use available cash if target exceeds cash
        if total_cost_needed > self.state.cash:
            # Calculate max we can afford
            if self.state.cash > self.commission:
                available_for_stock = (self.state.cash - self.commission) / cost_multiplier
                target_trade_value = available_for_stock
        
        # Check if we can afford at least $1 trade
        if target_trade_value < 1.0:
            return  # Not enough cash
        
        # Calculate shares to buy
        shares = target_trade_value / (price * cost_multiplier)
        
        # Calculate actual cost
        actual_cost = shares * price * cost_multiplier + self.commission
        
        # Execute trade
        if actual_cost <= self.state.cash and shares > 0:
            # Update cash
            self.state.cash -= actual_cost
            
            # Update or create position
            if ticker in self.state.positions:
                # Add to existing position (calculate new average cost)
                old_position = self.state.positions[ticker]
                total_shares = old_position.shares + shares
                total_cost = (old_position.shares * old_position.avg_cost) + (shares * price)
                new_avg_cost = total_cost / total_shares
                self.state.positions[ticker] = Position(
                    ticker=ticker,
                    shares=total_shares,
                    avg_cost=new_avg_cost
                )
            else:
                # New position
                self.state.positions[ticker] = Position(
                    ticker=ticker,
                    shares=shares,
                    avg_cost=price
                )
            
            # Record trade
            self.state.trades.append(Trade(
                date=date,
                ticker=ticker,
                action='BUY',
                price=price,
                shares=shares,
                value=shares * price,
                signal_probability=signal_prob
            ))
    
    def execute_sell_signal(self, date: str, ticker: str, price: float, signal_prob: float):
        """
        Execute a sell signal - Proportional sell based on buy count
        
        Args:
            date: Trading date
            ticker: Stock ticker
            price: Current price
            signal_prob: Signal probability (for record keeping)
        """
        # Check if we have a position
        if ticker not in self.state.positions:
            # No position to sell - reset buy counter to 0
            self.buy_count_since_last_sell = 0
            return
        
        position = self.state.positions[ticker]
        
        if position.shares <= 0:
            # No shares to sell - reset buy counter to 0
            self.buy_count_since_last_sell = 0
            return
        
        # Calculate sell percentage using configurable formula
        sell_percentage = self.calculate_sell_percentage()
        
        # Calculate shares to sell (percentage of position)
        shares_to_sell = position.shares * sell_percentage
        
        # Edge case: If calculated shares exceeds available, sell all remaining
        if shares_to_sell > position.shares:
            shares_to_sell = position.shares
        
        # Check minimum trade size ($1)
        if shares_to_sell * price < 1.0:
            return  # Trade too small
        
        # Calculate proceeds (after transaction costs)
        gross_proceeds = shares_to_sell * price
        net_proceeds = gross_proceeds * (1 - self.transaction_cost_pct) - self.commission
        
        # Execute trade
        if net_proceeds > 0 and shares_to_sell > 0:
            # Update cash
            self.state.cash += net_proceeds
            
            # Update position
            position.shares -= shares_to_sell
            
            # Remove position if fully sold (or tiny remainder)
            if position.shares < 1e-6:  # Tiny float precision
                del self.state.positions[ticker]
                # Reset buy counter when position is fully sold
                self.buy_count_since_last_sell = 0
            # Note: For multiple sells in a row, we do NOT reset the counter
            # The counter continues until position is fully sold or we get a new buy
            
            # Record trade
            self.state.trades.append(Trade(
                date=date,
                ticker=ticker,
                action='SELL',
                price=price,
                shares=shares_to_sell,
                value=gross_proceeds,
                signal_probability=signal_prob
            ))
    
    def process_signals(self, data: pd.DataFrame):
        """
        Process buy/sell signals from combined_buy_sell_model output
        
        Args:
            data: DataFrame with columns: Date, Close, Dip_Probability, Sell_Probability, Predicted_Dip, Sell_Signal
        """
        for _, row in data.iterrows():
            date = str(row['Date'])
            price = float(row['Close'])
            ticker = 'NVDA'  # Single stock strategy
            
            # Check for buy signal
            if row.get('Predicted_Dip', False) or row.get('Dip_Probability', 0) >= 0.9:
                signal_prob = float(row.get('Dip_Probability', 0))
                self.execute_buy_signal(date, ticker, price, signal_prob)
            
            # Check for sell signal
            if row.get('Sell_Signal', False) or row.get('Sell_Probability', 0) >= 0.9:
                signal_prob = float(row.get('Sell_Probability', 0))
                self.execute_sell_signal(date, ticker, price, signal_prob)
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        value = self.state.cash
        
        for position in self.state.positions.values():
            value += position.shares * current_price
        
        return value
    
    def get_summary(self, current_price: float) -> Dict:
        """Get portfolio summary"""
        portfolio_value = self.get_portfolio_value(current_price)
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Position details
        position_value = sum(pos.shares * current_price for pos in self.state.positions.values())
        position_pct = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
        cash_pct = (self.state.cash / portfolio_value * 100) if portfolio_value > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': portfolio_value,
            'total_return': total_return,
            'cash': self.state.cash,
            'cash_pct': cash_pct,
            'position_value': position_value,
            'position_pct': position_pct,
            'num_positions': len(self.state.positions),
            'num_trades': len(self.state.trades),
            'buy_trades': len([t for t in self.state.trades if t.action == 'BUY']),
            'sell_trades': len([t for t in self.state.trades if t.action == 'SELL']),
            'buy_size_pct': self.buy_size_pct,
            'sell_base_pct': self.sell_base_pct,
            'sell_formula': self.sell_formula
        }

