"""
Portfolio Allocation Strategy - Sell 3%

Uses buy/sell signals from combined_buy_sell_model to allocate portfolio.
Strategy: 3% of capital per buy, 3% of portfolio per sell signal.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
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


class PortfolioAllocator:
    """
    Portfolio allocator that uses signals from combined_buy_sell_model
    
    Strategy:
    - 3% of total capital per buy trade
    - 3% of total portfolio per sell trade
    - Execute on every buy/sell signal from the model
    - If insufficient cash, use available cash
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.03,  # 3% per trade
        transaction_cost_pct: float = 0.001,  # 0.1% transaction cost
        commission: float = 1.0  # $1 per trade
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.transaction_cost_pct = transaction_cost_pct
        self.commission = commission
        
        # Initialize portfolio state
        self.state = PortfolioState(
            cash=initial_capital,
            positions={},
            total_value=initial_capital,
            trades=[]
        )
    
    def get_total_capital(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        total = self.state.cash
        
        # Add value of all positions at current price
        for position in self.state.positions.values():
            total += position.shares * current_price
        
        return total
    
    def execute_buy_signal(self, date: str, ticker: str, price: float, signal_prob: float):
        """
        Execute a buy signal
        
        Args:
            date: Trading date
            ticker: Stock ticker
            price: Current price
            signal_prob: Signal probability (for record keeping)
        """
        # Calculate target position size (3% of total capital)
        total_capital = self.get_total_capital(price)
        target_trade_value = total_capital * self.position_size_pct
        
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
        Execute a sell signal - SELL 3% of portfolio value
        
        Args:
            date: Trading date
            ticker: Stock ticker
            price: Current price
            signal_prob: Signal probability (for record keeping)
        """
        # Check if we have a position
        if ticker not in self.state.positions:
            return  # No position to sell
        
        position = self.state.positions[ticker]
        
        if position.shares <= 0:
            return  # No shares to sell
        
        # Calculate target position size to sell (3% of total capital)
        total_capital = self.get_total_capital(price)
        target_sell_value = total_capital * self.position_size_pct
        
        # Calculate shares to sell
        shares_to_sell = min(target_sell_value / price, position.shares)
        
        # Check minimum trade size
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
            
            # Remove position if fully sold
            if position.shares < 1e-6:  # Tiny float precision
                del self.state.positions[ticker]
            
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
            'sell_trades': len([t for t in self.state.trades if t.action == 'SELL'])
        }

