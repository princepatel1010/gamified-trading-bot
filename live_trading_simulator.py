import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
from forex_hmm_strategy import ForexHMMStrategy
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    def __init__(self, max_daily_loss=500, max_drawdown=0.1, max_positions=3, 
                 position_size_pct=0.02, max_risk_per_trade=0.01):
        """
        Risk management system
        
        Parameters:
        max_daily_loss: Maximum daily loss in currency
        max_drawdown: Maximum drawdown percentage
        max_positions: Maximum number of concurrent positions
        position_size_pct: Position size as percentage of capital
        max_risk_per_trade: Maximum risk per trade as percentage of capital
        """
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.max_risk_per_trade = max_risk_per_trade
        
        self.daily_pnl = 0
        self.current_drawdown = 0
        self.active_positions = 0
        self.peak_capital = 0
        
    def can_open_position(self, capital, stop_loss_distance):
        """
        Check if a new position can be opened based on risk rules
        """
        # Update peak capital
        if capital > self.peak_capital:
            self.peak_capital = capital
        
        # Calculate current drawdown
        if self.peak_capital > 0:
            self.current_drawdown = (self.peak_capital - capital) / self.peak_capital
        else:
            self.current_drawdown = 0
        
        # Risk checks
        if self.daily_pnl <= -self.max_daily_loss:
            return False, "Daily loss limit exceeded"
        
        if self.current_drawdown >= self.max_drawdown:
            return False, "Maximum drawdown exceeded"
        
        if self.active_positions >= self.max_positions:
            return False, "Maximum positions limit reached"
        
        # Calculate position size based on risk
        risk_amount = capital * self.max_risk_per_trade
        
        # Avoid division by zero
        if stop_loss_distance <= 0:
            position_size = capital * self.position_size_pct  # Use default position size
        else:
            position_size = risk_amount / stop_loss_distance
        
        max_position_size = capital * self.position_size_pct
        
        if position_size > max_position_size:
            position_size = max_position_size
        
        # Ensure minimum position size
        if position_size <= 0:
            position_size = capital * 0.01  # 1% minimum
        
        return True, position_size
    
    def update_daily_pnl(self, pnl):
        """Update daily PnL"""
        self.daily_pnl += pnl
    
    def reset_daily_pnl(self):
        """Reset daily PnL (call at start of new day)"""
        self.daily_pnl = 0
    
    def position_opened(self):
        """Increment active positions counter"""
        self.active_positions += 1
    
    def position_closed(self):
        """Decrement active positions counter"""
        self.active_positions = max(0, self.active_positions - 1)

class LiveTradingSimulator:
    def __init__(self, strategy_params, initial_capital=10000, symbol='BTCUSDm'):
        """
        Live trading simulator
        
        Parameters:
        strategy_params: Dictionary with optimized strategy parameters
        initial_capital: Starting capital
        symbol: Currency pair to trade
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.strategy_params = strategy_params
        
        # Initialize strategy
        self.strategy = ForexHMMStrategy(
            symbol=symbol,
            n_components=strategy_params['n_components'],
            risk_reward_ratio=strategy_params['risk_reward_ratio']
        )
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            max_daily_loss=initial_capital * 0.05,  # 5% daily loss limit
            max_drawdown=0.15,  # 15% max drawdown
            max_positions=2,    # Max 2 concurrent positions for scalping
            position_size_pct=0.1,  # 10% position size
            max_risk_per_trade=0.02  # 2% risk per trade
        )
        
        # Trading state
        self.positions = []
        self.trade_history = []
        self.is_running = False
        self.last_signal_time = None
        
        # Performance tracking
        self.daily_stats = []
        self.equity_curve = []
        
    def initialize_strategy(self, training_period='3mo'):
        """
        Initialize and train the strategy
        """
        print("Initializing strategy...")
        
        # Fetch training data
        self.strategy.fetch_data(period=training_period, interval='5m')
        
        # Create features
        self.strategy.create_features()
        
        # Train HMM
        self.strategy.train_hmm()
        
        print("Strategy initialized and trained")
    
    def get_live_data(self):
        """
        Get latest market data (tries real-time tick data first, then recent bars)
        """
        try:
            # Use MT5 data if available, otherwise fallback to Yahoo Finance
            if hasattr(self.strategy, 'mt5_provider') and self.strategy.mt5_provider:
                try:
                    import MetaTrader5 as mt5
                    
                    # Try to get real-time tick data first
                    tick = mt5.symbol_info_tick(self.symbol)
                    if tick is not None:
                        # Create a data structure similar to OHLC bar
                        current_time = datetime.now()
                        tick_data = {
                            'Open': tick.bid,
                            'High': tick.bid,
                            'Low': tick.bid,
                            'Close': tick.bid,
                            'Volume': 1,
                            'time': current_time
                        }
                        print(f"Using real-time tick data: {tick.bid:.5f} at {current_time.strftime('%H:%M:%S')}")
                        return pd.Series(tick_data)
                    
                    # Fallback to latest 1-minute bar
                    data = self.strategy.mt5_provider.fetch_data(
                        symbol=self.symbol, 
                        timeframe=mt5.TIMEFRAME_M1,  # 1-minute timeframe for more frequent updates
                        bars=1
                    )
                    if data is not None and not data.empty:
                        print(f"Using 1-minute bar data")
                        return data.iloc[-1]  # Return latest bar
                        
                except ImportError:
                    data = None
            
            # Fallback to Yahoo Finance
            data = yf.download(self.symbol, period='1d', interval='1m', progress=False)
            if not data.empty:
                print(f"Using Yahoo Finance 1-minute data")
                return data.iloc[-1]  # Return latest bar
        except Exception as e:
            print(f"Error fetching live data: {e}")
        return None
    
    def update_strategy_data(self):
        """
        Update strategy with latest data point
        """
        latest_data = self.get_live_data()
        if latest_data is not None:
            # Add latest data to strategy (simplified)
            # In real implementation, you'd update the rolling window
            return True
        return False
    
    def generate_signal(self):
        """
        Generate trading signal based on current market conditions
        """
        try:
            # Update strategy with latest data
            if not self.update_strategy_data():
                return None
            
            # Generate signals using optimized parameters
            self.strategy.generate_signals_optimized(
                prob_threshold=self.strategy_params['prob_threshold'],
                state_prob_threshold=self.strategy_params['state_prob_threshold'],
                atr_multiplier=self.strategy_params['atr_multiplier']
            )
            
            # Get latest signal
            if not self.strategy.signals.empty:
                latest_signal = self.strategy.signals.iloc[-1]
                if latest_signal['signal'] != 0:
                    return {
                        'signal': latest_signal['signal'],
                        'stop_loss_pips': latest_signal['stop_loss_pips'],
                        'take_profit_pips': latest_signal['take_profit_pips'],
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            print(f"Error generating signal: {e}")
        
        return None
    
    def execute_trade(self, signal, current_price):
        """
        Execute a trade based on signal
        """
        # Calculate stop loss distance
        stop_loss_distance = signal['stop_loss_pips'] * 0.0001  # Convert pips to price
        
        # Ensure stop loss distance is reasonable
        if stop_loss_distance <= 0:
            print("Invalid stop loss distance, skipping trade")
            return False
        
        # Check risk management
        can_trade, position_info = self.risk_manager.can_open_position(
            self.capital, stop_loss_distance
        )
        
        if not can_trade:
            print(f"Trade rejected: {position_info}")
            return False
        
        # Calculate position size
        position_size = position_info
        
        # Create position
        if signal['signal'] == 1:  # Long
            stop_loss = current_price - stop_loss_distance
            take_profit = current_price + signal['take_profit_pips'] * 0.0001
        else:  # Short
            stop_loss = current_price + stop_loss_distance
            take_profit = current_price - signal['take_profit_pips'] * 0.0001
        
        position = {
            'id': len(self.positions) + 1,
            'direction': signal['signal'],
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'entry_time': signal['timestamp'],
            'status': 'open'
        }
        
        self.positions.append(position)
        self.risk_manager.position_opened()
        
        print(f"Position opened: {position['direction']} at {current_price:.5f}, "
              f"SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
        
        return True
    
    def check_positions(self, current_price):
        """
        Check open positions for exit conditions
        """
        closed_positions = []
        
        for position in self.positions:
            if position['status'] != 'open':
                continue
            
            exit_reason = None
            exit_price = current_price
            
            # Check stop loss
            if (position['direction'] == 1 and current_price <= position['stop_loss']) or \
               (position['direction'] == -1 and current_price >= position['stop_loss']):
                exit_reason = 'stop_loss'
            
            # Check take profit
            elif (position['direction'] == 1 and current_price >= position['take_profit']) or \
                 (position['direction'] == -1 and current_price <= position['take_profit']):
                exit_reason = 'take_profit'
            
            # Close position if exit condition met
            if exit_reason:
                position['status'] = 'closed'
                position['exit_price'] = exit_price
                position['exit_time'] = datetime.now()
                position['exit_reason'] = exit_reason
                
                # Calculate PnL (for crypto, we need to account for the actual position size in units)
                price_change = (exit_price - position['entry_price']) * position['direction']
                # For crypto, position_size represents the dollar amount, not units
                # So PnL = (price_change / entry_price) * position_size
                pnl = (price_change / position['entry_price']) * position['position_size']
                position['pnl'] = pnl
                
                # Update capital and risk manager
                self.capital += pnl
                self.risk_manager.update_daily_pnl(pnl)
                self.risk_manager.position_closed()
                
                # Add to trade history
                self.trade_history.append(position.copy())
                closed_positions.append(position)
                
                print(f"Position closed: {position['direction']} at {exit_price:.5f}, "
                      f"PnL: ${pnl:.2f}, Reason: {exit_reason}")
        
        # Remove closed positions
        self.positions = [p for p in self.positions if p['status'] == 'open']
        
        return closed_positions
    
    def update_equity_curve(self):
        """
        Update equity curve for performance tracking
        """
        unrealized_pnl = 0
        current_price = self.get_live_data()
        
        if current_price is not None:
            current_price = current_price['Close']
            
            # Calculate unrealized PnL
            for position in self.positions:
                if position['status'] == 'open':
                    price_change = (current_price - position['entry_price']) * position['direction']
                    unrealized_pnl += (price_change / position['entry_price']) * position['position_size']
        
        total_equity = self.capital + unrealized_pnl
        
        # Calculate drawdown safely
        if self.risk_manager.peak_capital > 0:
            drawdown = (self.risk_manager.peak_capital - total_equity) / self.risk_manager.peak_capital
        else:
            drawdown = 0
        
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'capital': self.capital,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': total_equity,
            'drawdown': drawdown
        })
    
    def trading_loop(self):
        """
        Main trading loop
        """
        print("Starting live trading simulation...")
        iteration_count = 0
        
        while self.is_running:
            try:
                iteration_count += 1
                current_time = datetime.now()
                print(f"\n--- Trading Loop Iteration {iteration_count} at {current_time.strftime('%H:%M:%S')} ---")
                
                # Get current market price
                latest_data = self.get_live_data()
                if latest_data is None:
                    print("No market data available, retrying...")
                    time.sleep(30)  # Wait 30 seconds before retry
                    continue
                
                current_price = latest_data['Close']
                print(f"Current price: {current_price:.5f}")
                
                # Validate price
                if current_price <= 0:
                    print("Invalid price data, skipping iteration")
                    time.sleep(60)
                    continue
                
                # Check existing positions
                self.check_positions(current_price)
                
                # Generate new signal (limit frequency to avoid overtrading)
                if self.last_signal_time is None or \
                   (current_time - self.last_signal_time).total_seconds() >= 300:  # 5 minutes
                    
                    signal = self.generate_signal()
                    if signal:
                        self.execute_trade(signal, current_price)
                        self.last_signal_time = current_time
                
                # Update equity curve
                self.update_equity_curve()
                
                # Reset daily PnL at start of new day
                if current_time.hour == 0 and current_time.minute == 0:
                    self.risk_manager.reset_daily_pnl()
                
                # Wait before next iteration
                time.sleep(10)  # Check every 10 seconds for more responsive live trading
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def start_trading(self):
        """
        Start the trading simulation
        """
        self.is_running = True
        self.trading_thread = threading.Thread(target=self.trading_loop)
        self.trading_thread.start()
    
    def stop_trading(self):
        """
        Stop the trading simulation
        """
        self.is_running = False
        if hasattr(self, 'trading_thread'):
            self.trading_thread.join()
        print("Trading simulation stopped")
    
    def get_performance_summary(self):
        """
        Get performance summary
        """
        if not self.trade_history:
            return {"message": "No trades executed yet"}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate profit factor safely
        if losing_trades > 0 and avg_loss != 0:
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades))
        elif winning_trades > 0:
            profit_factor = float('inf')  # All wins, no losses
        else:
            profit_factor = 0  # No profitable trades
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'current_capital': self.capital,
            'return_pct': (self.capital - self.initial_capital) / self.initial_capital,
            'active_positions': len(self.positions)
        }
    
    def plot_live_performance(self):
        """
        Plot live performance metrics
        """
        if not self.equity_curve:
            print("No performance data to plot")
            return
        
        import matplotlib.pyplot as plt
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        axes[0, 0].plot(equity_df['timestamp'], equity_df['total_equity'])
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Total Equity')
        
        # Drawdown
        axes[0, 1].fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, 
                               alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown %')
        
        # Trade PnL distribution
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            axes[1, 0].hist(trades_df['pnl'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Trade PnL Distribution')
            axes[1, 0].set_xlabel('PnL')
        
        # Win rate over time
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            trades_df['cumulative_wins'] = (trades_df['pnl'] > 0).cumsum()
            trades_df['cumulative_trades'] = range(1, len(trades_df) + 1)
            trades_df['rolling_win_rate'] = trades_df['cumulative_wins'] / trades_df['cumulative_trades']
            
            axes[1, 1].plot(trades_df['cumulative_trades'], trades_df['rolling_win_rate'])
            axes[1, 1].axhline(y=0.5, color='red', linestyle='--', label='50% Target')
            axes[1, 1].set_title('Win Rate Over Time')
            axes[1, 1].set_xlabel('Number of Trades')
            axes[1, 1].set_ylabel('Win Rate')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig

def main():
    """
    Main function to run live trading simulation
    """
    # Example optimized parameters (you would get these from optimization)
    strategy_params = {
        'n_components': 3,
        'risk_reward_ratio': 1.5,
        'prob_threshold': 0.6,
        'state_prob_threshold': 0.7,
        'atr_multiplier': 1.5
    }
    
    # Initialize simulator
    simulator = LiveTradingSimulator(
        strategy_params=strategy_params,
        initial_capital=10000,
        symbol='BTCUSDm'
    )
    
    # Initialize strategy
    simulator.initialize_strategy()
    
    # Start trading (run for demonstration)
    print("Starting live trading simulation...")
    print("This will run for 2 minutes as demonstration")
    
    simulator.start_trading()
    
    # Let it run for 2 minutes
    time.sleep(120)
    
    # Stop trading
    simulator.stop_trading()
    
    # Get performance summary
    performance = simulator.get_performance_summary()
    print("\nLive Trading Performance:")
    print("="*40)
    for key, value in performance.items():
        if isinstance(value, float):
            if 'rate' in key or 'return' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Plot performance
    simulator.plot_live_performance()
    
    return simulator

if __name__ == "__main__":
    simulator = main() 