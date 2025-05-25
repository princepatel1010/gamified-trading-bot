#!/usr/bin/env python3
"""
Futuristic Live Trading Simulator GUI
=====================================

A gaming-style interface for live forex trading simulation with:
- Real-time animated charts
- Futuristic UI design
- Live position tracking
- Animated P&L updates
- Gaming-style effects
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import seaborn as sns
from forex_hmm_strategy import ForexHMMStrategy
from live_trading_simulator import LiveTradingSimulator
import warnings
warnings.filterwarnings('ignore')

# Set dark theme for matplotlib
plt.style.use('dark_background')
# Create custom neon color palette
neon_colors = ['#00ff41', '#00ffff', '#ff0040', '#ffff00', '#ff00ff', '#00ff00']
sns.set_palette(neon_colors)

class FuturisticTradingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 FUTURISTIC FOREX TRADING SIMULATOR")
        self.root.configure(bg='#0a0a0a')
        self.root.geometry("1600x1000")
        self.root.state('zoomed')  # Maximize window
        
        # Trading data
        self.price_data = deque(maxlen=200)
        self.time_data = deque(maxlen=200)
        self.positions = []
        self.trades_history = []
        self.current_price = 0
        self.capital = 10000
        self.initial_capital = 10000
        
        # Animation control
        self.is_running = False
        self.animation = None
        
        # Initialize simulator
        self.init_simulator()
        
        # Create GUI
        self.create_futuristic_ui()
        
        # Start data simulation
        self.start_data_simulation()
        
        print("🎮 GUI initialized! You should see:")
        print("   - Price chart updating automatically")
        print("   - START TRADING button at the bottom")
        print("   - Current price updating in real-time")
        print("   - Click START TRADING to begin automated trading")
    
    def init_simulator(self):
        """Initialize the trading simulator"""
        strategy_params = {
            'n_components': 3,
            'risk_reward_ratio': 1.5,
            'prob_threshold': 0.6,
            'state_prob_threshold': 0.7,
            'atr_multiplier': 1.5
        }
        
        self.simulator = LiveTradingSimulator(
            strategy_params=strategy_params,
            initial_capital=10000,
            symbol='BTCUSDm'
        )
        
        # Initialize strategy in background
        threading.Thread(target=self.simulator.initialize_strategy, daemon=True).start()
    
    def create_futuristic_ui(self):
        """Create the futuristic gaming-style UI"""
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors for futuristic look
        style.configure('Futuristic.TFrame', background='#0a0a0a')
        style.configure('Neon.TLabel', background='#0a0a0a', foreground='#00ff41', 
                       font=('Orbitron', 12, 'bold'))
        style.configure('Title.TLabel', background='#0a0a0a', foreground='#00ffff', 
                       font=('Orbitron', 16, 'bold'))
        style.configure('Profit.TLabel', background='#0a0a0a', foreground='#00ff00', 
                       font=('Orbitron', 14, 'bold'))
        style.configure('Loss.TLabel', background='#0a0a0a', foreground='#ff0040', 
                       font=('Orbitron', 14, 'bold'))
        
        # Main container
        main_frame = ttk.Frame(self.root, style='Futuristic.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 FUTURISTIC FOREX TRADING SIMULATOR", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Top panel - Stats
        self.create_stats_panel(main_frame)
        
        # Middle panel - Chart
        self.create_chart_panel(main_frame)
        
        # Bottom panel - Controls and positions
        self.create_control_panel(main_frame)
    
    def create_stats_panel(self, parent):
        """Create the statistics panel"""
        stats_frame = ttk.Frame(parent, style='Futuristic.TFrame')
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create stats grid
        stats_grid = ttk.Frame(stats_frame, style='Futuristic.TFrame')
        stats_grid.pack()
        
        # Capital
        ttk.Label(stats_grid, text="💰 CAPITAL:", style='Neon.TLabel').grid(row=0, column=0, padx=20, sticky='w')
        self.capital_label = ttk.Label(stats_grid, text="$10,000.00", style='Profit.TLabel')
        self.capital_label.grid(row=0, column=1, padx=20, sticky='w')
        
        # P&L
        ttk.Label(stats_grid, text="📈 P&L:", style='Neon.TLabel').grid(row=0, column=2, padx=20, sticky='w')
        self.pnl_label = ttk.Label(stats_grid, text="$0.00", style='Neon.TLabel')
        self.pnl_label.grid(row=0, column=3, padx=20, sticky='w')
        
        # Current Price
        ttk.Label(stats_grid, text="💎 PRICE:", style='Neon.TLabel').grid(row=0, column=4, padx=20, sticky='w')
        self.price_label = ttk.Label(stats_grid, text="Loading...", style='Neon.TLabel')
        self.price_label.grid(row=0, column=5, padx=20, sticky='w')
        
        # Status indicator
        ttk.Label(stats_grid, text="📡 STATUS:", style='Neon.TLabel').grid(row=0, column=6, padx=20, sticky='w')
        self.status_label = ttk.Label(stats_grid, text="INITIALIZING", style='Neon.TLabel')
        self.status_label.grid(row=0, column=7, padx=20, sticky='w')
        
        # Win Rate
        ttk.Label(stats_grid, text="🎯 WIN RATE:", style='Neon.TLabel').grid(row=1, column=0, padx=20, sticky='w')
        self.winrate_label = ttk.Label(stats_grid, text="0%", style='Neon.TLabel')
        self.winrate_label.grid(row=1, column=1, padx=20, sticky='w')
        
        # Total Trades
        ttk.Label(stats_grid, text="⚡ TRADES:", style='Neon.TLabel').grid(row=1, column=2, padx=20, sticky='w')
        self.trades_label = ttk.Label(stats_grid, text="0", style='Neon.TLabel')
        self.trades_label.grid(row=1, column=3, padx=20, sticky='w')
        
        # Active Positions
        ttk.Label(stats_grid, text="🔥 POSITIONS:", style='Neon.TLabel').grid(row=1, column=4, padx=20, sticky='w')
        self.positions_label = ttk.Label(stats_grid, text="0", style='Neon.TLabel')
        self.positions_label.grid(row=1, column=5, padx=20, sticky='w')
    
    def create_chart_panel(self, parent):
        """Create the main chart panel"""
        chart_frame = ttk.Frame(parent, style='Futuristic.TFrame')
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create matplotlib figure with dark theme
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                                     facecolor='#0a0a0a', gridspec_kw={'height_ratios': [3, 1]})
        
        # Configure main price chart
        self.ax1.set_facecolor('#0a0a0a')
        self.ax1.tick_params(colors='#00ff41')
        self.ax1.set_title('🚀 LIVE PRICE CHART', color='#00ffff', fontsize=16, fontweight='bold')
        self.ax1.grid(True, alpha=0.3, color='#00ff41')
        
        # Configure P&L chart
        self.ax2.set_facecolor('#0a0a0a')
        self.ax2.tick_params(colors='#00ff41')
        self.ax2.set_title('💰 EQUITY CURVE', color='#00ffff', fontsize=14, fontweight='bold')
        self.ax2.grid(True, alpha=0.3, color='#00ff41')
        
        # Initialize empty plots
        self.price_line, = self.ax1.plot([], [], color='#00ff41', linewidth=2, label='Price')
        self.buy_markers = self.ax1.scatter([], [], color='#00ff00', s=100, marker='^', 
                                          label='BUY', alpha=0.8, edgecolors='white')
        self.sell_markers = self.ax1.scatter([], [], color='#ff0040', s=100, marker='v', 
                                           label='SELL', alpha=0.8, edgecolors='white')
        
        self.equity_line, = self.ax2.plot([], [], color='#00ffff', linewidth=2)
        
        self.ax1.legend(loc='upper left', facecolor='#0a0a0a', edgecolor='#00ff41')
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.tight_layout()
    
    def create_control_panel(self, parent):
        """Create the control panel"""
        control_frame = ttk.Frame(parent, style='Futuristic.TFrame')
        control_frame.pack(fill=tk.X, pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame, style='Futuristic.TFrame')
        button_frame.pack(side=tk.LEFT)
        
        # Start/Stop button - Make it more prominent
        self.start_button = tk.Button(button_frame, text="🚀 START TRADING", 
                                     command=self.toggle_trading,
                                     bg='#00ff41', fg='#0a0a0a', font=('Orbitron', 14, 'bold'),
                                     relief='raised', padx=30, pady=15, bd=3)
        self.start_button.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Reset button
        reset_button = tk.Button(button_frame, text="🔄 RESET", 
                               command=self.reset_simulation,
                               bg='#ff8800', fg='#0a0a0a', font=('Orbitron', 14, 'bold'),
                               relief='raised', padx=30, pady=15, bd=3)
        reset_button.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Positions panel
        positions_frame = ttk.Frame(control_frame, style='Futuristic.TFrame')
        positions_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(positions_frame, text="🔥 ACTIVE POSITIONS", style='Title.TLabel').pack()
        
        # Positions listbox with scrollbar
        listbox_frame = ttk.Frame(positions_frame, style='Futuristic.TFrame')
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        self.positions_listbox = tk.Listbox(listbox_frame, bg='#0a0a0a', fg='#00ff41',
                                          font=('Courier', 10), selectbackground='#00ff41',
                                          selectforeground='#0a0a0a', height=6)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.positions_listbox.yview)
        self.positions_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.positions_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def start_data_simulation(self):
        """Start the data simulation thread"""
        def simulate_data():
            # Initialize with some historical data
            base_price = 107000
            for i in range(50):
                timestamp = datetime.now() - timedelta(minutes=50-i)
                price = base_price + np.random.normal(0, 100) + np.sin(i/10) * 200
                self.time_data.append(timestamp)
                self.price_data.append(price)
            
            while True:
                # Always update price data (even when not trading)
                try:
                    live_data = self.simulator.get_live_data()
                    if live_data is not None:
                        self.current_price = live_data['Close']
                    else:
                        # Simulate price movement
                        self.current_price = self.price_data[-1] + np.random.normal(0, 50)
                except:
                    # Fallback simulation
                    self.current_price = self.price_data[-1] + np.random.normal(0, 50)
                
                # Add to data
                self.time_data.append(datetime.now())
                self.price_data.append(self.current_price)
                
                # Only trade when running
                if self.is_running:
                    # Simulate trading signals
                    if len(self.price_data) > 10 and np.random.random() < 0.05:  # 5% chance per update
                        self.simulate_trade_signal()
                    
                    # Update positions
                    self.update_positions()
                
                time.sleep(1)  # Update every second
        
        threading.Thread(target=simulate_data, daemon=True).start()
        
        # Start animation
        self.animation = FuncAnimation(self.fig, self.update_chart, interval=1000, blit=False)
    
    def simulate_trade_signal(self):
        """Simulate a trading signal"""
        if len(self.positions) < 2:  # Max 2 positions
            direction = np.random.choice([1, -1])  # 1 for buy, -1 for sell
            entry_price = self.current_price
            
            # Calculate stop loss and take profit
            atr = np.std(list(self.price_data)[-20:]) if len(self.price_data) >= 20 else 100
            stop_loss_distance = atr * 1.5
            
            if direction == 1:  # Buy
                stop_loss = entry_price - stop_loss_distance
                take_profit = entry_price + stop_loss_distance * 1.5
            else:  # Sell
                stop_loss = entry_price + stop_loss_distance
                take_profit = entry_price - stop_loss_distance * 1.5
            
            position = {
                'id': len(self.trades_history) + 1,
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': 1000,  # $1000 position
                'status': 'open'
            }
            
            self.positions.append(position)
            print(f"🚀 Position opened: {'BUY' if direction == 1 else 'SELL'} at {entry_price:.2f}")
    
    def update_positions(self):
        """Update open positions and check for exits"""
        closed_positions = []
        
        for position in self.positions:
            if position['status'] == 'open':
                # Check exit conditions
                if (position['direction'] == 1 and 
                    (self.current_price <= position['stop_loss'] or self.current_price >= position['take_profit'])) or \
                   (position['direction'] == -1 and 
                    (self.current_price >= position['stop_loss'] or self.current_price <= position['take_profit'])):
                    
                    # Close position
                    position['exit_price'] = self.current_price
                    position['exit_time'] = datetime.now()
                    position['status'] = 'closed'
                    
                    # Calculate P&L
                    price_change = (self.current_price - position['entry_price']) * position['direction']
                    position['pnl'] = (price_change / position['entry_price']) * position['size']
                    
                    self.capital += position['pnl']
                    closed_positions.append(position)
                    
                    exit_reason = 'TP' if position['pnl'] > 0 else 'SL'
                    print(f"💰 Position closed: {exit_reason} P&L: ${position['pnl']:.2f}")
        
        # Remove closed positions and add to history
        for pos in closed_positions:
            self.positions.remove(pos)
            self.trades_history.append(pos)
    
    def update_chart(self, frame):
        """Update the chart with new data"""
        if len(self.time_data) == 0:
            return
        
        try:
            # Update price chart
            self.price_line.set_data(self.time_data, self.price_data)
            
            # Update buy/sell markers
            buy_times, buy_prices = [], []
            sell_times, sell_prices = [], []
            
            for trade in self.trades_history[-20:]:  # Show last 20 trades
                if trade['direction'] == 1:
                    buy_times.append(trade['entry_time'])
                    buy_prices.append(trade['entry_price'])
                else:
                    sell_times.append(trade['entry_time'])
                    sell_prices.append(trade['entry_price'])
            
            # Clear previous markers safely
            try:
                self.buy_markers.remove()
            except (ValueError, AttributeError):
                pass
            try:
                self.sell_markers.remove()
            except (ValueError, AttributeError):
                pass
            
            # Add new markers
            if buy_times:
                self.buy_markers = self.ax1.scatter(buy_times, buy_prices, color='#00ff00', 
                                                  s=100, marker='^', alpha=0.8, edgecolors='white')
            else:
                self.buy_markers = self.ax1.scatter([], [], color='#00ff00', s=100, marker='^', alpha=0.8)
                
            if sell_times:
                self.sell_markers = self.ax1.scatter(sell_times, sell_prices, color='#ff0040', 
                                                   s=100, marker='v', alpha=0.8, edgecolors='white')
            else:
                self.sell_markers = self.ax1.scatter([], [], color='#ff0040', s=100, marker='v', alpha=0.8)
            
            # Update equity curve
            equity_data = [self.initial_capital]
            for trade in self.trades_history:
                equity_data.append(equity_data[-1] + trade['pnl'])
            
            if len(equity_data) > 1:
                equity_times = [self.time_data[0]] + [trade['exit_time'] for trade in self.trades_history]
                self.equity_line.set_data(equity_times, equity_data)
                self.ax2.relim()
                self.ax2.autoscale_view()
            
            # Auto-scale charts
            if len(self.time_data) > 1:
                self.ax1.relim()
                self.ax1.autoscale_view()
                
                # Format x-axis
                self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            
            # Update stats
            self.update_stats()
            
            # Update positions list
            self.update_positions_list()
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating chart: {e}")
            # Continue without crashing
    
    def update_stats(self):
        """Update the statistics display"""
        # Capital
        self.capital_label.config(text=f"${self.capital:,.2f}")
        if self.capital > self.initial_capital:
            self.capital_label.config(style='Profit.TLabel')
        elif self.capital < self.initial_capital:
            self.capital_label.config(style='Loss.TLabel')
        
        # P&L
        pnl = self.capital - self.initial_capital
        self.pnl_label.config(text=f"${pnl:+,.2f}")
        if pnl > 0:
            self.pnl_label.config(style='Profit.TLabel')
        elif pnl < 0:
            self.pnl_label.config(style='Loss.TLabel')
        
        # Current Price
        self.price_label.config(text=f"${self.current_price:,.2f}")
        
        # Win Rate
        if self.trades_history:
            wins = sum(1 for trade in self.trades_history if trade['pnl'] > 0)
            win_rate = wins / len(self.trades_history) * 100
            self.winrate_label.config(text=f"{win_rate:.1f}%")
        
        # Total Trades
        self.trades_label.config(text=str(len(self.trades_history)))
        
        # Active Positions
        self.positions_label.config(text=str(len(self.positions)))
        
        # Status
        if self.is_running:
            self.status_label.config(text="🟢 TRADING", style='Profit.TLabel')
        else:
            self.status_label.config(text="🟡 WATCHING", style='Neon.TLabel')
    
    def update_positions_list(self):
        """Update the positions listbox"""
        self.positions_listbox.delete(0, tk.END)
        
        for i, pos in enumerate(self.positions):
            direction = "BUY" if pos['direction'] == 1 else "SELL"
            unrealized_pnl = ((self.current_price - pos['entry_price']) * pos['direction'] / pos['entry_price']) * pos['size']
            
            status_color = "🟢" if unrealized_pnl > 0 else "🔴"
            position_text = f"{status_color} {direction} ${pos['entry_price']:.2f} | P&L: ${unrealized_pnl:+.2f}"
            self.positions_listbox.insert(tk.END, position_text)
    
    def toggle_trading(self):
        """Toggle trading on/off"""
        self.is_running = not self.is_running
        if self.is_running:
            self.start_button.config(text="⏸️ PAUSE TRADING", bg='#ff0040')
        else:
            self.start_button.config(text="🚀 START TRADING", bg='#00ff41')
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.is_running = False
        self.positions.clear()
        self.trades_history.clear()
        self.price_data.clear()
        self.time_data.clear()
        self.capital = self.initial_capital
        self.current_price = 107000
        
        # Clear charts
        self.price_line.set_data([], [])
        self.equity_line.set_data([], [])
        
        self.start_button.config(text="🚀 START TRADING", bg='#00ff41')
        print("🔄 Simulation reset")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def main():
    """Main function to run the futuristic trading GUI"""
    print("🚀 Starting Futuristic Trading Simulator...")
    gui = FuturisticTradingGUI()
    gui.run()

if __name__ == "__main__":
    main() 