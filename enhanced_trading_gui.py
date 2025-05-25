#!/usr/bin/env python3
"""
Enhanced Futuristic Trading GUI with Gaming Effects
===================================================

Features:
- Real-time candlestick charts
- Gaming-style sound effects
- Particle effects for trades
- HMM state visualization
- Advanced technical indicators
- Real MT5 data integration
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import seaborn as sns
from forex_hmm_strategy import ForexHMMStrategy
from live_trading_simulator import LiveTradingSimulator
from mt5_data_provider import MT5DataProvider
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

# Try to import sound library
try:
    import pygame
    pygame.mixer.init()
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

# Set dark theme
plt.style.use('dark_background')
# Create custom neon color palette
neon_colors = ['#00ff41', '#00ffff', '#ff0040', '#ffff00', '#ff00ff', '#00ff00']
try:
    import seaborn as sns
    sns.set_palette(neon_colors)
except ImportError:
    pass

class EnhancedTradingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎮 ENHANCED FOREX TRADING SIMULATOR")
        self.root.configure(bg='#000000')
        self.root.geometry("1800x1200")
        self.root.minsize(1600, 1000)  # Minimum size to ensure buttons are visible
        try:
            self.root.state('zoomed')  # Maximize window on Windows
        except:
            # Fallback for other systems
            try:
                self.root.attributes('-zoomed', True)
            except:
                pass  # If neither works, use default size
        
        # Trading data
        self.ohlc_data = deque(maxlen=200)
        self.time_data = deque(maxlen=200)
        self.positions = []
        self.trades_history = []
        self.current_price = 107000
        
        # HMM states data
        self.hmm_states = deque(maxlen=200)
        self.state_probabilities = deque(maxlen=200)
        
        # Animation and effects
        self.is_running = False
        self.animation = None
        self.particle_effects = []
        
        # Real trading data - Initialize early
        self.historical_data = None
        self.current_bar_index = 0
        self.is_simulating = False
        self.trading_symbol = 'BTCUSDm'  # Default symbol
        self.available_symbols = []
        
        # Trading settings
        self.risk_per_trade = 2.0  # Default 2% risk per trade
        self.custom_balance = 10000  # Default balance
        self.capital = 10000
        self.initial_capital = 10000
        
        # Contract specifications for proper position sizing and P&L calculation
        self.contract_specs = {
            "XAUUSDm": {"contract_size": 100, "pip_value": 0.01, "pip_size": 0.01},
            "BTCUSD": {"contract_size": 1, "pip_value": 1.0, "pip_size": 1.0},
            "BTCUSDm": {"contract_size": 1, "pip_value": 1.0, "pip_size": 1.0},
            "ETHUSD": {"contract_size": 1, "pip_value": 0.01, "pip_size": 0.01},
            "ETHUSDm": {"contract_size": 1, "pip_value": 0.01, "pip_size": 0.01},
            "USTECm": {"contract_size": 1, "pip_value": 0.01, "pip_size": 0.01},
            # Default for other symbols (forex)
            "DEFAULT": {"contract_size": 100000, "pip_value": 0.0001, "pip_size": 0.0001}
        }
        
        # Colors
        self.colors = {
            'bg': '#000000',
            'panel': '#0a0a0a',
            'neon_green': '#00ff41',
            'neon_blue': '#00ffff',
            'neon_red': '#ff0040',
            'neon_yellow': '#ffff00',
            'neon_purple': '#ff00ff',
            'profit': '#00ff00',
            'loss': '#ff0040'
        }
        
        # Initialize components
        self.init_mt5_provider()
        self.init_simulator()
        self.create_enhanced_ui()
        self.start_data_simulation()
        
        # Load sounds if available
        self.load_sounds()
    
    def init_mt5_provider(self):
        """Initialize MT5 data provider"""
        print("🔌 Initializing MT5 connection...")
        self.mt5_provider = MT5DataProvider()
        
        # Try to connect
        if self.mt5_provider.connect():
            print("✅ MT5 connected successfully")
            
            # Get available symbols
            symbols = self.mt5_provider.get_available_symbols()
            
            # Store all available symbols for dropdown
            self.available_symbols = []
            if symbols['crypto']:
                self.available_symbols.extend(symbols['crypto'])
            if symbols['forex']:
                self.available_symbols.extend(symbols['forex'])
            if symbols['all']:
                # Add other symbols not in crypto/forex
                for symbol in symbols['all']:
                    if symbol not in self.available_symbols:
                        self.available_symbols.append(symbol)
            
            # Set default symbol
            if self.available_symbols:
                # Prefer specific symbols if available
                preferred_symbols = ['BTCUSDm', 'XAUUSDm', 'USTECm', 'EURUSD', 'GBPUSD']
                for pref in preferred_symbols:
                    if pref in self.available_symbols:
                        self.trading_symbol = pref
                        break
                else:
                    self.trading_symbol = self.available_symbols[0]
                print(f"📊 Using symbol: {self.trading_symbol}")
                print(f"📋 Available symbols: {len(self.available_symbols)} found")
            else:
                print("⚠️ No symbols found, keeping default symbol")
        else:
            print("❌ MT5 connection failed, will use simulated data")
            self.mt5_provider = None
            # Set default symbols for offline mode
            self.available_symbols = ['BTCUSDm', 'XAUUSDm', 'USTECm', 'EURUSD', 'GBPUSD']
    
    def load_sounds(self):
        """Load sound effects"""
        self.sounds = {}
        if SOUND_AVAILABLE:
            try:
                # You can add sound files here
                # self.sounds['trade_open'] = pygame.mixer.Sound('trade_open.wav')
                # self.sounds['trade_close'] = pygame.mixer.Sound('trade_close.wav')
                pass
            except:
                pass
    
    def play_sound(self, sound_name):
        """Play a sound effect"""
        if SOUND_AVAILABLE and sound_name in self.sounds:
            self.sounds[sound_name].play()
    
    def init_simulator(self):
        """Initialize the trading simulator with real strategy"""
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
        def init_strategy():
            try:
                self.simulator.initialize_strategy()
                print("✅ Strategy initialized successfully")
            except Exception as e:
                print(f"❌ Strategy initialization failed: {e}")
        
        threading.Thread(target=init_strategy, daemon=True).start()
    
    def create_enhanced_ui(self):
        """Create the enhanced gaming-style UI"""
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Enhanced color scheme
        style.configure('Gaming.TFrame', background=self.colors['bg'])
        style.configure('Neon.TLabel', background=self.colors['bg'], foreground=self.colors['neon_green'], 
                       font=('Consolas', 11, 'bold'))
        style.configure('Title.TLabel', background=self.colors['bg'], foreground=self.colors['neon_blue'], 
                       font=('Consolas', 18, 'bold'))
        style.configure('Profit.TLabel', background=self.colors['bg'], foreground=self.colors['profit'], 
                       font=('Consolas', 13, 'bold'))
        style.configure('Loss.TLabel', background=self.colors['bg'], foreground=self.colors['loss'], 
                       font=('Consolas', 13, 'bold'))
        
        # Main container
        main_frame = ttk.Frame(self.root, style='Gaming.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Title with animation effect
        title_frame = ttk.Frame(main_frame, style='Gaming.TFrame')
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="🎮 ENHANCED FOREX TRADING SIMULATOR", 
                               style='Title.TLabel')
        title_label.pack()
        
        # Create main layout - Ensure controls are always visible
        self.create_dashboard(main_frame)
        self.create_charts_panel(main_frame)
        self.create_control_panel(main_frame)
        
        print("🎮 Enhanced GUI initialized with advanced controls!")
        print("   📊 Real-time candlestick charts with MT5 data")
        print("   ⚙️ Trading Settings: Symbol, Balance, Risk % with SAVE button")
        print("   💾 SAVE button applies balance and risk changes")
        print("   🚀 START button loads last week's 1-minute data")
        print("   🔄 RESET and AUTO buttons for control")
        print("   🧠 HMM states visualization on the right")
        print("   💰 Multiple chart panels with historical simulation")
        print(f"   📈 Default symbol: {self.trading_symbol}")
        print(f"   💰 Current balance: ${self.custom_balance:,.2f}")
        print(f"   ⚡ Current risk: {self.risk_per_trade}% per trade")
        print("   🎯 Adjust settings, click SAVE, then START to begin!")
    
    def create_dashboard(self, parent):
        """Create the main dashboard"""
        dashboard_frame = ttk.Frame(parent, style='Gaming.TFrame')
        dashboard_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left panel - Main stats
        left_panel = ttk.Frame(dashboard_frame, style='Gaming.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Stats grid
        stats_grid = ttk.Frame(left_panel, style='Gaming.TFrame')
        stats_grid.pack(pady=10)
        
        # Row 1
        ttk.Label(stats_grid, text="💰 CAPITAL:", style='Neon.TLabel').grid(row=0, column=0, padx=15, sticky='w')
        self.capital_label = ttk.Label(stats_grid, text=f"${self.capital:,.2f}", style='Profit.TLabel')
        self.capital_label.grid(row=0, column=1, padx=15, sticky='w')
        
        ttk.Label(stats_grid, text="📈 P&L:", style='Neon.TLabel').grid(row=0, column=2, padx=15, sticky='w')
        self.pnl_label = ttk.Label(stats_grid, text="$0.00", style='Neon.TLabel')
        self.pnl_label.grid(row=0, column=3, padx=15, sticky='w')
        
        ttk.Label(stats_grid, text="💎 PRICE:", style='Neon.TLabel').grid(row=0, column=4, padx=15, sticky='w')
        self.price_label = ttk.Label(stats_grid, text="Loading...", style='Neon.TLabel')
        self.price_label.grid(row=0, column=5, padx=15, sticky='w')
        
        # Row 2
        ttk.Label(stats_grid, text="🎯 WIN RATE:", style='Neon.TLabel').grid(row=1, column=0, padx=15, sticky='w')
        self.winrate_label = ttk.Label(stats_grid, text="0%", style='Neon.TLabel')
        self.winrate_label.grid(row=1, column=1, padx=15, sticky='w')
        
        ttk.Label(stats_grid, text="⚡ TRADES:", style='Neon.TLabel').grid(row=1, column=2, padx=15, sticky='w')
        self.trades_label = ttk.Label(stats_grid, text="0", style='Neon.TLabel')
        self.trades_label.grid(row=1, column=3, padx=15, sticky='w')
        
        ttk.Label(stats_grid, text="🔥 POSITIONS:", style='Neon.TLabel').grid(row=1, column=4, padx=15, sticky='w')
        self.positions_label = ttk.Label(stats_grid, text="0", style='Neon.TLabel')
        self.positions_label.grid(row=1, column=5, padx=15, sticky='w')
        
        # Row 3 - Total P&L (realized)
        ttk.Label(stats_grid, text="💰 TOTAL P&L:", style='Neon.TLabel').grid(row=2, column=0, padx=15, sticky='w')
        self.total_pnl_label = ttk.Label(stats_grid, text="$0.00", style='Neon.TLabel')
        self.total_pnl_label.grid(row=2, column=1, padx=15, sticky='w')
        
        # Middle panel - Trading Settings
        middle_panel = ttk.Frame(dashboard_frame, style='Gaming.TFrame')
        middle_panel.pack(side=tk.LEFT, padx=20, fill=tk.BOTH, expand=True)
        
        ttk.Label(middle_panel, text="⚙️ TRADING SETTINGS", style='Title.TLabel').pack()
        
        # Settings container
        settings_container = ttk.Frame(middle_panel, style='Gaming.TFrame')
        settings_container.pack(pady=10, fill=tk.X)
        
        # Row 1 - Symbol and Balance
        row1 = ttk.Frame(settings_container, style='Gaming.TFrame')
        row1.pack(fill=tk.X, pady=5)
        
        # Trading Symbol Selection
        ttk.Label(row1, text="📊 SYMBOL:", style='Neon.TLabel').pack(side=tk.LEFT, padx=(0,5))
        self.symbol_var = tk.StringVar(value=self.trading_symbol)
        self.symbol_combo = ttk.Combobox(row1, textvariable=self.symbol_var, 
                                        values=self.available_symbols,
                                        state="readonly", width=12,
                                        font=('Consolas', 10, 'bold'))
        self.symbol_combo.pack(side=tk.LEFT, padx=(0,15))
        self.symbol_combo.bind('<<ComboboxSelected>>', self.on_symbol_change)
        
        # Timeframe Selection
        ttk.Label(row1, text="⏰ TIMEFRAME:", style='Neon.TLabel').pack(side=tk.LEFT, padx=(0,5))
        self.timeframe_var = tk.StringVar(value="M1")
        timeframe_options = ["M1", "M5", "M15", "M30", "H1"]
        self.timeframe_combo = ttk.Combobox(row1, textvariable=self.timeframe_var,
                                           values=timeframe_options,
                                           state="readonly", width=8,
                                           font=('Consolas', 10, 'bold'))
        self.timeframe_combo.pack(side=tk.LEFT, padx=(0,20))
        self.timeframe_combo.bind('<<ComboboxSelected>>', self.on_timeframe_change)
        
        # Account Balance
        ttk.Label(row1, text="💰 BALANCE:", style='Neon.TLabel').pack(side=tk.LEFT, padx=(0,5))
        self.balance_var = tk.StringVar(value=str(self.custom_balance))
        balance_entry = tk.Entry(row1, textvariable=self.balance_var, width=10,
                               bg=self.colors['panel'], fg=self.colors['neon_green'],
                               font=('Consolas', 10, 'bold'), relief='raised', bd=2)
        balance_entry.pack(side=tk.LEFT, padx=(0,10))
        
        # Current Balance Display
        ttk.Label(row1, text="CURRENT:", style='Neon.TLabel').pack(side=tk.LEFT, padx=(0,5))
        self.current_balance_label = ttk.Label(row1, text=f"${self.custom_balance:.2f}", style='Profit.TLabel')
        self.current_balance_label.pack(side=tk.LEFT, padx=(0,20))
        
        # Row 2 - Risk and Save Button
        row2 = ttk.Frame(settings_container, style='Gaming.TFrame')
        row2.pack(fill=tk.X, pady=5)
        
        # Risk per Trade
        ttk.Label(row2, text="⚡ RISK %:", style='Neon.TLabel').pack(side=tk.LEFT, padx=(0,5))
        self.risk_var = tk.StringVar(value=str(self.risk_per_trade))
        risk_entry = tk.Entry(row2, textvariable=self.risk_var, width=8,
                            bg=self.colors['panel'], fg=self.colors['neon_green'],
                            font=('Consolas', 10, 'bold'), relief='raised', bd=2)
        risk_entry.pack(side=tk.LEFT, padx=(0,10))
        
        # Current Risk Display
        ttk.Label(row2, text="CURRENT:", style='Neon.TLabel').pack(side=tk.LEFT, padx=(0,5))
        self.current_risk_label = ttk.Label(row2, text=f"{self.risk_per_trade}%", style='Profit.TLabel')
        self.current_risk_label.pack(side=tk.LEFT, padx=(0,15))
        
        # SAVE Button
        save_button = tk.Button(row2, text="💾 SAVE", 
                              command=self.save_settings,
                              bg=self.colors['neon_blue'], fg=self.colors['bg'], 
                              font=('Consolas', 11, 'bold'),
                              relief='raised', padx=15, pady=8, bd=3,
                              activebackground='#0088cc')
        save_button.pack(side=tk.LEFT, padx=(0,10))
        
        # Status message
        self.settings_status_label = ttk.Label(row2, text="", style='Neon.TLabel')
        self.settings_status_label.pack(side=tk.LEFT, padx=(0,10))
        
        # Row 3 - Control buttons
        row3 = ttk.Frame(settings_container, style='Gaming.TFrame')
        row3.pack(fill=tk.X, pady=5)
        
        # Control buttons - Horizontal layout
        buttons_frame = ttk.Frame(row3, style='Gaming.TFrame')
        buttons_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.start_button = tk.Button(buttons_frame, text="🚀 START", 
                                     command=self.toggle_trading,
                                     bg=self.colors['neon_green'], fg=self.colors['bg'], 
                                     font=('Consolas', 11, 'bold'),
                                     relief='raised', padx=15, pady=8, bd=3,
                                     activebackground=self.colors['profit'])
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        reset_button = tk.Button(buttons_frame, text="🔄 RESET", 
                               command=self.reset_simulation,
                               bg=self.colors['neon_yellow'], fg=self.colors['bg'], 
                               font=('Consolas', 11, 'bold'),
                               relief='raised', padx=15, pady=8, bd=3,
                               activebackground='#ffcc00')
        reset_button.pack(side=tk.LEFT, padx=5)
        
        strategy_button = tk.Button(buttons_frame, text="🧠 AUTO", 
                                  command=self.toggle_auto_strategy,
                                  bg=self.colors['neon_purple'], fg=self.colors['bg'], 
                                  font=('Consolas', 11, 'bold'),
                                  relief='raised', padx=15, pady=8, bd=3,
                                  activebackground='#cc00cc')
        strategy_button.pack(side=tk.LEFT, padx=5)
        
        # Right panel - HMM State indicator
        right_panel = ttk.Frame(dashboard_frame, style='Gaming.TFrame')
        right_panel.pack(side=tk.RIGHT, padx=20)
        
        ttk.Label(right_panel, text="🧠 HMM STATE", style='Title.TLabel').pack()
        self.state_label = ttk.Label(right_panel, text="ANALYZING...", style='Neon.TLabel')
        self.state_label.pack(pady=5)
        
        # Progress indicator for simulation
        ttk.Label(right_panel, text="📈 PROGRESS", style='Neon.TLabel').pack(pady=(10,0))
        self.progress_label = ttk.Label(right_panel, text="0%", style='Neon.TLabel')
        self.progress_label.pack(pady=2)
        
        # Status indicator
        ttk.Label(right_panel, text="📡 STATUS", style='Neon.TLabel').pack(pady=(10,0))
        self.status_label = ttk.Label(right_panel, text="READY", style='Neon.TLabel')
        self.status_label.pack(pady=2)
        
        # State probability bars
        self.state_bars_frame = ttk.Frame(right_panel, style='Gaming.TFrame')
        self.state_bars_frame.pack(pady=10)
    
    def create_charts_panel(self, parent):
        """Create the charts panel with multiple subplots"""
        charts_frame = ttk.Frame(parent, style='Gaming.TFrame')
        charts_frame.pack(fill=tk.BOTH, expand=False, pady=10)
        
        # Create figure with multiple subplots - Make it smaller to leave room for controls
        self.fig = plt.figure(figsize=(18, 8), facecolor=self.colors['bg'])
        
        # Define subplot layout
        gs = self.fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[3, 1])
        
        # Main price chart (candlesticks)
        self.ax_price = self.fig.add_subplot(gs[0, 0])
        self.setup_price_chart()
        
        # HMM states chart
        self.ax_states = self.fig.add_subplot(gs[0, 1])
        self.setup_states_chart()
        
        # Equity curve
        self.ax_equity = self.fig.add_subplot(gs[1, :])
        self.setup_equity_chart()
        
        # Volume/Activity chart
        self.ax_volume = self.fig.add_subplot(gs[2, :])
        self.setup_volume_chart()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.tight_layout()
    
    def setup_price_chart(self):
        """Setup the main price chart"""
        self.ax_price.set_facecolor(self.colors['panel'])
        self.ax_price.tick_params(colors=self.colors['neon_green'])
        
        # Dynamic title based on trading symbol and mode
        if self.is_simulating:
            title = f'🚀 {self.trading_symbol} - REAL HISTORICAL DATA'
        elif self.is_running:
            title = f'🚀 {self.trading_symbol} - LIVE MT5 DATA'
        else:
            title = f'🚀 {self.trading_symbol} - READY'
        
        self.ax_price.set_title(title, color=self.colors['neon_blue'], 
                               fontsize=14, fontweight='bold')
        self.ax_price.grid(True, alpha=0.2, color=self.colors['neon_green'])
        
        # Initialize empty candlestick data
        self.candlestick_patches = []
        self.buy_markers = []
        self.sell_markers = []
    
    def setup_states_chart(self):
        """Setup the HMM states chart"""
        self.ax_states.set_facecolor(self.colors['panel'])
        self.ax_states.tick_params(colors=self.colors['neon_green'])
        self.ax_states.set_title('🧠 HMM STATES', color=self.colors['neon_blue'], 
                                fontsize=12, fontweight='bold')
        self.ax_states.set_ylim(0, 3)
        self.ax_states.set_ylabel('State', color=self.colors['neon_green'])
    
    def setup_equity_chart(self):
        """Setup the equity curve chart"""
        self.ax_equity.set_facecolor(self.colors['panel'])
        self.ax_equity.tick_params(colors=self.colors['neon_green'])
        self.ax_equity.set_title('💰 EQUITY CURVE', color=self.colors['neon_blue'], 
                                fontsize=12, fontweight='bold')
        self.ax_equity.grid(True, alpha=0.2, color=self.colors['neon_green'])
        
        self.equity_line, = self.ax_equity.plot([], [], color=self.colors['neon_blue'], linewidth=2)
    
    def setup_volume_chart(self):
        """Setup the volume/activity chart"""
        self.ax_volume.set_facecolor(self.colors['panel'])
        self.ax_volume.tick_params(colors=self.colors['neon_green'])
        self.ax_volume.set_title('📊 TRADING ACTIVITY', color=self.colors['neon_blue'], 
                                fontsize=12, fontweight='bold')
        self.ax_volume.grid(True, alpha=0.2, color=self.colors['neon_green'])
    
    def create_control_panel(self, parent):
        """Create the positions panel (buttons moved to header)"""
        control_frame = ttk.Frame(parent, style='Gaming.TFrame')
        control_frame.pack(fill=tk.X, pady=10)
        
        # Positions panel
        positions_frame = ttk.Frame(control_frame, style='Gaming.TFrame')
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        ttk.Label(positions_frame, text="🔥 ACTIVE POSITIONS", style='Title.TLabel').pack()
        
        # Enhanced positions listbox
        listbox_frame = ttk.Frame(positions_frame, style='Gaming.TFrame')
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.positions_listbox = tk.Listbox(listbox_frame, bg=self.colors['panel'], 
                                          fg=self.colors['neon_green'],
                                          font=('Consolas', 10), 
                                          selectbackground=self.colors['neon_green'],
                                          selectforeground=self.colors['bg'], height=6)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.positions_listbox.yview)
        self.positions_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.positions_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def start_data_simulation(self):
        """Start the enhanced data simulation"""
        def simulate_data():
            # Wait for real historical data to be loaded
            print("📊 Data simulation thread started - waiting for historical data...")
            
            # Initialize empty - no dummy data
            # Real data will be loaded when user clicks START button
            
            while True:
                if self.is_running and self.is_simulating and self.historical_data is not None:
                    # Use real historical data for simulation
                    if self.current_bar_index < len(self.historical_data):
                        row = self.historical_data.iloc[self.current_bar_index]
                        
                        # Create OHLC bar from historical data
                        new_ohlc = {
                            'time': row.name,
                            'open': row['Open'],
                            'high': row['High'],
                            'low': row['Low'],
                            'close': row['Close'],
                            'volume': row['Volume']
                        }
                        
                        self.ohlc_data.append(new_ohlc)
                        self.time_data.append(row.name)
                        self.current_price = row['Close']
                        
                        # Simulate HMM state (you could integrate real HMM here)
                        state = np.random.randint(0, 3)
                        self.hmm_states.append(state)
                        self.state_probabilities.append([np.random.random(), np.random.random(), np.random.random()])
                        
                        # Generate trading signals using real HMM strategy (less frequently)
                        if len(self.ohlc_data) > 50 and self.current_bar_index % 10 == 0:  # Every 10 bars
                            self.generate_hmm_trade_signal()
                        
                        # Fallback: Generate simple trades occasionally for testing
                        if len(self.ohlc_data) > 20 and len(self.positions) < 2 and self.current_bar_index % 25 == 0:
                            self.generate_simple_test_trade()
                        
                        # Update positions
                        self.update_positions()
                        
                        self.current_bar_index += 1
                        
                        # Check if simulation is complete
                        if self.current_bar_index >= len(self.historical_data):
                            print("📊 Historical simulation completed!")
                            self.is_running = False
                            self.is_simulating = False
                            self.start_button.config(text="🚀 START", bg=self.colors['neon_green'])
                    
                elif self.is_running and not self.is_simulating:
                    # Live data mode - fetch real data from MT5
                    if self.mt5_provider and self.mt5_provider.is_connected():
                        try:
                            # Get selected timeframe for live data
                            timeframe_str = self.timeframe_var.get()
                            timeframe_map = {
                                "M1": mt5.TIMEFRAME_M1,
                                "M5": mt5.TIMEFRAME_M5,
                                "M15": mt5.TIMEFRAME_M15,
                                "M30": mt5.TIMEFRAME_M30,
                                "H1": mt5.TIMEFRAME_H1
                            }
                            mt5_timeframe = timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M1)
                            
                            # Fetch latest bar from MT5
                            latest_data = self.mt5_provider.fetch_data(
                                symbol=self.trading_symbol,
                                timeframe=mt5_timeframe,
                                bars=1
                            )
                            
                            if latest_data is not None and len(latest_data) > 0:
                                row = latest_data.iloc[-1]
                                self.current_price = row['Close']
                                
                                # Create new OHLC bar from real data
                                new_ohlc = {
                                    'time': row.name,
                                    'open': row['Open'],
                                    'high': row['High'],
                                    'low': row['Low'],
                                    'close': row['Close'],
                                    'volume': row['Volume']
                                }
                                
                                self.ohlc_data.append(new_ohlc)
                                self.time_data.append(row.name)
                                
                                print(f"📡 Live data: {self.trading_symbol} @ {self.current_price:.2f}")
                            else:
                                print("⚠️ No live data received")
                                
                        except Exception as e:
                            print(f"❌ Error fetching live data: {e}")
                    else:
                        print("❌ No MT5 connection for live data")
                    
                    # Simulate HMM state
                    state = np.random.randint(0, 3)
                    self.hmm_states.append(state)
                    self.state_probabilities.append([np.random.random(), np.random.random(), np.random.random()])
                    
                    # Generate trading signals - increased probability
                    if len(self.ohlc_data) > 20 and np.random.random() < 0.1:  # 10% chance
                        self.generate_trade_signal()
                    
                    # Update positions
                    self.update_positions()
                
                # Adjust sleep time based on mode and timeframe
                if self.is_simulating:
                    # Faster for lower timeframes, slower for higher timeframes
                    timeframe_str = self.timeframe_var.get()
                    if timeframe_str == "M1":
                        time.sleep(0.2)  # Fast for M1 scalping
                    elif timeframe_str == "M5":
                        time.sleep(0.5)  # Moderate for M5
                    elif timeframe_str == "M15":
                        time.sleep(0.8)  # Slower for M15
                    else:
                        time.sleep(1.0)  # Slowest for M30/H1
                else:
                    # Live mode - adjust based on timeframe
                    timeframe_str = self.timeframe_var.get()
                    if timeframe_str == "M1":
                        time.sleep(5)   # 5 seconds for M1 live
                    elif timeframe_str == "M5":
                        time.sleep(30)  # 30 seconds for M5 live
                    elif timeframe_str == "M15":
                        time.sleep(60)  # 1 minute for M15 live
                    elif timeframe_str == "M30":
                        time.sleep(120) # 2 minutes for M30 live
                    else:  # H1
                        time.sleep(300) # 5 minutes for H1 live
        
        threading.Thread(target=simulate_data, daemon=True).start()
        
        # Start animation - faster updates for smoother rendering
        self.animation = FuncAnimation(self.fig, self.update_charts, interval=500, blit=False)
    
    def generate_hmm_trade_signal(self):
        """Generate trading signal using real HMM strategy"""
        if len(self.positions) >= 3:  # Max 3 positions
            return
        
        try:
            # Check if simulator and strategy are properly initialized
            if not hasattr(self.simulator, 'strategy') or not self.simulator.strategy:
                # Use simplified signal generation
                self.generate_simplified_signal_from_ohlc()
                return
            
            # Use the simulator's generate_signal method which properly handles the strategy
            signal = self.simulator.generate_signal()
            if signal and signal.get('signal', 0) != 0:
                self.execute_hmm_trade(signal)
            else:
                # No signal generated, occasionally use simplified logic
                if np.random.random() < 0.1:  # 10% chance
                    self.generate_simplified_signal_from_ohlc()
                
        except Exception as e:
            print(f"❌ Error generating HMM signal: {e}")
            # Fallback to simplified signal occasionally
            if np.random.random() < 0.1:  # 10% chance
                self.generate_simplified_signal_from_ohlc()
    
    def generate_simple_test_trade(self):
        """Generate a simple test trade for demonstration"""
        if len(self.positions) >= 2:  # Max 2 positions for testing
            return
        
        direction = np.random.choice([1, -1])
        entry_price = self.current_price
        
        # Scalping stop loss and take profit - much tighter levels
        timeframe_str = self.timeframe_var.get()
        
        # Adjust SL/TP based on timeframe for scalping
        if timeframe_str == "M1":
            stop_loss_distance = entry_price * 0.001  # 0.1% for M1 scalping
        elif timeframe_str == "M5":
            stop_loss_distance = entry_price * 0.002  # 0.2% for M5 scalping
        elif timeframe_str == "M15":
            stop_loss_distance = entry_price * 0.003  # 0.3% for M15
        elif timeframe_str == "M30":
            stop_loss_distance = entry_price * 0.004  # 0.4% for M30
        else:  # H1
            stop_loss_distance = entry_price * 0.005  # 0.5% for H1
        
        if direction == 1:  # Buy
            stop_loss = entry_price - stop_loss_distance
            take_profit = entry_price + stop_loss_distance * 1.5  # 1:1.5 RR
        else:  # Sell
            stop_loss = entry_price + stop_loss_distance
            take_profit = entry_price - stop_loss_distance * 1.5
        
        # Calculate position size
        risk_amount = self.capital * (self.risk_per_trade / 100)
        lot_size = self.calculate_lot_size(risk_amount, stop_loss_distance, entry_price)
        
        position = {
            'id': len(self.trades_history) + 1,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'risk_amount': risk_amount,
            'status': 'open'
        }
        
        self.positions.append(position)
        self.play_sound('trade_open')
        self.add_particle_effect(entry_price, direction)
        
        print(f"🧪 Scalping Trade ({timeframe_str}): {'BUY' if direction == 1 else 'SELL'} at {entry_price:.2f}")
        print(f"   SL: {stop_loss:.2f} | TP: {take_profit:.2f} | Distance: {stop_loss_distance:.2f}")
        print(f"   Lot: {lot_size} | Risk: ${risk_amount:.2f} | Total positions: {len(self.positions)}")

    def generate_simplified_signal_from_ohlc(self):
        """Generate simplified trading signal from OHLC data"""
        if len(self.ohlc_data) < 20:
            return
        
        # Get recent price data
        recent_closes = [bar['close'] for bar in list(self.ohlc_data)[-20:]]
        current_price = recent_closes[-1]
        
        # Simple moving average
        sma_20 = np.mean(recent_closes)
        
        # Simple RSI calculation
        price_changes = np.diff(recent_closes)
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        if len(gains) >= 14:
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
        else:
            rsi = 50  # Neutral
        
        # Generate signal based on conditions - increased probability for more trades
        if current_price > sma_20 and 30 < rsi < 70 and np.random.random() < 0.15:  # 15% chance
            # Buy signal
            self.execute_simple_trade(1, current_price)
        elif current_price < sma_20 and 30 < rsi < 70 and np.random.random() < 0.15:  # 15% chance
            # Sell signal
            self.execute_simple_trade(-1, current_price)
    
    def generate_simplified_signal(self, df):
        """Generate simplified trading signal based on price action (legacy method)"""
        if len(df) < 20:
            return
        
        # Simple momentum strategy
        current_price = df['Close'].iloc[-1]
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        rsi = self.calculate_rsi(df['Close'], 14)
        
        # Generate signal based on conditions
        if current_price > sma_20 and rsi < 70 and np.random.random() < 0.03:
            # Buy signal
            self.execute_simple_trade(1, current_price)
        elif current_price < sma_20 and rsi > 30 and np.random.random() < 0.03:
            # Sell signal
            self.execute_simple_trade(-1, current_price)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def execute_simple_trade(self, direction, entry_price):
        """Execute a simple trade with proper risk management"""
        # Scalping stop loss based on timeframe
        timeframe_str = self.timeframe_var.get()
        
        # Tight scalping levels based on timeframe
        if timeframe_str == "M1":
            stop_loss_distance = entry_price * 0.001  # 0.1% for M1 scalping
        elif timeframe_str == "M5":
            stop_loss_distance = entry_price * 0.002  # 0.2% for M5 scalping
        elif timeframe_str == "M15":
            stop_loss_distance = entry_price * 0.003  # 0.3% for M15
        elif timeframe_str == "M30":
            stop_loss_distance = entry_price * 0.004  # 0.4% for M30
        else:  # H1
            stop_loss_distance = entry_price * 0.005  # 0.5% for H1
        
        if direction == 1:  # Buy
            stop_loss = entry_price - stop_loss_distance
            take_profit = entry_price + stop_loss_distance * 1.5  # 1:1.5 RR
        else:  # Sell
            stop_loss = entry_price + stop_loss_distance
            take_profit = entry_price - stop_loss_distance * 1.5
        
        # Calculate position size using proper lot sizing
        risk_amount = self.capital * (self.risk_per_trade / 100)
        stop_loss_distance = abs(entry_price - stop_loss)
        lot_size = self.calculate_lot_size(risk_amount, stop_loss_distance, entry_price)
        
        position = {
            'id': len(self.trades_history) + 1,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'risk_amount': risk_amount,  # Store original risk amount
            'status': 'open'
        }
        
        self.positions.append(position)
        self.play_sound('trade_open')
        self.add_particle_effect(entry_price, direction)
        
        timeframe_str = self.timeframe_var.get()
        print(f"🎯 Scalping Trade ({timeframe_str}): {'BUY' if direction == 1 else 'SELL'} at {entry_price:.2f}")
        print(f"   SL: {stop_loss:.2f} | TP: {take_profit:.2f} | Distance: {stop_loss_distance:.2f}")
        print(f"   Risk: ${risk_amount:.2f} | Lot: {lot_size} | Total positions: {len(self.positions)}")
    
    def execute_hmm_trade(self, signal):
        """Execute trade from real HMM strategy signal"""
        direction = signal['signal']
        entry_price = self.current_price
        
        # Use signal's stop loss and take profit
        if 'stop_loss_pips' in signal and 'take_profit_pips' in signal:
            pip_value = 0.0001 if 'JPY' not in self.trading_symbol else 0.01
            stop_loss = entry_price - (signal['stop_loss_pips'] * pip_value * direction)
            take_profit = entry_price + (signal['take_profit_pips'] * pip_value * direction)
        else:
            # Fallback to ATR-based levels
            recent_prices = [bar['close'] for bar in list(self.ohlc_data)[-20:]]
            atr = np.std(recent_prices) if len(recent_prices) >= 20 else entry_price * 0.02
            
            stop_loss_distance = atr * 1.5
            if direction == 1:
                stop_loss = entry_price - stop_loss_distance
                take_profit = entry_price + stop_loss_distance * 1.5
            else:
                stop_loss = entry_price + stop_loss_distance
                take_profit = entry_price - stop_loss_distance * 1.5
        
        # Calculate position size using proper lot sizing
        risk_amount = self.capital * (self.risk_per_trade / 100)
        stop_loss_distance = abs(entry_price - stop_loss)
        lot_size = self.calculate_lot_size(risk_amount, stop_loss_distance, entry_price)
        
        position = {
            'id': len(self.trades_history) + 1,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': lot_size,
            'risk_amount': risk_amount,  # Store original risk amount
            'status': 'open'
        }
        
        self.positions.append(position)
        self.play_sound('trade_open')
        self.add_particle_effect(entry_price, direction)
        
        print(f"🧠 HMM Strategy Trade: {'BUY' if direction == 1 else 'SELL'} at {entry_price:.5f}")
        print(f"   SL: {stop_loss:.5f} | TP: {take_profit:.5f} | Risk: ${risk_amount:.2f} | Lot: {lot_size}")
    
    def generate_trade_signal(self):
        """Generate a trading signal using strategy logic"""
        if len(self.positions) < 3:  # Max 3 positions
            # Use actual strategy if available
            try:
                signal = self.simulator.generate_signal()
                if signal:
                    self.execute_trade_from_signal(signal)
                    return
            except:
                pass
            
            # Fallback to simulated signal
            direction = np.random.choice([1, -1])
            entry_price = self.current_price
            
            # Calculate stop loss and take profit based on recent volatility
            recent_prices = [bar['close'] for bar in list(self.ohlc_data)[-20:]]
            atr = np.std(recent_prices) if len(recent_prices) >= 20 else 100
            
            stop_loss_distance = atr * 1.5
            
            if direction == 1:  # Buy
                stop_loss = entry_price - stop_loss_distance
                take_profit = entry_price + stop_loss_distance * 1.5
            else:  # Sell
                stop_loss = entry_price + stop_loss_distance
                take_profit = entry_price - stop_loss_distance * 1.5
            
            # Calculate position size using proper lot sizing
            risk_amount = self.capital * (self.risk_per_trade / 100)
            stop_loss_distance = abs(entry_price - stop_loss)
            lot_size = self.calculate_lot_size(risk_amount, stop_loss_distance, entry_price)
            
            position = {
                'id': len(self.trades_history) + 1,
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'lot_size': lot_size,
                'risk_amount': risk_amount,  # Store original risk amount
                'status': 'open'
            }
            
            self.positions.append(position)
            self.play_sound('trade_open')
            
            # Add particle effect
            self.add_particle_effect(entry_price, direction)
            
            print(f"🚀 Position opened: {'BUY' if direction == 1 else 'SELL'} at {entry_price:.2f}")
    
    def execute_trade_from_signal(self, signal):
        """Execute trade from actual strategy signal"""
        direction = signal['signal']
        entry_price = self.current_price
        
        # Calculate position size using proper lot sizing
        spec = self.get_contract_spec()
        stop_loss_price = entry_price - (signal['stop_loss_pips'] * spec['pip_size'] * direction)
        risk_amount = self.capital * (self.risk_per_trade / 100)
        stop_loss_distance = abs(entry_price - stop_loss_price)
        lot_size = self.calculate_lot_size(risk_amount, stop_loss_distance, entry_price)
        
        position = {
            'id': len(self.trades_history) + 1,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'stop_loss': stop_loss_price,
            'take_profit': entry_price + (signal['take_profit_pips'] * spec['pip_size'] * direction),
            'lot_size': lot_size,
            'risk_amount': risk_amount,  # Store original risk amount
            'status': 'open'
        }
        
        self.positions.append(position)
        self.play_sound('trade_open')
        self.add_particle_effect(entry_price, direction)
    
    def add_particle_effect(self, price, direction):
        """Add particle effect for trade execution"""
        color = self.colors['profit'] if direction == 1 else self.colors['loss']
        effect = {
            'price': price,
            'time': datetime.now(),
            'color': color,
            'direction': direction,
            'life': 30  # frames
        }
        self.particle_effects.append(effect)
    
    def update_positions(self):
        """Update positions with enhanced effects"""
        closed_positions = []
        
        for position in self.positions:
            if position['status'] == 'open':
                # Check exit conditions
                hit_sl = (position['direction'] == 1 and self.current_price <= position['stop_loss']) or \
                        (position['direction'] == -1 and self.current_price >= position['stop_loss'])
                
                hit_tp = (position['direction'] == 1 and self.current_price >= position['take_profit']) or \
                        (position['direction'] == -1 and self.current_price <= position['take_profit'])
                
                if hit_sl or hit_tp:
                    # Close position
                    position['exit_price'] = self.current_price
                    position['exit_time'] = datetime.now()
                    position['status'] = 'closed'
                    
                    # Calculate P&L using proper contract specifications
                    lot_size = position.get('lot_size', 0.01)  # Default to minimum lot size if not found
                    position['pnl'] = self.calculate_pnl(
                        position['entry_price'], 
                        self.current_price, 
                        lot_size, 
                        position['direction']
                    )
                    
                    # Update capital with P&L
                    old_capital = self.capital
                    self.capital += position['pnl']
                    closed_positions.append(position)
                    
                    # Play sound and add effect
                    self.play_sound('trade_close')
                    self.add_particle_effect(self.current_price, position['direction'])
                    
                    exit_reason = 'TP' if hit_tp else 'SL'
                    print(f"💰 Position closed: {exit_reason}")
                    print(f"   P&L: ${position['pnl']:.2f}")
                    print(f"   Capital: ${old_capital:.2f} → ${self.capital:.2f}")
                    print(f"   Lot Size: {lot_size}")
        
        # Remove closed positions
        for pos in closed_positions:
            self.positions.remove(pos)
            self.trades_history.append(pos)
    
    def update_charts(self, frame):
        """Update all charts with enhanced visuals"""
        if len(self.ohlc_data) == 0:
            return
        
        # Update candlestick chart
        self.update_candlestick_chart()
        
        # Update HMM states chart
        self.update_states_chart()
        
        # Update equity curve
        self.update_equity_curve()
        
        # Update volume chart
        self.update_volume_chart()
        
        # Update particle effects
        self.update_particle_effects()
        
        # Update UI stats
        self.update_stats()
        self.update_positions_list()
        
        self.canvas.draw()
    
    def update_candlestick_chart(self):
        """Update the candlestick chart"""
        self.ax_price.clear()
        self.setup_price_chart()
        
        if len(self.ohlc_data) < 2:
            return
        
        # Draw candlesticks - show more bars for better visibility
        visible_bars = list(self.ohlc_data)[-100:]  # Show last 100 bars
        for i, bar in enumerate(visible_bars):
            x = i
            open_price = bar['open']
            high_price = bar['high']
            low_price = bar['low']
            close_price = bar['close']
            
            # Determine color
            color = self.colors['profit'] if close_price > open_price else self.colors['loss']
            
            # Draw high-low line (wick)
            self.ax_price.plot([x, x], [low_price, high_price], color=color, linewidth=1.5, alpha=0.8)
            
            # Draw body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            # Make body more visible
            rect = Rectangle((x-0.4, body_bottom), 0.8, body_height, 
                           facecolor=color, edgecolor=color, alpha=0.9, linewidth=1)
            self.ax_price.add_patch(rect)
        
        # Add trade markers and levels
        self.add_trade_markers()
        self.add_trade_levels()
        
        # Auto-scale with padding for SL/TP levels
        if len(visible_bars) > 0:
            # Get price range from visible bars
            all_prices = []
            for bar in visible_bars:
                all_prices.extend([bar['high'], bar['low'], bar['open'], bar['close']])
            
            # Add SL/TP levels to price range for proper scaling
            if self.positions:
                for pos in self.positions:
                    if pos['status'] == 'open':
                        all_prices.extend([pos['entry_price'], pos['stop_loss'], pos['take_profit']])
            
            if all_prices:
                min_price = min(all_prices)
                max_price = max(all_prices)
                price_range = max_price - min_price
                
                # Always use normal view with proper padding to show all levels
                if price_range > 0:
                    # Add 10% padding to ensure all levels are visible
                    padding = price_range * 0.1
                    self.ax_price.set_ylim(min_price - padding, max_price + padding)
                else:
                    # Fallback when price range is 0
                    center_price = (min_price + max_price) / 2
                    fallback_range = center_price * 0.01  # 1% range
                    self.ax_price.set_ylim(center_price - fallback_range, center_price + fallback_range)
        
        self.ax_price.relim()
        self.ax_price.autoscale_view(scalex=True, scaley=False)  # Don't auto-scale Y since we set it manually
    
    def add_trade_markers(self):
        """Add trade entry/exit markers to chart"""
        for trade in self.trades_history[-20:]:
            # Find approximate x position
            trade_time = trade['entry_time']
            x_pos = len(list(self.ohlc_data)) - 1  # Approximate position
            
            if trade['direction'] == 1:
                self.ax_price.scatter(x_pos, trade['entry_price'], color=self.colors['profit'], 
                                    s=100, marker='^', alpha=0.8, edgecolors='white')
            else:
                self.ax_price.scatter(x_pos, trade['entry_price'], color=self.colors['loss'], 
                                    s=100, marker='v', alpha=0.8, edgecolors='white')
    
    def add_trade_levels(self):
        """Add active trade levels (entry, SL, TP) to chart"""
        if not self.positions:
            return
        
        chart_width = len(list(self.ohlc_data)[-100:])  # Width of visible chart (updated to match candlesticks)
        
        for position in self.positions:
            if position['status'] == 'open':
                # Entry level (solid line) - more visible
                entry_color = self.colors['neon_blue']
                self.ax_price.axhline(y=position['entry_price'], color=entry_color, 
                                    linestyle='-', linewidth=4, alpha=1.0, zorder=10,
                                    label=f"Entry: {position['entry_price']:.2f}")
                
                # Stop Loss level (red dashed line) - more visible
                sl_color = self.colors['neon_red']
                self.ax_price.axhline(y=position['stop_loss'], color=sl_color, 
                                    linestyle=(0, (5, 5)), linewidth=4, alpha=1.0, zorder=10,
                                    label=f"SL: {position['stop_loss']:.2f}")
                
                # Take Profit level (green dotted line) - more visible
                tp_color = self.colors['neon_green']
                self.ax_price.axhline(y=position['take_profit'], color=tp_color, 
                                    linestyle=(0, (3, 3)), linewidth=4, alpha=1.0, zorder=10,
                                    label=f"TP: {position['take_profit']:.2f}")
                
                # Add visual separation lines to make levels more distinct
                
                # Add text labels on the right side - larger and more visible
                direction_text = "BUY" if position['direction'] == 1 else "SELL"
                
                # Entry label
                self.ax_price.text(chart_width * 0.85, position['entry_price'], 
                                 f"{direction_text} {position['entry_price']:.2f}", 
                                 color=entry_color, fontsize=11, fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.4", facecolor='black', alpha=0.9, edgecolor=entry_color))
                
                # SL label
                self.ax_price.text(chart_width * 0.85, position['stop_loss'], 
                                 f"SL {position['stop_loss']:.2f}", 
                                 color=sl_color, fontsize=10, fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.9, edgecolor=sl_color))
                
                # TP label
                self.ax_price.text(chart_width * 0.85, position['take_profit'], 
                                 f"TP {position['take_profit']:.2f}", 
                                 color=tp_color, fontsize=10, fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.9, edgecolor=tp_color))
    
    def update_states_chart(self):
        """Update the HMM states chart"""
        if len(self.hmm_states) == 0:
            return
        
        self.ax_states.clear()
        self.setup_states_chart()
        
        # Plot states as colored bars
        states = list(self.hmm_states)[-50:]
        colors = [self.colors['neon_green'], self.colors['neon_yellow'], self.colors['neon_red']]
        
        for i, state in enumerate(states):
            self.ax_states.bar(i, 1, bottom=state, color=colors[state], alpha=0.7, width=0.8)
        
        self.ax_states.set_xlim(0, len(states))
    
    def update_equity_curve(self):
        """Update the equity curve"""
        # Always show equity curve, even with no trades
        if not self.trades_history:
            # Show starting capital as flat line
            equity_data = [self.initial_capital, self.capital]
            x_data = [0, 1]
        else:
            # Build equity curve from trade history
            equity_data = [self.initial_capital]
            for trade in self.trades_history:
                if 'pnl' in trade:
                    equity_data.append(equity_data[-1] + trade['pnl'])
            
            # Add current capital if different (for open positions)
            if len(equity_data) > 0 and equity_data[-1] != self.capital:
                equity_data.append(self.capital)
            
            x_data = range(len(equity_data))
        
        # Update the line
        self.equity_line.set_data(x_data, equity_data)
        
        # Update axis limits
        if len(equity_data) > 0:
            self.ax_equity.set_xlim(0, max(1, len(equity_data)-1))
            y_min = min(equity_data) * 0.95
            y_max = max(equity_data) * 1.05
            self.ax_equity.set_ylim(y_min, y_max)
        
        self.ax_equity.relim()
        self.ax_equity.autoscale_view()
    
    def update_volume_chart(self):
        """Update the volume/activity chart"""
        self.ax_volume.clear()
        self.setup_volume_chart()
        
        if len(self.ohlc_data) < 2:
            return
        
        # Plot volume bars
        volumes = [bar['volume'] for bar in list(self.ohlc_data)[-50:]]
        x_data = range(len(volumes))
        
        self.ax_volume.bar(x_data, volumes, color=self.colors['neon_blue'], alpha=0.6)
    
    def update_particle_effects(self):
        """Update particle effects"""
        # Remove expired effects
        self.particle_effects = [effect for effect in self.particle_effects if effect['life'] > 0]
        
        # Update remaining effects
        for effect in self.particle_effects:
            effect['life'] -= 1
            # Add visual effect to chart (simplified)
    
    def update_stats(self):
        """Update statistics with enhanced styling"""
        # Capital with color coding
        self.capital_label.config(text=f"${self.capital:,.2f}")
        if self.capital > self.initial_capital:
            self.capital_label.config(style='Profit.TLabel')
        elif self.capital < self.initial_capital:
            self.capital_label.config(style='Loss.TLabel')
        else:
            self.capital_label.config(style='Neon.TLabel')
        
        # P&L with color coding - Show current open positions P&L
        if self.positions:
            # Calculate total unrealized P&L of open positions
            total_unrealized_pnl = 0
            for pos in self.positions:
                if pos['status'] == 'open':
                    lot_size = pos.get('lot_size', 0.01)
                    unrealized_pnl = self.calculate_pnl(
                        pos['entry_price'], 
                        self.current_price, 
                        lot_size, 
                        pos['direction']
                    )
                    total_unrealized_pnl += unrealized_pnl
                    
                    # Calculate unrealized P&L for display
            
            self.pnl_label.config(text=f"${total_unrealized_pnl:+,.2f}")
            if total_unrealized_pnl > 0:
                self.pnl_label.config(style='Profit.TLabel')
            elif total_unrealized_pnl < 0:
                self.pnl_label.config(style='Loss.TLabel')
            else:
                self.pnl_label.config(style='Neon.TLabel')
        else:
            # No open positions - show 0
            self.pnl_label.config(text="$0.00", style='Neon.TLabel')
        
        # Current price
        self.price_label.config(text=f"${self.current_price:,.2f}")
        
        # Win rate
        if self.trades_history:
            wins = sum(1 for trade in self.trades_history if trade['pnl'] > 0)
            win_rate = wins / len(self.trades_history) * 100
            self.winrate_label.config(text=f"{win_rate:.1f}%")
        
        # Trade count
        self.trades_label.config(text=str(len(self.trades_history)))
        
        # Active positions
        self.positions_label.config(text=str(len(self.positions)))
        
        # Total realized P&L
        total_realized_pnl = self.capital - self.initial_capital
        self.total_pnl_label.config(text=f"${total_realized_pnl:+,.2f}")
        if total_realized_pnl > 0:
            self.total_pnl_label.config(style='Profit.TLabel')
        elif total_realized_pnl < 0:
            self.total_pnl_label.config(style='Loss.TLabel')
        else:
            self.total_pnl_label.config(style='Neon.TLabel')
        
        # HMM State
        if self.hmm_states:
            current_state = self.hmm_states[-1]
            state_names = ['BEARISH', 'NEUTRAL', 'BULLISH']
            self.state_label.config(text=state_names[current_state])
        
        # Progress and Status
        if self.is_simulating and self.historical_data is not None:
            progress = (self.current_bar_index / len(self.historical_data)) * 100
            self.progress_label.config(text=f"{progress:.1f}%")
            self.status_label.config(text="SIMULATING")
        elif self.is_running:
            self.progress_label.config(text="LIVE")
            self.status_label.config(text="TRADING")
        else:
            self.progress_label.config(text="0%")
            self.status_label.config(text="READY")
    
    def update_positions_list(self):
        """Update positions list with enhanced formatting"""
        self.positions_listbox.delete(0, tk.END)
        
        for pos in self.positions:
            direction = "BUY" if pos['direction'] == 1 else "SELL"
            
            # Calculate unrealized P&L using proper contract specifications
            lot_size = pos.get('lot_size', 0.01)  # Default to minimum lot size if not found
            unrealized_pnl = self.calculate_pnl(
                pos['entry_price'], 
                self.current_price, 
                lot_size, 
                pos['direction']
            )
            
            status_icon = "🟢" if unrealized_pnl > 0 else "🔴"
            
            # More detailed position info
            position_text = f"{status_icon} {direction} @ {pos['entry_price']:.5f}"
            position_text += f" | P&L: ${unrealized_pnl:+.2f}"
            position_text += f" | Lot: {lot_size}"
            position_text += f" | SL: {pos['stop_loss']:.5f}"
            position_text += f" | TP: {pos['take_profit']:.5f}"
            
            self.positions_listbox.insert(tk.END, position_text)
    
    def load_historical_data(self):
        """Load historical data from last week in 1-minute timeframe"""
        if not self.mt5_provider:
            print("❌ No MT5 connection, using simulated data")
            return False
        
        try:
            # Get selected timeframe
            timeframe_str = self.timeframe_var.get()
            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1
            }
            mt5_timeframe = timeframe_map.get(timeframe_str, mt5.TIMEFRAME_M1)
            
            # Calculate date range and bars based on timeframe
            end_date = datetime.now()
            if timeframe_str == "M1":
                start_date = end_date - timedelta(days=2)  # 2 days for M1
                max_bars = 2 * 24 * 60  # 2 days * 24 hours * 60 minutes
            elif timeframe_str == "M5":
                start_date = end_date - timedelta(days=7)  # 1 week for M5
                max_bars = 7 * 24 * 12  # 7 days * 24 hours * 12 (5-min bars per hour)
            elif timeframe_str == "M15":
                start_date = end_date - timedelta(days=14)  # 2 weeks for M15
                max_bars = 14 * 24 * 4  # 2 weeks * 24 hours * 4 (15-min bars per hour)
            elif timeframe_str == "M30":
                start_date = end_date - timedelta(days=30)  # 1 month for M30
                max_bars = 30 * 24 * 2  # 1 month * 24 hours * 2 (30-min bars per hour)
            else:  # H1
                start_date = end_date - timedelta(days=60)  # 2 months for H1
                max_bars = 60 * 24  # 2 months * 24 hours
            
            print(f"📈 Loading historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"   Symbol: {self.trading_symbol}")
            print(f"   Timeframe: {timeframe_str}")
            
            # Fetch data for selected timeframe
            self.historical_data = self.mt5_provider.fetch_data(
                symbol=self.trading_symbol,
                timeframe=mt5_timeframe,
                bars=max_bars,
                from_date=start_date
            )
            
            if self.historical_data is not None and len(self.historical_data) > 0:
                print(f"✅ Loaded {len(self.historical_data)} bars of historical data")
                print(f"   Date range: {self.historical_data.index[0]} to {self.historical_data.index[-1]}")
                print(f"   Price range: {self.historical_data['Close'].min():.5f} - {self.historical_data['Close'].max():.5f}")
                
                # Reset simulation state
                self.current_bar_index = 0
                self.ohlc_data.clear()
                self.time_data.clear()
                self.hmm_states.clear()
                self.state_probabilities.clear()
                
                # Initialize with first few bars from real historical data
                initial_bars = min(100, len(self.historical_data))  # Start with 100 bars for context
                print(f"📊 Initializing chart with first {initial_bars} bars of real data...")
                
                for i in range(initial_bars):
                    row = self.historical_data.iloc[i]
                    ohlc = {
                        'time': row.name,
                        'open': row['Open'],
                        'high': row['High'],
                        'low': row['Low'],
                        'close': row['Close'],
                        'volume': row['Volume']
                    }
                    self.ohlc_data.append(ohlc)
                    self.time_data.append(row.name)
                    
                    # Set current price to the latest bar
                    self.current_price = row['Close']
                    
                    # Simulate HMM state
                    state = np.random.randint(0, 3)
                    self.hmm_states.append(state)
                    self.state_probabilities.append([np.random.random(), np.random.random(), np.random.random()])
                
                # Start simulation from bar 100
                self.current_bar_index = initial_bars
                print(f"✅ Chart initialized with real {self.trading_symbol} data")
                print(f"   Starting simulation from bar {self.current_bar_index}")
                print(f"   Current price: ${self.current_price:.2f}")
                return True
            else:
                print("❌ Failed to load historical data")
                return False
                
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return False
    
    def toggle_trading(self):
        """Toggle trading with enhanced button effects"""
        if not self.is_running:
            # Starting trading - load historical data
            print("🚀 Starting historical simulation...")
            if self.load_historical_data():
                self.is_running = True
                self.is_simulating = True
                self.start_button.config(text="⏸️ PAUSE", bg=self.colors['neon_red'])
                print("✅ Historical simulation started!")
            else:
                print("❌ Failed to start historical simulation")
        else:
            # Pausing trading
            self.is_running = False
            self.is_simulating = False
            self.start_button.config(text="🚀 START", bg=self.colors['neon_green'])
    
    def on_symbol_change(self, event=None):
        """Handle trading symbol change"""
        new_symbol = self.symbol_var.get()
        if new_symbol != self.trading_symbol:
            self.trading_symbol = new_symbol
            print(f"📊 Trading symbol changed to: {self.trading_symbol}")
            
            # Reset simulation if running
            if self.is_running:
                self.reset_simulation()
    
    def on_timeframe_change(self, event=None):
        """Handle timeframe change"""
        new_timeframe = self.timeframe_var.get()
        print(f"⏰ Timeframe changed to: {new_timeframe}")
        
        # Reset simulation if running to load new timeframe data
        if self.is_running:
            self.reset_simulation()
    
    def on_balance_change(self, event=None):
        """Handle account balance change (deprecated - use save button)"""
        # This function is now mainly for display purposes
        pass
    
    def save_settings(self):
        """Save balance and risk settings"""
        try:
            # Validate and save balance
            new_balance = float(self.balance_var.get())
            if new_balance <= 0:
                self.settings_status_label.config(text="❌ Balance must be positive", foreground=self.colors['neon_red'])
                self.balance_var.set(str(self.custom_balance))
                return
            
            # Validate and save risk
            new_risk = float(self.risk_var.get())
            if not (0.1 <= new_risk <= 10.0):
                self.settings_status_label.config(text="❌ Risk must be 0.1% - 10%", foreground=self.colors['neon_red'])
                self.risk_var.set(str(self.risk_per_trade))
                return
            
            # Apply changes
            old_balance = self.custom_balance
            old_risk = self.risk_per_trade
            
            self.custom_balance = new_balance
            self.capital = new_balance
            self.initial_capital = new_balance
            self.risk_per_trade = new_risk
            
            # Update displays
            self.current_balance_label.config(text=f"${self.custom_balance:.2f}")
            self.current_risk_label.config(text=f"{self.risk_per_trade}%")
            self.update_stats()
            self.root.update_idletasks()
            
            # Show success message
            self.settings_status_label.config(text="✅ Settings Saved!", foreground=self.colors['neon_green'])
            
            print(f"💾 Settings Saved:")
            print(f"   Balance: ${old_balance:.2f} → ${new_balance:.2f}")
            print(f"   Risk: {old_risk}% → {new_risk}%")
            
            # Reset simulation if running
            if self.is_running:
                print("🔄 Resetting simulation with new settings...")
                self.reset_simulation()
            
            # Clear status message after 3 seconds
            self.root.after(3000, lambda: self.settings_status_label.config(text=""))
                
        except ValueError:
            self.settings_status_label.config(text="❌ Invalid number format", foreground=self.colors['neon_red'])
            self.balance_var.set(str(self.custom_balance))
            self.risk_var.set(str(self.risk_per_trade))
    
    def on_risk_change(self, event=None):
        """Handle risk per trade change (deprecated - use save button)"""
        # This function is now mainly for display purposes
        pass
    
    def toggle_auto_strategy(self):
        """Toggle automatic strategy mode"""
        # Implementation for auto strategy
        print("🧠 Auto strategy toggled")
    
    def reset_simulation(self):
        """Reset simulation with effects"""
        self.is_running = False
        self.is_simulating = False
        self.positions.clear()
        self.trades_history.clear()
        self.ohlc_data.clear()
        self.time_data.clear()
        self.hmm_states.clear()
        self.state_probabilities.clear()
        self.particle_effects.clear()
        
        # Reset simulation data
        self.historical_data = None
        self.current_bar_index = 0
        
        self.capital = self.custom_balance
        self.initial_capital = self.custom_balance
        self.current_price = 107000
        
        self.start_button.config(text="🚀 START", bg=self.colors['neon_green'])
        print("🔄 Simulation reset - Ready for new historical data")
    
    def get_contract_spec(self, symbol=None):
        """Get contract specifications for the current or specified symbol"""
        if symbol is None:
            symbol = self.trading_symbol
        return self.contract_specs.get(symbol, self.contract_specs["DEFAULT"])
    
    def calculate_lot_size(self, risk_amount, stop_loss_distance, entry_price):
        """Calculate reasonable lot size based on risk amount and stop loss distance"""
        if stop_loss_distance <= 0:
            return 0.01  # Minimum lot size
        
        # Use a simple, consistent formula for all symbols
        # Lot size = Risk Amount / Stop Loss Distance
        # This ensures the maximum loss equals the risk amount
        lot_size = risk_amount / stop_loss_distance
        
        # Apply reasonable scaling based on symbol type
        if self.trading_symbol in ['BTCUSD', 'BTCUSDm']:
            # For Bitcoin: scale down significantly due to high price
            lot_size = lot_size / 10000  # Much smaller lots for BTC
        elif self.trading_symbol in ['ETHUSD', 'ETHUSDm']:
            # For Ethereum: moderate scaling
            lot_size = lot_size / 1000
        elif self.trading_symbol in ['XAUUSDm']:
            # For Gold: light scaling
            lot_size = lot_size / 100
        else:
            # For forex: minimal scaling
            lot_size = lot_size / 10
        
        # Ensure reasonable lot size range
        lot_size = max(0.001, min(0.1, lot_size))  # Cap at 0.1 lots max
        
        # Round to 3 decimal places for lot size
        lot_size = round(lot_size, 3)
        
        return lot_size
    
    def calculate_pnl(self, entry_price, exit_price, lot_size, direction):
        """Calculate P&L using consistent and realistic calculation"""
        # Simple P&L calculation: (Exit Price - Entry Price) * Direction * Lot Size
        price_diff = (exit_price - entry_price) * direction
        
        # Use consistent scaling that matches lot size calculation
        if self.trading_symbol in ['BTCUSD', 'BTCUSDm']:
            # For Bitcoin: use the same scaling as lot size calculation
            pnl = price_diff * lot_size * 10000  # Reverse the lot size scaling
        elif self.trading_symbol in ['ETHUSD', 'ETHUSDm']:
            # For Ethereum: reverse the scaling
            pnl = price_diff * lot_size * 1000
        elif self.trading_symbol in ['XAUUSDm']:
            # For Gold: reverse the scaling
            pnl = price_diff * lot_size * 100
        else:
            # For forex: reverse the scaling
            pnl = price_diff * lot_size * 10
        
        # Cap P&L to prevent unrealistic numbers (max 20% of capital per trade)
        max_pnl = self.capital * 0.2
        pnl = max(-max_pnl, min(max_pnl, pnl))
        
        return pnl
    
    def price_to_pips(self, price_distance):
        """Convert price distance to pips"""
        spec = self.get_contract_spec()
        return abs(price_distance) / spec["pip_size"]
    
    def cleanup(self):
        """Cleanup resources"""
        if self.mt5_provider:
            self.mt5_provider.disconnect()
    
    def run(self):
        """Run the enhanced GUI"""
        try:
            self.root.mainloop()
        finally:
            self.cleanup()

def main():
    """Main function"""
    print("🎮 Starting Enhanced Futuristic Trading Simulator...")
    gui = EnhancedTradingGUI()
    gui.run()

if __name__ == "__main__":
    main() 