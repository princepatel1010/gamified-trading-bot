#!/usr/bin/env python3
"""
MetaTrader 5 Data Provider for Forex HMM Strategy
=================================================

This module provides real forex data from MetaTrader 5 terminal.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

class MT5DataProvider:
    def __init__(self, login=204724510, password="Prince$007", 
                 server="Exness-MT5Trial7", 
                 path="C:\\Program Files\\MetaTrader 5 EXNESS\\terminal64.exe"):
        """
        Initialize MT5 connection
        
        Parameters:
        login: MT5 account login
        password: MT5 account password
        server: MT5 server name
        path: Path to MT5 terminal executable
        """
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        
    def connect(self):
        """Connect to MT5 terminal"""
        try:
            # Initialize MT5 connection
            if not mt5.initialize(path=self.path):
                print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
                return False
            
            # Login to account
            if not mt5.login(self.login, password=self.password, server=self.server):
                print(f"❌ Failed to login to MT5: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                print("❌ Failed to get account info")
                mt5.shutdown()
                return False
            
            print(f"✅ Connected to MT5")
            print(f"   Account: {account_info.login}")
            print(f"   Server: {account_info.server}")
            print(f"   Balance: ${account_info.balance:.2f}")
            print(f"   Currency: {account_info.currency}")
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("🔌 Disconnected from MT5")
    
    def get_available_symbols(self):
        """Get list of available symbols"""
        if not self.connected:
            print("❌ Not connected to MT5")
            return {'forex': [], 'crypto': [], 'all': []}
        
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                print("❌ Failed to get symbols")
                return {'forex': [], 'crypto': [], 'all': []}
            
            # Get all visible symbols
            all_symbols = []
            forex_symbols = []
            crypto_symbols = []
            
            for symbol in symbols:
                if symbol.visible:
                    all_symbols.append(symbol.name)
                    # Categorize symbols
                    if 'USD' in symbol.name and len(symbol.name) <= 8:
                        if any(curr in symbol.name for curr in ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']):
                            forex_symbols.append(symbol.name)
                        elif any(crypto in symbol.name for crypto in ['BTC', 'ETH', 'LTC']):
                            crypto_symbols.append(symbol.name)
            
            print(f"📊 Symbol Categories:")
            print(f"   All symbols: {len(all_symbols)}")
            print(f"   Forex-like: {len(forex_symbols)}")
            print(f"   Crypto: {len(crypto_symbols)}")
            
            # Return forex first, then crypto, then all
            return {
                'forex': sorted(forex_symbols),
                'crypto': sorted(crypto_symbols), 
                'all': sorted(all_symbols)
            }
            
        except Exception as e:
            print(f"❌ Error getting symbols: {e}")
            return {'forex': [], 'crypto': [], 'all': []}
    
    def fetch_data(self, symbol='EURUSD', timeframe=mt5.TIMEFRAME_M5, 
                   bars=5000, from_date=None):
        """
        Fetch historical data from MT5
        
        Parameters:
        symbol: Currency pair symbol (e.g., 'EURUSD', 'GBPUSD')
        timeframe: MT5 timeframe constant
        bars: Number of bars to fetch
        from_date: Start date (if None, fetches last N bars)
        
        Returns:
        pandas.DataFrame with OHLCV data
        """
        if not self.connected:
            print("❌ Not connected to MT5")
            return None
        
        try:
            print(f"📊 Fetching {symbol} data...")
            
            # Check if symbol exists
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Symbol {symbol} not found")
                available = self.get_available_symbols()
                if available['all']:
                    print(f"Available symbols: {available['all'][:10]}...")  # Show first 10
                return None
            
            # Select the symbol in Market Watch
            if not mt5.symbol_select(symbol, True):
                print(f"❌ Failed to select symbol {symbol}")
                return None
            
            # Get rates
            if from_date:
                rates = mt5.copy_rates_from(symbol, timeframe, from_date, bars)
            else:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                print(f"❌ No data received for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns to match yfinance format
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            # Add timezone info (MT5 data is usually in broker timezone)
            df.index = df.index.tz_localize('UTC')
            
            print(f"✅ Fetched {len(df)} bars for {symbol}")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Latest price: {df['Close'].iloc[-1]:.5f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def get_symbol_info(self, symbol):
        """Get detailed symbol information"""
        if not self.connected:
            return None
        
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            
            return {
                'name': info.name,
                'description': info.description,
                'point': info.point,
                'digits': info.digits,
                'spread': info.spread,
                'trade_mode': info.trade_mode,
                'min_lot': info.volume_min,
                'max_lot': info.volume_max,
                'lot_step': info.volume_step,
                'currency_base': info.currency_base,
                'currency_profit': info.currency_profit,
                'currency_margin': info.currency_margin
            }
            
        except Exception as e:
            print(f"❌ Error getting symbol info: {e}")
            return None

def test_mt5_connection():
    """Test MT5 connection and data fetching"""
    print("🔍 Testing MT5 Connection")
    print("=" * 40)
    
    # Initialize provider
    provider = MT5DataProvider()
    
    # Connect
    if not provider.connect():
        print("❌ Failed to connect to MT5")
        return None
    
    try:
        # Get available symbols
        print("\n📈 Available Symbols:")
        symbols_dict = provider.get_available_symbols()
        
        # Test with available symbols
        test_symbols = []
        if symbols_dict['crypto']:
            test_symbols.extend(symbols_dict['crypto'][:2])  # Try first 2 crypto
        if symbols_dict['all']:
            test_symbols.extend(symbols_dict['all'][:5])  # Try first 5 any symbols
        
        working_symbols = []
        
        print("\n📊 Testing Data Fetching:")
        for symbol in test_symbols:
            data = provider.fetch_data(symbol, bars=100)
            if data is not None and len(data) > 0:
                working_symbols.append(symbol)
                print(f"✅ {symbol}: {len(data)} bars")
            else:
                print(f"❌ {symbol}: No data")
        
        print(f"\n✅ Working symbols: {working_symbols}")
        
        # Test with 5-minute data for strategy
        if working_symbols:
            test_symbol = working_symbols[0]
            print(f"\n🎯 Testing 5-minute data for {test_symbol}:")
            data_5m = provider.fetch_data(test_symbol, mt5.TIMEFRAME_M5, bars=1000)
            if data_5m is not None:
                print(f"✅ 5-minute data: {len(data_5m)} bars")
                print(f"   Sample data:")
                print(data_5m.tail())
                return test_symbol, provider
        
    finally:
        provider.disconnect()
    
    return None

def main():
    """Main test function"""
    result = test_mt5_connection()
    
    if result:
        symbol, provider = result
        print(f"\n🚀 MT5 connection successful!")
        print(f"Recommended symbol for strategy: {symbol}")
        print("\nYou can now run the forex strategy with real MT5 data!")
    else:
        print("\n❌ MT5 connection failed. Please check:")
        print("1. MT5 terminal is installed and running")
        print("2. Account credentials are correct")
        print("3. Internet connection is stable")

if __name__ == "__main__":
    main() 