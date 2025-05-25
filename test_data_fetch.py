#!/usr/bin/env python3
"""
Test script to verify data fetching works
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def test_symbol(symbol, period='1mo', interval='5m'):
    """Test if a symbol works"""
    try:
        print(f"Testing {symbol}...")
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if not data.empty:
            print(f"✅ {symbol}: {len(data)} records fetched")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            print(f"   Latest price: {data['Close'].iloc[-1]:.4f}")
            return True
        else:
            print(f"❌ {symbol}: No data returned")
            return False
    except Exception as e:
        print(f"❌ {symbol}: Error - {e}")
        return False

def main():
    """Test various symbols"""
    print("🔍 Testing Data Sources")
    print("=" * 40)
    
    # Test forex symbols
    forex_symbols = [
        'EUR=X', 'EURUSD=X', 'EURUSD',
        'GBP=X', 'GBPUSD=X', 'GBPUSD', 
        'JPY=X', 'USDJPY=X', 'USDJPY',
        'CHF=X', 'USDCHF=X', 'USDCHF'
    ]
    
    print("\n📈 Testing Forex Symbols:")
    working_forex = []
    for symbol in forex_symbols:
        if test_symbol(symbol):
            working_forex.append(symbol)
    
    # Test major stocks/ETFs as fallback
    print("\n📊 Testing Stock/ETF Symbols:")
    stock_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
    working_stocks = []
    for symbol in stock_symbols:
        if test_symbol(symbol):
            working_stocks.append(symbol)
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 SUMMARY")
    print("=" * 40)
    print(f"Working Forex symbols: {working_forex}")
    print(f"Working Stock symbols: {working_stocks}")
    
    if working_forex:
        recommended = working_forex[0]
        print(f"\n✅ Recommended forex symbol: {recommended}")
    elif working_stocks:
        recommended = working_stocks[0]
        print(f"\n✅ Recommended fallback symbol: {recommended}")
    else:
        print("\n❌ No working symbols found. Check internet connection.")
        return None
    
    return recommended

if __name__ == "__main__":
    recommended_symbol = main()
    
    if recommended_symbol:
        print(f"\n🚀 You can now run the strategy with: {recommended_symbol}")
        print("Example:")
        print(f"python run_strategy.py --mode basic --symbol {recommended_symbol}") 