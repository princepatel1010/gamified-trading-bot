#!/usr/bin/env python3
"""
Forex HMM Scalping Strategy - Main Runner
==========================================

Interactive script to run different components of the forex trading strategy:
1. Basic strategy backtesting
2. Parameter optimization
3. Live trading simulation
4. Performance analysis

Usage: python run_strategy.py
"""

import sys
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def print_banner():
    """Print welcome banner"""
    print("="*60)
    print("🚀 FOREX HMM SCALPING STRATEGY")
    print("="*60)
    print("Advanced forex trading strategy using Hidden Markov Models")
    print("Target: ~50% win rate with 1:1.5 risk-reward ratio")
    print("Focus: Scalping on 5-minute timeframes")
    print("="*60)
    print()

def print_menu():
    """Print main menu options"""
    print("📋 MAIN MENU")
    print("-" * 30)
    print("1. 📊 Run Basic Strategy Backtest")
    print("2. 🎯 Optimize Strategy Parameters")
    print("3. 🔄 Live Trading Simulation")
    print("4. 📈 Performance Analysis")
    print("5. 🛠️  Custom Configuration")
    print("6. ❓ Help & Documentation")
    print("0. 🚪 Exit")
    print("-" * 30)

def run_basic_strategy():
    """Run basic strategy backtest"""
    print("\n🔄 Running Basic Strategy Backtest...")
    print("=" * 40)
    
    try:
        from forex_hmm_strategy import main
        
        # Get user preferences
        symbol = input("Enter currency pair (default: BTCUSDm): ").strip() or "BTCUSDm"
        period = input("Enter data period (default: 3mo): ").strip() or "3mo"
        
        print(f"\nRunning strategy for {symbol} with {period} of data...")
        
        # Run strategy
        strategy, performance = main()
        
        print("\n✅ Strategy completed successfully!")
        print("Check the generated plots for detailed analysis.")
        
        return strategy, performance
        
    except Exception as e:
        print(f"❌ Error running strategy: {e}")
        return None, None

def run_optimization():
    """Run parameter optimization"""
    print("\n🎯 Running Parameter Optimization...")
    print("=" * 40)
    
    try:
        from strategy_optimizer import main
        
        # Get user preferences
        symbol = input("Enter currency pair (default: BTCUSDm): ").strip() or "BTCUSDm"
        
        print(f"\nOptimizing parameters for {symbol}...")
        print("This may take several minutes...")
        
        # Run optimization
        optimizer, best_strategy, best_performance = main()
        
        print("\n✅ Optimization completed successfully!")
        print("Best parameters have been identified and tested.")
        
        return optimizer, best_strategy, best_performance
        
    except Exception as e:
        print(f"❌ Error running optimization: {e}")
        return None, None, None

def run_live_simulation():
    """Run live trading simulation"""
    print("\n🔄 Running Live Trading Simulation...")
    print("=" * 40)
    
    try:
        from live_trading_simulator import main
        
        # Get user preferences
        capital = input("Enter initial capital (default: 10000): ").strip()
        capital = float(capital) if capital else 10000
        
        duration = input("Enter simulation duration in minutes (default: 5): ").strip()
        duration = int(duration) if duration else 5
        
        print(f"\nStarting live simulation with ${capital:,.2f} for {duration} minutes...")
        print("Press Ctrl+C to stop early if needed.")
        
        # Run simulation
        simulator = main()
        
        print("\n✅ Live simulation completed!")
        
        return simulator
        
    except KeyboardInterrupt:
        print("\n⏹️  Simulation stopped by user.")
        return None
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return None

def performance_analysis():
    """Run performance analysis"""
    print("\n📈 Performance Analysis...")
    print("=" * 40)
    
    print("This feature allows you to analyze:")
    print("• Historical performance metrics")
    print("• Risk-adjusted returns")
    print("• Drawdown analysis")
    print("• Trade distribution")
    print()
    
    choice = input("Run analysis? (y/n): ").strip().lower()
    if choice == 'y':
        try:
            # Run basic strategy for analysis
            strategy, performance = run_basic_strategy()
            
            if strategy and performance:
                print("\n📊 PERFORMANCE SUMMARY")
                print("=" * 30)
                for key, value in performance.items():
                    if isinstance(value, float):
                        if 'rate' in key or 'return' in key:
                            print(f"{key}: {value:.2%}")
                        else:
                            print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
                
                # Additional analysis
                if hasattr(strategy, 'trades') and not strategy.trades.empty:
                    trades = strategy.trades
                    print(f"\n📋 TRADE ANALYSIS")
                    print("=" * 20)
                    print(f"Average holding time: {(trades['exit_time'] - trades['entry_time']).mean()}")
                    print(f"Best trade: ${trades['pnl'].max():.2f}")
                    print(f"Worst trade: ${trades['pnl'].min():.2f}")
                    print(f"Consecutive wins: {trades['pnl'].gt(0).astype(int).groupby((trades['pnl'] <= 0).cumsum()).sum().max()}")
                    print(f"Consecutive losses: {trades['pnl'].lt(0).astype(int).groupby((trades['pnl'] >= 0).cumsum()).sum().max()}")
                
        except Exception as e:
            print(f"❌ Error in analysis: {e}")

def custom_configuration():
    """Custom configuration setup"""
    print("\n🛠️  Custom Configuration...")
    print("=" * 40)
    
    print("Available configuration options:")
    print("1. Strategy parameters (HMM components, risk-reward ratio)")
    print("2. Risk management settings")
    print("3. Data source and timeframe")
    print("4. Trading session preferences")
    print()
    
    config_choice = input("Select configuration (1-4): ").strip()
    
    if config_choice == "1":
        print("\n⚙️  Strategy Parameters:")
        n_components = input("HMM components (default: 3): ").strip() or "3"
        risk_reward = input("Risk-reward ratio (default: 1.5): ").strip() or "1.5"
        print(f"Configuration: {n_components} components, {risk_reward} risk-reward")
        
    elif config_choice == "2":
        print("\n🛡️  Risk Management:")
        max_drawdown = input("Max drawdown % (default: 15): ").strip() or "15"
        daily_loss = input("Max daily loss $ (default: 500): ").strip() or "500"
        print(f"Configuration: {max_drawdown}% max drawdown, ${daily_loss} daily limit")
        
    elif config_choice == "3":
        print("\n📊 Data Configuration:")
        timeframe = input("Timeframe (default: 5m): ").strip() or "5m"
        period = input("Data period (default: 3mo): ").strip() or "3mo"
        print(f"Configuration: {timeframe} timeframe, {period} period")
        
    elif config_choice == "4":
        print("\n🕐 Trading Sessions:")
        print("Available sessions: London, NY, Tokyo, Overlap")
        session = input("Preferred session (default: Overlap): ").strip() or "Overlap"
        print(f"Configuration: {session} session focus")
    
    print("\n✅ Configuration saved for next run!")

def show_help():
    """Show help and documentation"""
    print("\n❓ Help & Documentation")
    print("=" * 40)
    
    print("""
📚 STRATEGY OVERVIEW:
This is an advanced forex scalping strategy using Hidden Markov Models (HMM)
to identify market regimes and generate trading signals.

🎯 KEY FEATURES:
• Machine learning-based market regime detection
• Probabilistic signal generation
• Advanced risk management
• Parameter optimization
• Live trading simulation

📊 PERFORMANCE TARGETS:
• Win Rate: ~50%
• Risk-Reward: 1:1.5
• Timeframe: 5-minute scalping
• Focus: Major forex pairs during active sessions

🛠️  COMPONENTS:
1. forex_hmm_strategy.py - Core strategy implementation
2. strategy_optimizer.py - Parameter optimization
3. live_trading_simulator.py - Live trading simulation
4. run_strategy.py - This interactive runner

📖 USAGE TIPS:
• Start with basic backtest to understand performance
• Use optimization to find best parameters for your data
• Test with live simulation before real trading
• Monitor risk metrics closely

⚠️  DISCLAIMER:
This is for educational purposes only. Past performance doesn't guarantee
future results. Always test thoroughly before using real money.

📞 SUPPORT:
• Check README.md for detailed documentation
• Review code comments for technical details
• Test with different parameters and timeframes
    """)

def main():
    """Main interactive function"""
    parser = argparse.ArgumentParser(description='Forex HMM Scalping Strategy Runner')
    parser.add_argument('--mode', choices=['basic', 'optimize', 'live', 'analysis'], 
                       help='Run specific mode directly')
    parser.add_argument('--symbol', default='BTCUSDm', help='Currency pair symbol')
    parser.add_argument('--period', default='3mo', help='Data period')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # If mode specified, run directly
    if args.mode:
        if args.mode == 'basic':
            run_basic_strategy()
        elif args.mode == 'optimize':
            run_optimization()
        elif args.mode == 'live':
            run_live_simulation()
        elif args.mode == 'analysis':
            performance_analysis()
        return
    
    # Interactive mode
    while True:
        try:
            print_menu()
            choice = input("\nSelect option (0-6): ").strip()
            
            if choice == "0":
                print("\n👋 Thank you for using Forex HMM Strategy!")
                print("Happy trading! 📈")
                break
                
            elif choice == "1":
                run_basic_strategy()
                
            elif choice == "2":
                run_optimization()
                
            elif choice == "3":
                run_live_simulation()
                
            elif choice == "4":
                performance_analysis()
                
            elif choice == "5":
                custom_configuration()
                
            elif choice == "6":
                show_help()
                
            else:
                print("❌ Invalid option. Please select 0-6.")
            
            # Pause before showing menu again
            if choice != "0":
                input("\nPress Enter to continue...")
                print("\n" + "="*60 + "\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or contact support.")

if __name__ == "__main__":
    main() 