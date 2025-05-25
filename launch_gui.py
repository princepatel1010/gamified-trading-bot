#!/usr/bin/env python3
"""
GUI Launcher for Futuristic Trading Simulator
=============================================

Choose between different GUI versions:
1. Basic Futuristic GUI
2. Enhanced Gaming GUI with Candlesticks
"""

import sys
import os

def show_menu():
    """Display the GUI selection menu"""
    print("🎮" + "="*60 + "🎮")
    print("    FUTURISTIC FOREX TRADING SIMULATOR")
    print("           GUI LAUNCHER")
    print("🎮" + "="*60 + "🎮")
    print()
    print("Choose your trading interface:")
    print()
    print("1. 🚀 Basic Futuristic GUI")
    print("   - Real-time line charts")
    print("   - Live P&L tracking")
    print("   - Gaming-style interface")
    print("   - Simple and fast")
    print()
    print("2. 🎮 Enhanced Gaming GUI")
    print("   - Real-time candlestick charts")
    print("   - HMM state visualization")
    print("   - Multiple chart panels")
    print("   - Advanced gaming effects")
    print("   - Particle effects")
    print()
    print("3. 📊 Command Line Interface")
    print("   - Text-based interface")
    print("   - Full strategy features")
    print("   - No graphics required")
    print()
    print("0. ❌ Exit")
    print()

def launch_basic_gui():
    """Launch the basic futuristic GUI"""
    print("🚀 Launching Basic Futuristic GUI...")
    try:
        from futuristic_trading_gui import main
        main()
    except ImportError as e:
        print(f"❌ Error importing basic GUI: {e}")
        print("Make sure futuristic_trading_gui.py exists")
    except Exception as e:
        print(f"❌ Error launching basic GUI: {e}")

def launch_enhanced_gui():
    """Launch the enhanced gaming GUI"""
    print("🎮 Launching Enhanced Gaming GUI...")
    try:
        from enhanced_trading_gui import main
        main()
    except ImportError as e:
        print(f"❌ Error importing enhanced GUI: {e}")
        print("Make sure enhanced_trading_gui.py exists")
    except Exception as e:
        print(f"❌ Error launching enhanced GUI: {e}")

def launch_cli():
    """Launch the command line interface"""
    print("📊 Launching Command Line Interface...")
    try:
        from run_strategy import main
        main()
    except ImportError as e:
        print(f"❌ Error importing CLI: {e}")
        print("Make sure run_strategy.py exists")
    except Exception as e:
        print(f"❌ Error launching CLI: {e}")

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies:")
        print("pip install " + " ".join(missing_deps))
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🎮 Initializing Futuristic Trading Simulator...")
    
    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (0-3): ").strip()
            
            if choice == "1":
                launch_basic_gui()
                break
            elif choice == "2":
                launch_enhanced_gui()
                break
            elif choice == "3":
                launch_cli()
                break
            elif choice == "0":
                print("👋 Goodbye! Happy trading!")
                break
            else:
                print("❌ Invalid choice. Please enter 0, 1, 2, or 3.")
                input("Press Enter to continue...")
                os.system('cls' if os.name == 'nt' else 'clear')
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Happy trading!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main() 