# Forex HMM Scalping Strategy

A sophisticated forex trading strategy based on Hidden Markov Models (HMM) and probability analysis, designed for scalping with a focus on achieving ~50% win rate with 1:1.5 risk-reward ratio.

## Features

### 🧠 Advanced Machine Learning

- **Hidden Markov Models**: Uses HMM to identify market regimes and states
- **Probabilistic Trading**: Makes decisions based on state probabilities
- **Feature Engineering**: 17+ technical indicators and market session features
- **Adaptive Learning**: Continuously learns from market patterns

### 📊 Comprehensive Backtesting

- **Historical Analysis**: Backtest on multiple timeframes
- **Performance Metrics**: Win rate, profit factor, Sharpe ratio, drawdown
- **Visual Analytics**: Detailed charts and performance visualization
- **Trade Analysis**: Individual trade breakdown and statistics

### 🎯 Parameter Optimization

- **Grid Search**: Systematic parameter optimization
- **Multi-Objective**: Balances win rate, profit factor, and risk metrics
- **Composite Scoring**: Weighted scoring system for best parameters
- **Visualization**: Optimization results plotting and analysis

### 🛡️ Risk Management

- **Position Sizing**: Dynamic position sizing based on risk
- **Stop Loss/Take Profit**: ATR-based dynamic levels
- **Daily Loss Limits**: Maximum daily loss protection
- **Drawdown Control**: Maximum drawdown limits
- **Concurrent Positions**: Limit on simultaneous trades

### 🔄 Live Trading Simulation

- **Real-time Data**: Uses live market data feeds
- **Threading**: Non-blocking trading loop
- **Performance Tracking**: Real-time equity curve and metrics
- **Risk Monitoring**: Continuous risk assessment

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd forex-hmm-strategy
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Strategy Run

```python
from forex_hmm_strategy import main

# Run the basic strategy
strategy, performance = main()
```

### 2. Parameter Optimization

```python
from strategy_optimizer import main

# Run optimization to find best parameters
optimizer, best_strategy, best_performance = main()
```

### 3. Live Trading Simulation

```python
from live_trading_simulator import main

# Run live trading simulation
simulator = main()
```

## Strategy Components

### Hidden Markov Model

- **States**: 2-5 hidden market states (optimizable)
- **Features**: Technical indicators, price patterns, market sessions
- **Training**: Gaussian HMM with full covariance matrix
- **Prediction**: State probabilities for signal generation

### Technical Features

1. **Price Features**: Returns, log returns, price changes
2. **Momentum**: RSI, MACD, ROC, momentum indicators
3. **Volatility**: ATR, Bollinger Bands, rolling volatility
4. **Trend**: Multiple moving averages and ratios
5. **Volume**: Volume-based indicators (when available)
6. **Time**: Market session indicators (London, NY, Tokyo, Overlap)

### Signal Generation

- **Bullish Signals**: High probability of bullish state + confirmations
- **Bearish Signals**: High probability of bearish state + confirmations
- **Filters**: RSI overbought/oversold, market session timing
- **Thresholds**: Optimizable probability thresholds

### Risk Management

- **Stop Loss**: ATR-based dynamic stop loss
- **Take Profit**: 1:1.5 risk-reward ratio (configurable)
- **Position Size**: Risk-based position sizing
- **Daily Limits**: Maximum daily loss protection
- **Drawdown**: Maximum drawdown limits

## Configuration

### Strategy Parameters

```python
strategy_params = {
    'n_components': 3,           # Number of HMM states
    'risk_reward_ratio': 1.5,    # Risk to reward ratio
    'prob_threshold': 0.6,       # Minimum state probability
    'state_prob_threshold': 0.7, # Current state confidence
    'atr_multiplier': 1.5        # ATR multiplier for stops
}
```

### Risk Parameters

```python
risk_params = {
    'max_daily_loss': 500,       # Maximum daily loss
    'max_drawdown': 0.15,        # Maximum drawdown (15%)
    'max_positions': 2,          # Maximum concurrent positions
    'position_size_pct': 0.1,    # Position size (10% of capital)
    'max_risk_per_trade': 0.02   # Maximum risk per trade (2%)
}
```

## Performance Targets

### Target Metrics

- **Win Rate**: ~50% (balanced approach)
- **Risk-Reward**: 1:1.5 minimum
- **Profit Factor**: >1.2
- **Sharpe Ratio**: >0.5
- **Maximum Drawdown**: <15%

### Scalping Focus

- **Timeframe**: 5-minute charts
- **Holding Period**: Minutes to hours
- **Market Sessions**: Focus on London-NY overlap
- **Currency Pairs**: Major pairs (EUR/USD, GBP/USD, etc.)

## File Structure

```
forex-hmm-strategy/
├── forex_hmm_strategy.py      # Main strategy implementation
├── strategy_optimizer.py      # Parameter optimization
├── live_trading_simulator.py  # Live trading simulation
├── run_strategy.py           # Main runner script
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Usage Examples

### Example 1: Run Basic Strategy

```python
from forex_hmm_strategy import ForexHMMStrategy

# Initialize strategy
strategy = ForexHMMStrategy(symbol='EURUSD=X', n_components=3, risk_reward_ratio=1.5)

# Fetch data and train
strategy.fetch_data(period='3mo', interval='5m')
strategy.create_features()
strategy.train_hmm()

# Generate signals and backtest
strategy.generate_signals()
performance = strategy.backtest()

# Plot results
strategy.plot_results()
```

### Example 2: Optimize Parameters

```python
from strategy_optimizer import StrategyOptimizer

# Initialize optimizer
optimizer = StrategyOptimizer(symbol='EURUSD=X')

# Run optimization
results = optimizer.optimize_parameters(
    n_components_range=[2, 3, 4],
    risk_reward_range=[1.2, 1.5, 1.8],
    prob_threshold_range=[0.5, 0.6, 0.7]
)

# Get best parameters
best_params = optimizer.find_best_parameters()

# Run strategy with best parameters
best_strategy, best_performance = optimizer.run_best_strategy()
```

### Example 3: Live Trading

```python
from live_trading_simulator import LiveTradingSimulator

# Initialize with optimized parameters
simulator = LiveTradingSimulator(
    strategy_params=best_params,
    initial_capital=10000
)

# Initialize and start trading
simulator.initialize_strategy()
simulator.start_trading()

# Monitor performance
performance = simulator.get_performance_summary()
simulator.plot_live_performance()
```

## Advanced Features

### Custom Indicators

Add your own technical indicators to the feature set:

```python
def create_custom_features(self):
    # Add custom indicators here
    df['custom_indicator'] = your_custom_function(df['Close'])
    return df
```

### Alternative Data Sources

Replace Yahoo Finance with your preferred data source:

```python
def fetch_custom_data(self):
    # Implement your data fetching logic
    return data
```

### Risk Management Customization

Modify risk management rules:

```python
def custom_risk_check(self, capital, position):
    # Implement custom risk logic
    return can_trade, position_size
```

## Performance Monitoring

### Key Metrics

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Win/Loss**: Average profit/loss per trade

### Visualization

- Equity curve with drawdown
- Trade distribution analysis
- HMM state visualization
- Signal timing analysis
- Risk metrics dashboard

## Troubleshooting

### Common Issues

1. **Data Fetching**: Ensure internet connection for Yahoo Finance
2. **HMM Convergence**: Try different n_components if training fails
3. **No Signals**: Adjust probability thresholds if too restrictive
4. **Poor Performance**: Run optimization to find better parameters

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Disclaimer

This trading strategy is for educational and research purposes only. Past performance does not guarantee future results. Trading forex involves substantial risk and may not be suitable for all investors. Always test strategies thoroughly before using real money.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:

1. Check the troubleshooting section
2. Review the code documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Trading! 📈**
