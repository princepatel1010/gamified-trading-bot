import numpy as np
import pandas as pd
from itertools import product
from forex_hmm_strategy import ForexHMMStrategy
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class StrategyOptimizer:
    def __init__(self, symbol='EUR=X'):
        self.symbol = symbol
        self.best_params = None
        self.optimization_results = None
        
    def optimize_parameters(self, 
                          n_components_range=[2, 3, 4, 5],
                          risk_reward_range=[1.2, 1.5, 1.8, 2.0],
                          prob_threshold_range=[0.5, 0.6, 0.7, 0.8],
                          state_prob_threshold_range=[0.6, 0.7, 0.8],
                          atr_multiplier_range=[1.0, 1.5, 2.0],
                          period='3mo',
                          interval='5m'):
        """
        Optimize strategy parameters using grid search
        """
        print("Starting parameter optimization...")
        
        # Generate parameter combinations
        param_combinations = list(product(
            n_components_range,
            risk_reward_range,
            prob_threshold_range,
            state_prob_threshold_range,
            atr_multiplier_range
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        results = []
        
        for i, (n_comp, rr_ratio, prob_thresh, state_prob_thresh, atr_mult) in enumerate(param_combinations):
            try:
                print(f"Testing combination {i+1}/{len(param_combinations)}: "
                      f"n_comp={n_comp}, rr={rr_ratio}, prob_thresh={prob_thresh}, "
                      f"state_prob_thresh={state_prob_thresh}, atr_mult={atr_mult}")
                
                # Create strategy with current parameters
                strategy = ForexHMMStrategy(symbol=self.symbol, 
                                          n_components=n_comp, 
                                          risk_reward_ratio=rr_ratio)
                
                # Fetch data
                strategy.fetch_data(period=period, interval=interval)
                
                # Create features
                strategy.create_features()
                
                # Train HMM
                strategy.train_hmm()
                
                # Generate signals with custom thresholds
                strategy.generate_signals_optimized(
                    prob_threshold=prob_thresh,
                    state_prob_threshold=state_prob_thresh,
                    atr_multiplier=atr_mult
                )
                
                # Backtest
                performance = strategy.backtest(initial_capital=10000, position_size=0.1)
                
                # Store results
                result = {
                    'n_components': n_comp,
                    'risk_reward_ratio': rr_ratio,
                    'prob_threshold': prob_thresh,
                    'state_prob_threshold': state_prob_thresh,
                    'atr_multiplier': atr_mult,
                    **performance
                }
                results.append(result)
                
                print(f"  -> Win Rate: {performance['win_rate']:.2%}, "
                      f"Total PnL: ${performance['total_pnl']:.2f}, "
                      f"Trades: {performance['total_trades']}")
                
            except Exception as e:
                print(f"  -> Error: {str(e)}")
                continue
        
        self.optimization_results = pd.DataFrame(results)
        
        if not self.optimization_results.empty:
            # Find best parameters based on multiple criteria
            self.find_best_parameters()
        
        return self.optimization_results
    
    def find_best_parameters(self):
        """
        Find best parameters based on multiple criteria
        """
        df = self.optimization_results.copy()
        
        # Filter for strategies with reasonable number of trades
        df = df[df['total_trades'] >= 10]
        
        if df.empty:
            print("No strategies with sufficient trades found")
            return None
        
        # Calculate composite score
        # Normalize metrics to 0-1 scale
        df['win_rate_norm'] = (df['win_rate'] - df['win_rate'].min()) / (df['win_rate'].max() - df['win_rate'].min())
        df['profit_factor_norm'] = np.clip((df['profit_factor'] - 1) / 4, 0, 1)  # Cap at 5
        df['sharpe_norm'] = np.clip(df['sharpe_ratio'] / 2, 0, 1)  # Cap at 2
        df['drawdown_norm'] = 1 - np.clip(abs(df['max_drawdown']), 0, 0.5) / 0.5  # Penalize high drawdown
        df['total_pnl_norm'] = np.clip(df['total_pnl'] / df['total_pnl'].max(), 0, 1)
        
        # Composite score with weights
        df['composite_score'] = (
            0.3 * df['win_rate_norm'] +
            0.25 * df['profit_factor_norm'] +
            0.2 * df['sharpe_norm'] +
            0.15 * df['drawdown_norm'] +
            0.1 * df['total_pnl_norm']
        )
        
        # Find best parameters
        best_idx = df['composite_score'].idxmax()
        self.best_params = df.loc[best_idx].to_dict()
        
        print("\nBest Parameters Found:")
        print("="*40)
        print(f"N Components: {self.best_params['n_components']}")
        print(f"Risk-Reward Ratio: {self.best_params['risk_reward_ratio']}")
        print(f"Probability Threshold: {self.best_params['prob_threshold']}")
        print(f"State Probability Threshold: {self.best_params['state_prob_threshold']}")
        print(f"ATR Multiplier: {self.best_params['atr_multiplier']}")
        print(f"Win Rate: {self.best_params['win_rate']:.2%}")
        print(f"Total Trades: {self.best_params['total_trades']}")
        print(f"Profit Factor: {self.best_params['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {self.best_params['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.best_params['max_drawdown']:.2%}")
        print(f"Composite Score: {self.best_params['composite_score']:.3f}")
        
        return self.best_params
    
    def plot_optimization_results(self):
        """
        Plot optimization results
        """
        if self.optimization_results is None or self.optimization_results.empty:
            print("No optimization results to plot")
            return
        
        df = self.optimization_results
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Win Rate vs Risk-Reward Ratio
        scatter = axes[0, 0].scatter(df['risk_reward_ratio'], df['win_rate'], 
                                   c=df['total_pnl'], cmap='viridis', alpha=0.7)
        axes[0, 0].set_xlabel('Risk-Reward Ratio')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_title('Win Rate vs Risk-Reward Ratio')
        plt.colorbar(scatter, ax=axes[0, 0], label='Total PnL')
        
        # Profit Factor vs Number of Components
        axes[0, 1].scatter(df['n_components'], df['profit_factor'], 
                          c=df['win_rate'], cmap='plasma', alpha=0.7)
        axes[0, 1].set_xlabel('Number of HMM Components')
        axes[0, 1].set_ylabel('Profit Factor')
        axes[0, 1].set_title('Profit Factor vs HMM Components')
        
        # Sharpe Ratio vs Drawdown
        axes[0, 2].scatter(df['max_drawdown'], df['sharpe_ratio'], 
                          c=df['total_trades'], cmap='coolwarm', alpha=0.7)
        axes[0, 2].set_xlabel('Max Drawdown')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].set_title('Sharpe Ratio vs Max Drawdown')
        
        # Win Rate Distribution
        axes[1, 0].hist(df['win_rate'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(0.5, color='red', linestyle='--', label='50% Target')
        axes[1, 0].set_xlabel('Win Rate')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Win Rate Distribution')
        axes[1, 0].legend()
        
        # Total PnL vs Total Trades
        axes[1, 1].scatter(df['total_trades'], df['total_pnl'], 
                          c=df['win_rate'], cmap='RdYlGn', alpha=0.7)
        axes[1, 1].set_xlabel('Total Trades')
        axes[1, 1].set_ylabel('Total PnL')
        axes[1, 1].set_title('Total PnL vs Total Trades')
        
        # Composite Score
        if 'composite_score' in df.columns:
            top_10 = df.nlargest(10, 'composite_score')
            axes[1, 2].barh(range(len(top_10)), top_10['composite_score'])
            axes[1, 2].set_yticks(range(len(top_10)))
            axes[1, 2].set_yticklabels([f"Params {i}" for i in range(len(top_10))])
            axes[1, 2].set_xlabel('Composite Score')
            axes[1, 2].set_title('Top 10 Parameter Combinations')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def run_best_strategy(self, period='3mo', interval='5m'):
        """
        Run the strategy with best parameters
        """
        if self.best_params is None:
            print("No best parameters found. Run optimization first.")
            return None
        
        print("Running strategy with best parameters...")
        
        # Create strategy with best parameters
        strategy = ForexHMMStrategy(
            symbol=self.symbol,
            n_components=int(self.best_params['n_components']),
            risk_reward_ratio=self.best_params['risk_reward_ratio']
        )
        
        # Fetch data
        strategy.fetch_data(period=period, interval=interval)
        
        # Create features
        strategy.create_features()
        
        # Train HMM
        strategy.train_hmm()
        
        # Generate signals with optimized parameters
        strategy.generate_signals_optimized(
            prob_threshold=self.best_params['prob_threshold'],
            state_prob_threshold=self.best_params['state_prob_threshold'],
            atr_multiplier=self.best_params['atr_multiplier']
        )
        
        # Backtest
        performance = strategy.backtest(initial_capital=10000, position_size=0.1)
        
        # Plot results
        strategy.plot_results()
        
        return strategy, performance



def main():
    """
    Main optimization function
    """
    # Initialize optimizer
    optimizer = StrategyOptimizer(symbol='BTCUSDm')
    
    # Run optimization
    results = optimizer.optimize_parameters(
        n_components_range=[2, 3, 4],
        risk_reward_range=[1.2, 1.5, 1.8],
        prob_threshold_range=[0.5, 0.6, 0.7],
        state_prob_threshold_range=[0.6, 0.7, 0.8],
        atr_multiplier_range=[1.0, 1.5, 2.0]
    )
    
    # Plot results
    optimizer.plot_optimization_results()
    
    # Run best strategy
    best_strategy, best_performance = optimizer.run_best_strategy()
    
    return optimizer, best_strategy, best_performance

if __name__ == "__main__":
    optimizer, strategy, performance = main() 