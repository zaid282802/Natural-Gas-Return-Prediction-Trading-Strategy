# Walk-forward backtesting engine with no look-ahead bias.
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class WalkForwardBacktest:
    def __init__(self, model, data, train_window=36, expanding=True): #used expanding window because natural gas fundamentals are relatively stable
        self.model = model
        self.data = data
        self.train_window = train_window
        self.expanding = expanding
        
    def run(self):
        predictions = []
        actuals = []
        dates = []
        
        total_periods = len(self.data)
        
        print("WALK-FORWARD BACKTEST - OUT-OF-SAMPLE EVALUATION")
        print(f"Total observations: {total_periods}")
        print(f"Training window: {self.train_window} months")
        print(f"Out-of-sample periods: {total_periods - self.train_window}")
        print(f"Window type: {'Expanding' if self.expanding else 'Rolling'}")
        
        # Walk-forward loop
        for i in range(self.train_window, total_periods):
            # Define training window
            if self.expanding:
                # Use all past data
                train_start = 0
                train_end = i
            else:
                # Use fixed 36-month window
                train_start = i - self.train_window
                train_end = i
            
            # Split data
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[i:i+1]
            
            # Fit model on training data only
            self.model.fit(train_data)
            
            # Predict next period (no look-ahead!)
            pred = self.model.predict(test_data)[0]
            actual = test_data['NG_Return'].values[0]
            date = test_data.index[0]
            
            # Store results
            predictions.append(pred)
            actuals.append(actual)
            dates.append(date)
            
            # Progress indicator
            if (i - self.train_window + 1) % 10 == 0:
                periods_done = i - self.train_window + 1
                print(f"  Processed {periods_done}/{total_periods - self.train_window} periods...")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': dates,
            'Predicted_Return': predictions,
            'Actual_Return': actuals
        })
        
        # Generate trading signals
        results['Signal'] = self.generate_signals(results['Predicted_Return'])
        
        # Calculate strategy returns
        results['Strategy_Return'] = results['Signal'] * results['Actual_Return']
        
        # Buy-and-hold benchmark
        results['BuyHold_Return'] = results['Actual_Return']
        
        # Calculate cumulative returns
        results['Strategy_Cumulative'] = (1 + results['Strategy_Return']).cumprod()
        results['BuyHold_Cumulative'] = (1 + results['BuyHold_Return']).cumprod()
        
        self.results = results
        
        print("\nSUCCESS: Backtest complete!")
        
        return results
    
    #Convert predictions to trading signals.
    def generate_signals(self, predictions, threshold=0.02): #2% THRESHOLD FOR NATURAL GAS Buy/SELL SIGNALS
        
        signals = np.zeros(len(predictions))
        signals[predictions > threshold] = 1   # Long
        signals[predictions < -threshold] = -1  # Short
        return signals
    
    def calculate_metrics(self):
        results = self.results
        strat_returns = results['Strategy_Return']
        
        # Cumulative return
        cumulative = results['Strategy_Cumulative'].iloc[-1] - 1
        
        # Annualized return (monthly data)
        n_months = len(strat_returns)
        annualized = (1 + cumulative) ** (12 / n_months) - 1
        
        # Sharpe ratio
        sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(12)
        
        # Sortino ratio
        downside_returns = strat_returns[strat_returns < 0]
        sortino = strat_returns.mean() / downside_returns.std() * np.sqrt(12)
        
        # Max drawdown
        cumulative_series = results['Strategy_Cumulative']
        running_max = cumulative_series.expanding().max()
        drawdown_series = (cumulative_series - running_max) / running_max
        max_drawdown = drawdown_series.min()

        # Calmar ratio (handle zero or near-zero drawdown)
        if abs(max_drawdown) < 0.0001:  # Less than 0.01% drawdown
            calmar = 0  # Display as 0 instead of inf
        else:
            calmar = annualized / abs(max_drawdown)
            # Cap at reasonable maximum to avoid display issues
            calmar = min(calmar, 1000)
        
        # Win rate
        win_rate = (strat_returns > 0).sum() / len(strat_returns)
        
        # Number of trades (signal changes)
        n_trades = results['Signal'].diff().abs().sum()
        
        # VaR and CVaR
        var_95 = np.percentile(strat_returns, 5)
        cvar_95 = strat_returns[strat_returns <= var_95].mean()
        
        # Average win/loss
        wins = strat_returns[strat_returns > 0]
        losses = strat_returns[strat_returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Benchmark comparison
        bh_cumulative = results['BuyHold_Cumulative'].iloc[-1] - 1
        
        metrics = {
            'Total Return': f"{cumulative*100:.2f}%",
            'Annualized Return': f"{annualized*100:.2f}%",
            'Sharpe Ratio': f"{sharpe:.3f}",
            'Sortino Ratio': f"{sortino:.3f}",
            'Calmar Ratio': f"{calmar:.3f}",
            'Max Drawdown': f"{max_drawdown*100:.2f}%",
            'VaR (95%)': f"{var_95*100:.2f}%",
            'CVaR (95%)': f"{cvar_95*100:.2f}%",
            'Win Rate': f"{win_rate*100:.2f}%",
            'Number of Trades': int(n_trades),
            'Avg Win': f"{avg_win*100:.2f}%",
            'Avg Loss': f"{avg_loss*100:.2f}%",
            'Buy & Hold Return': f"{bh_cumulative*100:.2f}%"
        }
        
        return metrics
    
    def print_performance(self):
        """Print formatted performance summary."""
        metrics = self.calculate_metrics()
        
        print("STRATEGY PERFORMANCE SUMMARY")
        
        print("\nRETURNS:")
        print(f"  Total Return:          {metrics['Total Return']:>12}")
        print(f"  Annualized Return:     {metrics['Annualized Return']:>12}")
        print(f"  Buy & Hold Return:     {metrics['Buy & Hold Return']:>12}")

        print("\nRISK-ADJUSTED:")
        print(f"  Sharpe Ratio:          {metrics['Sharpe Ratio']:>12}")
        print(f"  Sortino Ratio:         {metrics['Sortino Ratio']:>12}")
        print(f"  Calmar Ratio:          {metrics['Calmar Ratio']:>12}")

        print("\nRISK METRICS:")
        print(f"  Max Drawdown:          {metrics['Max Drawdown']:>12}")
        print(f"  VaR (95%):             {metrics['VaR (95%)']:>12}")
        print(f"  CVaR (95%):            {metrics['CVaR (95%)']:>12}")

        print("\nTRADING STATS:")
        print(f"  Win Rate:              {metrics['Win Rate']:>12}")
        print(f"  Number of Trades:      {metrics['Number of Trades']:>12}")
        print(f"  Avg Win:               {metrics['Avg Win']:>12}")
        print(f"  Avg Loss:              {metrics['Avg Loss']:>12}")
        