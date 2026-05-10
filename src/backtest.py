"""Natural Gas Return Prediction - Walk-Forward Backtesting Engine.

Implements expanding-window walk-forward testing to avoid look-ahead bias.

Methodology:
    Walk-forward analysis: Aronson (2006) "Evidence-Based Technical Analysis"
    Expanding window: 36-month minimum training, monthly out-of-sample steps
    Signal generation: Long if predicted return > +2%, Short if < -2%
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAIN_WINDOW_MONTHS = 36                  # Minimum training window (3 years)
MIN_TRAIN_WINDOW = 24                     # Absolute minimum training months
SIGNAL_THRESHOLD_PCT = 0.02               # +/- 2% threshold for long/short signals
ANNUALIZATION_FACTOR = 12                 # Monthly to annual conversion
VAR_CONFIDENCE = 0.95                     # VaR confidence level
CALMAR_DRAWDOWN_FLOOR = 0.0001           # Minimum drawdown to avoid division by zero
CALMAR_CAP = 1000                         # Cap Calmar ratio for display purposes
PROGRESS_INTERVAL = 10                    # Print progress every N periods


class WalkForwardBacktest:
    """Expanding-window walk-forward backtest with no look-ahead bias."""

    def __init__(self, model, data, train_window=TRAIN_WINDOW_MONTHS, expanding=True):
        assert train_window >= MIN_TRAIN_WINDOW, \
            f"Training window must be at least {MIN_TRAIN_WINDOW} months"
        self.model = model
        self.data = data
        self.train_window = train_window
        self.expanding = expanding

    def run(self):
        """Execute expanding-window walk-forward backtest."""
        predictions = []
        actuals = []
        dates = []

        total_periods = len(self.data)

        print("WALK-FORWARD BACKTEST - OUT-OF-SAMPLE EVALUATION")
        print(f"Total observations: {total_periods}")
        print(f"Training window: {self.train_window} months")
        print(f"Out-of-sample periods: {total_periods - self.train_window}")
        print(f"Window type: {'Expanding' if self.expanding else 'Rolling'}")

        # Walk-forward loop: Aronson (2006) expanding-window methodology
        for i in range(self.train_window, total_periods):
            # Define training window
            if self.expanding:
                train_start = 0
                train_end = i
            else:
                train_start = i - self.train_window
                train_end = i

            # Split data - strictly no future information
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[i:i+1]

            # Fit model on training data only
            self.model.fit(train_data)

            # Predict next period (no look-ahead)
            pred = self.model.predict(test_data)[0]
            actual = test_data['NG_Return'].values[0]
            date = test_data.index[0]

            predictions.append(pred)
            actuals.append(actual)
            dates.append(date)

            # Progress indicator
            if (i - self.train_window + 1) % PROGRESS_INTERVAL == 0:
                periods_done = i - self.train_window + 1
                print(f"  Processed {periods_done}/{total_periods - self.train_window} periods...")

        # Assemble results
        results = pd.DataFrame({
            'Date': dates,
            'Predicted_Return': predictions,
            'Actual_Return': actuals
        })

        results['Signal'] = self._generate_signals(results['Predicted_Return'])
        results['Strategy_Return'] = results['Signal'] * results['Actual_Return']
        results['BuyHold_Return'] = results['Actual_Return']
        results['Strategy_Cumulative'] = (1 + results['Strategy_Return']).cumprod()
        results['BuyHold_Cumulative'] = (1 + results['BuyHold_Return']).cumprod()

        self.results = results

        print("\nSUCCESS: Backtest complete!")

        return results

    def _generate_signals(self, predictions, threshold=SIGNAL_THRESHOLD_PCT):
        """Convert predicted returns to trading signals (+1 Long, -1 Short, 0 Flat)."""
        signals = np.zeros(len(predictions))
        signals[predictions > threshold] = 1    # Long
        signals[predictions < -threshold] = -1  # Short
        return signals

    def calculate_metrics(self):
        """Calculate full suite of performance and risk metrics."""
        results = self.results
        strat_returns = results['Strategy_Return']

        # Cumulative return
        cumulative = results['Strategy_Cumulative'].iloc[-1] - 1

        # Annualized return (monthly data)
        n_months = len(strat_returns)
        annualized = (1 + cumulative) ** (ANNUALIZATION_FACTOR / n_months) - 1

        # Sharpe (1994): SR = mean(R) / std(R) * sqrt(12)
        sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)

        # Sortino & Price (1994): uses downside deviation only
        downside_returns = strat_returns[strat_returns < 0]
        sortino = strat_returns.mean() / downside_returns.std() * np.sqrt(ANNUALIZATION_FACTOR)

        # Max drawdown: peak-to-trough
        cumulative_series = results['Strategy_Cumulative']
        running_max = cumulative_series.expanding().max()
        drawdown_series = (cumulative_series - running_max) / running_max
        max_drawdown = drawdown_series.min()

        # Calmar ratio (handle zero or near-zero drawdown)
        if abs(max_drawdown) < CALMAR_DRAWDOWN_FLOOR:
            calmar = 0
        else:
            calmar = annualized / abs(max_drawdown)
            calmar = min(calmar, CALMAR_CAP)

        # Win rate
        win_rate = (strat_returns > 0).sum() / len(strat_returns)

        # Number of trades (signal changes)
        n_trades = results['Signal'].diff().abs().sum()

        # VaR and CVaR at 95% confidence
        var_95 = np.percentile(strat_returns, (1 - VAR_CONFIDENCE) * 100)
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
        