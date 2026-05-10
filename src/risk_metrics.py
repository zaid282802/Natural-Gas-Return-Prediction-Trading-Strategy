"""Natural Gas Return Prediction - Risk and Performance Metrics.

Calculates standard portfolio performance metrics.

Methodology:
    Sharpe ratio: Sharpe (1994) "The Sharpe Ratio"
    Sortino ratio: Sortino & Price (1994)
    Maximum drawdown: Standard peak-to-trough calculation
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional

# Configuration
ANNUALIZATION_FACTOR = 12                 # Monthly to annual conversion
DEFAULT_CONFIDENCE = 0.95                 # VaR/CVaR confidence level
CALMAR_DRAWDOWN_FLOOR = 0.0001           # Minimum drawdown to avoid div/zero
CALMAR_CAP = 1000                         # Cap Calmar ratio for display
DEFAULT_TAIL_PERCENTILE = 95              # Percentile for tail ratio


class RiskMetrics:
    """Risk metrics for evaluating trading strategy downside exposure."""

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0, periods_per_year=ANNUALIZATION_FACTOR):
        """Annualized Sharpe ratio: (mean_return - rf) / std_return * sqrt(12)."""
        # Sharpe (1994): SR = (R_p - R_f) / sigma_p * sqrt(12)
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def sortino_ratio(returns, target=0, periods_per_year=ANNUALIZATION_FACTOR):
        """Sortino ratio: excess return over downside deviation only."""
        # Sortino & Price (1994): uses downside deviation instead of total volatility
        excess = returns - target
        downside = excess[excess < 0]

        if len(downside) == 0:
            return np.inf  # No downside = infinite Sortino

        downside_std = downside.std()
        return excess.mean() / downside_std * np.sqrt(periods_per_year)

    @staticmethod
    def calmar_ratio(returns, max_drawdown, periods_per_year=ANNUALIZATION_FACTOR):
        """Return per unit of max drawdown. Returns 0 if drawdown < 0.01%."""
        # Handle zero or near-zero drawdown
        if abs(max_drawdown) < CALMAR_DRAWDOWN_FLOOR:
            return 0

        annualized_return = returns.mean() * periods_per_year
        calmar = annualized_return / abs(max_drawdown)

        return min(calmar, CALMAR_CAP)

    @staticmethod
    def value_at_risk(returns, confidence=DEFAULT_CONFIDENCE):
        """Historical VaR at given confidence level."""
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def conditional_var(returns, confidence=DEFAULT_CONFIDENCE):
        """CVaR (Expected Shortfall): average loss beyond VaR threshold."""
        var = RiskMetrics.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def max_drawdown(returns):
        """Maximum peak-to-trough drawdown from cumulative return series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def max_drawdown_duration(returns):
        """Duration (in periods) of the longest drawdown episode."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        in_drawdown = drawdown < 0

        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration


class PerformanceMetrics:
    """Trading performance statistics (win rate, profit factor, expectancy)."""

    @staticmethod
    def win_rate(returns):
        """Fraction of periods with positive returns."""
        return (returns > 0).sum() / len(returns)

    @staticmethod
    def profit_factor(returns):
        """Ratio of gross profits to gross losses."""
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses == 0:
            return np.inf

        return wins / losses

    @staticmethod
    def average_win_loss_ratio(returns):
        """Average winning trade divided by average losing trade."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(losses) == 0:
            return np.inf

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean())

        return avg_win / avg_loss

    @staticmethod
    def expectancy(returns):
        """Expected value per trade: P(win)*avg_win - P(loss)*avg_loss."""
        win_rate = PerformanceMetrics.win_rate(returns)

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    @staticmethod
    def tail_ratio(returns, percentile=DEFAULT_TAIL_PERCENTILE):
        """Ratio of right-tail magnitude to left-tail magnitude. >1.0 is favorable."""
        right_tail = abs(np.percentile(returns, percentile))
        left_tail = abs(np.percentile(returns, 100 - percentile))

        if left_tail == 0:
            return np.inf

        return right_tail / left_tail


class ComprehensiveAnalysis:
    """Full performance report combining return, risk, and trading metrics."""

    @staticmethod
    def print_report(returns):
        """Print comprehensive performance report for a return series."""
        print("PERFORMANCE REPORT")

        # Calculate return metrics
        cumulative = (1 + returns).cumprod().iloc[-1] - 1
        n_periods = len(returns)
        annualized = (1 + cumulative) ** (ANNUALIZATION_FACTOR / n_periods) - 1
        volatility = returns.std() * np.sqrt(ANNUALIZATION_FACTOR)

        print("\n RETURN METRICS:")
        print(f"  Total Return:           {cumulative*100:>10.2f}%")
        print(f"  Annualized Return:      {annualized*100:>10.2f}%")
        print(f"  Volatility (Annual):    {volatility*100:>10.2f}%")

        # Calculate and print risk-adjusted ratios
        sharpe = RiskMetrics.sharpe_ratio(returns)
        sortino = RiskMetrics.sortino_ratio(returns)
        max_dd = RiskMetrics.max_drawdown(returns)
        calmar = RiskMetrics.calmar_ratio(returns, max_dd)

        print(f"\n RISK-ADJUSTED RATIOS:")
        print(f"  Sharpe Ratio:           {sharpe:>10.3f}")
        print(f"  Sortino Ratio:          {sortino:>10.3f}")

        # Handle special cases for Calmar ratio display
        if calmar == 0:
            print(f"  Calmar Ratio:           {'N/A (No DD)':>10}")
        elif calmar >= 1000:
            print(f"  Calmar Ratio:           {'>1000.0':>10}")
        else:
            print(f"  Calmar Ratio:           {calmar:>10.3f}")

        # Calculate and print downside risk
        var_95 = RiskMetrics.value_at_risk(returns)
        cvar_95 = RiskMetrics.conditional_var(returns)

        print(f"\n DOWNSIDE RISK:")
        print(f"  Max Drawdown:           {max_dd*100:>10.2f}%")
        print(f"  VaR (95%):              {var_95*100:>10.2f}%")
        print(f"  CVaR (95%):             {cvar_95*100:>10.2f}%")

        # Calculate and print distribution stats
        skew = returns.skew()
        kurt = returns.kurtosis()

        print(f"\n DISTRIBUTION:")
        print(f"  Skewness:               {skew:>10.3f}")
        print(f"  Kurtosis:               {kurt:>10.3f}")
        print(f"  Best Month:             {returns.max()*100:>10.2f}%")
        print(f"  Worst Month:            {returns.min()*100:>10.2f}%")

        # Calculate and print trading statistics
        win_rate = PerformanceMetrics.win_rate(returns)
        profit_factor = PerformanceMetrics.profit_factor(returns)
        avg_win_loss = PerformanceMetrics.average_win_loss_ratio(returns)
        expectancy = PerformanceMetrics.expectancy(returns)

        print(f"\n TRADING STATISTICS:")
        print(f"  Win Rate:               {win_rate*100:>10.2f}%")
        print(f"  Profit Factor:          {profit_factor:>10.3f}")
        print(f"  Avg Win/Loss Ratio:     {avg_win_loss:>10.3f}")
        print(f"  Expectancy:             {expectancy*100:>10.2f}%")
