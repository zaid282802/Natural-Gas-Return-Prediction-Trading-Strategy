import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional


class RiskMetrics:# To calculate risk metrics for trading strategies.
    
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0, periods_per_year=12):
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def sortino_ratio(returns, target=0, periods_per_year=12): #Downside risk-adjusted return.
        excess = returns - target
        downside = excess[excess < 0]
        
        if len(downside) == 0:
            return np.inf  # No downside = infinite Sortino
        
        downside_std = downside.std()
        return excess.mean() / downside_std * np.sqrt(periods_per_year)
    
    @staticmethod
    def calmar_ratio(returns, max_drawdown, periods_per_year=12): # Return per unit of max drawdown. Returns 0 if drawdown is effectively zero (< 0.01%)
                                                                    #to avoid displaying infinity or huge numbers.

        # Handle zero or near-zero drawdown
        if abs(max_drawdown) < 0.0001:  # Less than 0.01% drawdown
            return 0  # Better than returning inf for display purposes

        annualized_return = returns.mean() * periods_per_year
        calmar = annualized_return / abs(max_drawdown)

        # Cap at reasonable maximum to avoid display issues
        return min(calmar, 1000)

    @staticmethod
    def value_at_risk(returns, confidence=0.95): # Value at Risk (VaR) calculation.
        
        return np.percentile(returns, (1-confidence)*100)
    
    @staticmethod
    def conditional_var(returns, confidence=0.95): #Conditional VaR (CVaR / Expected Shortfall).
        var = RiskMetrics.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def max_drawdown_duration(returns): # Duration of the worst drawdown.
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find periods in drawdown
        in_drawdown = drawdown < 0
        
        # Calculate duration
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
    @staticmethod
    def win_rate(returns): # Win rate calculation.
        return (returns > 0).sum() / len(returns)
    
    @staticmethod
    def profit_factor(returns):# Profit factor calculation.
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return np.inf
        
        return wins / losses
    
    @staticmethod
    def average_win_loss_ratio(returns): # Average win/loss ratio calculation.
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0:
            return np.inf
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean())
        
        return avg_win / avg_loss
    
    @staticmethod
    def expectancy(returns): # Expectancy calculation.
        win_rate = PerformanceMetrics.win_rate(returns)
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    @staticmethod
    def tail_ratio(returns, percentile=95):# Tail ratio calculation.Want tail_ratio > 1.0 (big winners, small losers)

        right_tail = abs(np.percentile(returns, percentile)) # magnitude of upper tail (big positive returns)
        left_tail = abs(np.percentile(returns, 100 - percentile))# magnitude of lower tail (big negative returns)

        if left_tail == 0:
            return np.inf
        
        return right_tail / left_tail  #> 1.0 = Bigger wins than losses (positive skew)
                                       # < 1.0 = Bigger losses than wins (negative skew)


class ComprehensiveAnalysis:

    @staticmethod
    def print_report(returns):

        print("PERFORMANCE REPORT")
        
        # Calculate return metrics
        cumulative = (1 + returns).cumprod().iloc[-1] - 1
        n_periods = len(returns)
        annualized = (1 + cumulative) ** (12 / n_periods) - 1
        volatility = returns.std() * np.sqrt(12)

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
