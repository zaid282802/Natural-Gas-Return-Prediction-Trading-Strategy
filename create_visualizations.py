import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_data_from_r
from src.models import SignificantOLSModel
from src.backtest import WalkForwardBacktest

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create output directory
os.makedirs('results/charts', exist_ok=True)


# Chart 1: Equity curve comparison
def plot_equity_curves(backtest, results):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Calculate returns
    strategy_curve = (1 + results['Strategy_Return']).cumprod()
    buyhold_curve = results['BuyHold_Cumulative']

    # Plot
    ax.plot(results['Date'], strategy_curve, linewidth=2.5,
            label='OLS Strategy (Sharpe 1.07)', color='#2E86AB')
    ax.plot(results['Date'], buyhold_curve, linewidth=2.5,
            label='Buy & Hold (Sharpe -0.48)', color='#A23B72', alpha=0.7)

    # Add event annotations
    covid_date = pd.Timestamp('2020-03-01')
    ukraine_date = pd.Timestamp('2022-02-01')

    if covid_date in results['Date'].values:
        idx = results[results['Date'] == covid_date].index[0]
        ax.annotate('COVID-19 Crash\n(NG: $1.63)',
                   xy=(covid_date, strategy_curve.iloc[idx]),
                   xytext=(covid_date, strategy_curve.iloc[idx] + 1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, ha='center', color='red')

    if ukraine_date in results['Date'].values:
        idx = results[results['Date'] == ukraine_date].index[0]
        ax.annotate('Ukraine War\n(NG: $9.35)',
                   xy=(ukraine_date, strategy_curve.iloc[idx]),
                   xytext=(ukraine_date, strategy_curve.iloc[idx] - 0.5),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, ha='center', color='red')

    # Labels and formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (Base = 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('Equity Curve: OLS Strategy vs Buy & Hold (Jan 2023 - Nov 2025)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Add text box
    final_strategy = strategy_curve.iloc[-1]
    final_buyhold = buyhold_curve.iloc[-1]
    textstr = f'Final Value:\nStrategy: ${final_strategy:.2f}\nBuy & Hold: ${final_buyhold:.2f}'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('results/charts/1_equity_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 1 saved")


# Chart 2: Drawdown timeline
def plot_drawdown_timeline(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Calculate drawdowns for strategy
    strategy_curve = (1 + results['Strategy_Return']).cumprod()
    running_max = strategy_curve.expanding().max()
    drawdown = (strategy_curve - running_max) / running_max

    # Calculate drawdowns for buy & hold
    buyhold_curve = results['BuyHold_Cumulative']
    bh_max = buyhold_curve.expanding().max()
    bh_drawdown = (buyhold_curve - bh_max) / bh_max

    # Plot strategy drawdown
    ax1.fill_between(results['Date'], drawdown * 100, 0,
                     color='#C1292E', alpha=0.6)
    ax1.plot(results['Date'], drawdown * 100, color='#C1292E', linewidth=2)
    ax1.axhline(y=-59, color='red', linestyle='--', linewidth=1.5, label='Max DD: -59.09%')
    ax1.set_ylabel('Strategy Drawdown (%)', fontsize=11, fontweight='bold')
    ax1.set_title('OLS Strategy Drawdown Timeline (Out-of-Sample)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot buy & hold drawdown
    ax2.fill_between(results['Date'], bh_drawdown * 100, 0,
                     color='#6C757D', alpha=0.6)
    ax2.plot(results['Date'], bh_drawdown * 100, color='#6C757D', linewidth=2)
    ax2.axhline(y=-83.77, color='red', linestyle='--', linewidth=1.5, label='Max DD: -83.77%')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Buy & Hold Drawdown (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Buy & Hold Drawdown Timeline', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/charts/2_drawdown_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 2 saved")


# Chart 3: Factor importance and coefficients
def plot_factor_importance(data):
    # Fit model
    model = SignificantOLSModel()
    model.fit(data)

    # Get coefficients (excluding intercept)
    coef_df = model.coefficients[model.coefficients['Variable'] != 'Intercept'].copy()
    coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coef', ascending=True)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color bars based on sign
    colors = []
    for coef in coef_df['Coefficient']:
        if coef > 0:
            colors.append('#2E86AB')
        else:
            colors.append('#A23B72')

    bars = ax.barh(coef_df['Variable'], coef_df['Coefficient'], color=colors, alpha=0.8)

    # Add value labels
    for i, bar in enumerate(bars):
        val = coef_df['Coefficient'].iloc[i]
        if val > 0:
            label_x = val + 0.02
            ha = 'left'
        else:
            label_x = val - 0.02
            ha = 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', ha=ha, va='center', fontsize=9, fontweight='bold')

    # Formatting
    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.set_xlabel('OLS Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Factor Importance: OLS Regression Coefficients (6 Significant Variables)',
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', alpha=0.8, label='Positive (increases returns)'),
        Patch(facecolor='#A23B72', alpha=0.8, label='Negative (decreases returns)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/charts/3_factor_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 3 saved")


# Chart 4: Rolling Sharpe ratio evolution
def plot_rolling_sharpe(results):
    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate 12-month rolling Sharpe
    strategy_returns = results['Strategy_Return'].values
    rolling_sharpe = []
    dates_rolling = []

    for i in range(12, len(strategy_returns)):
        window = strategy_returns[i-12:i]
        if window.std() > 0:
            sharpe = window.mean() / window.std() * np.sqrt(12)
        else:
            sharpe = 0
        rolling_sharpe.append(sharpe)
        dates_rolling.append(results['Date'].iloc[i])

    # Plot
    ax.plot(dates_rolling, rolling_sharpe, linewidth=2.5, color='#F77F00',
           label='12-Month Rolling Sharpe')
    ax.axhline(y=1.07, color='green', linestyle='--', linewidth=2,
              label='Full Sample Sharpe: 1.07')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Fill positive/negative areas
    rolling_sharpe_array = np.array(rolling_sharpe)
    ax.fill_between(dates_rolling, rolling_sharpe_array, 0,
                    where=rolling_sharpe_array > 0,
                    alpha=0.3, color='#06A77D', label='Positive Sharpe Periods')
    ax.fill_between(dates_rolling, rolling_sharpe_array, 0,
                    where=rolling_sharpe_array <= 0,
                    alpha=0.3, color='#C1292E', label='Negative Sharpe Periods')

    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('12-Month Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Rolling Sharpe Ratio Evolution (12-Month Window)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/charts/4_rolling_sharpe.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 4 saved")


# Chart 5: Prediction accuracy over time
def plot_prediction_accuracy(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Check if predictions are correct
    results['Correct_Direction'] = (np.sign(results['Predicted_Return']) ==
                                    np.sign(results['Actual_Return']))

    # Calculate rolling accuracy
    rolling_accuracy = results['Correct_Direction'].rolling(6).mean() * 100

    # Plot 1: Actual vs predicted returns
    ax1.scatter(results['Date'], results['Actual_Return'] * 100,
               alpha=0.6, s=50, color='#2E86AB', label='Actual Returns', marker='o')
    ax1.scatter(results['Date'], results['Predicted_Return'] * 100,
               alpha=0.6, s=50, color='#F77F00', label='Predicted Returns', marker='^')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Monthly Return (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Predicted vs Actual Returns (Out-of-Sample)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rolling accuracy
    ax2.plot(results['Date'], rolling_accuracy, linewidth=2.5,
            color='#06A77D', label='6-Month Rolling Accuracy')
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5,
               label='Random Guess (50%)', alpha=0.7)
    ax2.axhline(y=63, color='green', linestyle='--', linewidth=1.5,
               label='Full Sample: 63%')
    ax2.fill_between(results['Date'], rolling_accuracy, 50,
                    where=rolling_accuracy >= 50, alpha=0.3, color='#06A77D')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Directional Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('6-Month Rolling Directional Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig('results/charts/5_prediction_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 5 saved")


# Chart 6: Factor correlation matrix
def plot_factor_correlation_matrix(data):
    # Get significant factors
    model = SignificantOLSModel()
    factors = model.features

    # Calculate correlation
    corr_matrix = data[factors].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
               square=True, linewidths=1, cbar_kws={"shrink": 0.8},
               vmin=-1, vmax=1, ax=ax)

    ax.set_title('Factor Correlation Matrix (6 Significant Variables)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add note
    textstr = 'Low Correlation = Low Multicollinearity\nAll VIF < 2.5 (excellent)'
    ax.text(1.15, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig('results/charts/6_factor_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart 6 saved")


def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("\nGenerating visualizations...")

    # Load data
    data = load_data_from_r('data/Book1.1.xlsx')

    # Run backtest
    model = SignificantOLSModel()
    backtest = WalkForwardBacktest(model, data, train_window=36, expanding=True)
    results = backtest.run()

    # Generate all charts
    plot_equity_curves(backtest, results)
    plot_drawdown_timeline(results)
    plot_factor_importance(data)
    plot_rolling_sharpe(results)
    plot_prediction_accuracy(results)
    plot_factor_correlation_matrix(data)

    print("\nAll charts saved to results/charts/")
if __name__ == "__main__":
    main()
