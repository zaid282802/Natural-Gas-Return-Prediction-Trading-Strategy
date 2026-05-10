"""Natural Gas Return Prediction - Data Loading Utilities.

Loads and preprocesses monthly natural gas fundamental data.

Data Source:
    data/Book1.1.xlsx - 71 monthly observations (Jan 2020 - Nov 2025)
    Sources: EIA (Henry Hub spot, storage), World Bank (coal price index),
    EIA International (LNG trade balance), ICE (Carbon EUA futures)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configuration
REQUIRED_COLUMN = 'NG_Return'             # Target variable column name
DEFAULT_TRANSACTION_COST = 0.001          # 10 bps per trade


def load_data_from_r(filepath):
    """Load monthly natural gas data from CSV or Excel file."""
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif filepath.endswith(('.xlsx', '.xls')):
        data = pd.read_excel(filepath, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    assert len(data) > 0, "No data loaded from file"

    # Check for required columns
    if REQUIRED_COLUMN not in data.columns:
        raise ValueError(f"Missing required column: {REQUIRED_COLUMN}")

    # Handle NaN - explicit drop with warning
    if data.isnull().any().any():
        print("WARNING: Data contains NaN values. Dropping rows with NaN...")
        data = data.dropna()

    data = data.sort_index()

    print(f"Data loaded: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    return data


def validate_model_data(data, model):
    """Validate that data contains all features required by the model."""
    missing = [f for f in model.features if f not in data.columns]
    if missing:
        raise ValueError(f"Data missing required features: {missing}")
    return True


def export_results_to_excel(backtest_results, metrics, output_path):
    """Export backtest results and metrics to Excel file."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary metrics
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.to_excel(writer, sheet_name='Summary')

        # Returns
        returns_df = backtest_results[['Date', 'Actual_Return',
                                       'Predicted_Return', 'Strategy_Return']]
        returns_df.to_excel(writer, sheet_name='Returns', index=False)

        # Signals
        signals_df = backtest_results[['Date', 'Signal', 'Strategy_Return']]
        signals_df.to_excel(writer, sheet_name='Signals', index=False)

    print(f"Results exported to: {output_path}")


def create_performance_summary_table(metrics):
    """Create a formatted performance summary table."""
    summary = "NATURAL GAS STRATEGY - PERFORMANCE SUMMARY\n"

    summary += "RETURN METRICS\n"
    for key in ['Total Return', 'Annualized Return', 'Volatility (Annual)']:
        if key in metrics:
            summary += f"  {key:<25} {metrics[key]}\n"

    summary += "\nRISK-ADJUSTED RATIOS\n"
    for key in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
        if key in metrics:
            summary += f"  {key:<25} {metrics[key]}\n"

    summary += "\nRISK METRICS\n"
    for key in ['Max Drawdown', 'VaR (95%)', 'CVaR (95%)']:
        if key in metrics:
            summary += f"  {key:<25} {metrics[key]}\n"

    summary += "\nTRADING STATISTICS\n"
    for key in ['Win Rate', 'Profit Factor', 'Expectancy']:
        if key in metrics:
            summary += f"  {key:<25} {metrics[key]}\n"

    return summary


def calculate_transaction_costs(signals, cost_per_trade=DEFAULT_TRANSACTION_COST):
    """Calculate transaction costs based on signal changes."""
    trades = signals.diff().abs()
    costs = trades * cost_per_trade
    return costs


def calculate_model_comparison_table(fitted_models):
    """Build comparison table across fitted models, sorted by SSR."""
    comparison = []

    for name, model in fitted_models.items():
        row = {'Model': name}

        if hasattr(model, 'ssr'):
            row['SSR'] = model.ssr
        if hasattr(model, 'r2'):
            row['R²'] = model.r2
        if hasattr(model, 'adj_r2'):
            row['Adj_R²'] = model.adj_r2

        comparison.append(row)

    return pd.DataFrame(comparison).sort_values('SSR')
