import pandas as pd
import numpy as np
from datetime import datetime
import os


def load_data_from_r(filepath):# Load data from CSV or Excel file.

    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif filepath.endswith(('.xlsx', '.xls')):
        data = pd.read_excel(filepath, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    # Check for required columns
    if 'NG_Return' not in data.columns:
        raise ValueError("Missing required column: NG_Return")

    # Handle NaN
    if data.isnull().any().any():
        print("WARNING: Data contains NaN values. Dropping rows with NaN...")
        data = data.dropna()

    data = data.sort_index()

    print(f"Data loaded: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    return data


def validate_model_data(data, model):# Validate that data contains all features required by the model.
    missing = [f for f in model.features if f not in data.columns]
    if missing:
        raise ValueError(f"Data missing required features: {missing}")
    return True


def export_results_to_excel(backtest_results, metrics, output_path):# Export backtest results and metrics to Excel file.
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


def create_performance_summary_table(metrics):# Create a formatted performance summary table.
    summary += "NATURAL GAS STRATEGY - PERFORMANCE SUMMARY\n"

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


def calculate_transaction_costs(signals, cost_per_trade=0.001):# Calculate transaction costs.
    trades = signals.diff().abs()
    costs = trades * cost_per_trade
    return costs

def calculate_model_comparison_table(fitted_models):# Calculate comparison table for fitted models.
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
