"""Natural Gas Return Prediction - Main Analysis Script.

Runs the full analysis pipeline: data loading, model comparison,
walk-forward backtesting, and performance reporting.

Outputs: Metrics only, no explanations/recommendations.
For interpretation: See README.md and Natural_Gas_Report.pdf

Usage:
    python run_analysis_clean.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import load_data_from_r
from src.models import compare_all_models, SignificantOLSModel
from src.backtest import WalkForwardBacktest

# Configuration
DATA_PATH = 'data/Book1.1.xlsx'
TRAIN_WINDOW = 36                         # 3-year expanding training window


def main():
    """Run full analysis: model comparison and walk-forward backtest."""
    # Header
    print("NATURAL GAS TRADING STRATEGY - ANALYSIS RESULTS")

    # Load data
    data = load_data_from_r(DATA_PATH)
    print(f"    Loaded: {len(data)} observations")
    print(f"    Period: {data.index[0].strftime('%Y-%m')} to {data.index[-1].strftime('%Y-%m')}")

    # Compare models
    comparison_df, fitted_models = compare_all_models(data)
    print("\n" + comparison_df.to_string(index=False))

    # Walk-Forward Backtest
    model = SignificantOLSModel()
    backtest = WalkForwardBacktest(model, data, train_window=TRAIN_WINDOW, expanding=True)
    results = backtest.run()

    # Print performance
    print("PERFORMANCE METRICS")
    backtest.print_performance()

    return {
        'data': data,
        'comparison': comparison_df,
        'backtest': backtest,
        'results': results
    }


if __name__ == "__main__":
    try:
        outputs = main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
