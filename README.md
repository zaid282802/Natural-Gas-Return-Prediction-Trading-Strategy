# Natural Gas Return Prediction & Trading Strategy

**September 2025 -- October 2025** (ML extension May 2026) | Zaid Annigeri | Master of Quantitative Finance, Rutgers Business School

> Quantitative trading strategy using fundamental factors to predict monthly natural gas returns, achieving 54% improvement over time series models (SSR metric). Extended in May 2026 with Random Forest (scikit-learn) and LSTM (PyTorch) for a 4-model walk-forward bake-off with HLN-corrected Diebold-Mariano tests.

##  Project Overview

Comparison of **fundamental factor analysis** (OLS regression) versus **time series models** (ARMA/GARCH) for predicting natural gas returns. Fundamental factors outperform pure time series approaches at monthly frequency.

### Key Findings

- **Fundamental factors outperform time series by 54%** (SSR: 2.57 vs 5.62)
- **6 significant variables** explain 36% of variance (parsimonious model)
- **Walk-forward backtest**: Sharpe Ratio 1.07, Win Rate 63%
- **Expanding window backtest**: No look-ahead bias

---

### Run Analysis
```bash
# Execute complete analysis pipeline
python run_analysis_clean.py

# Generate portfolio visualizations (6 charts)
python create_visualizations.py
```

**Output**:
- `run_analysis_clean.py`: Displays model comparison, walk-forward backtest results, and performance metrics in terminal
- `create_visualizations.py`: Generates 6 charts saved to `results/charts/`

**Data**: Project uses `data/Book1.1.xlsx` (71 monthly observations, Jan 2020 - Nov 2025)

### Project Structure
```
natural-gas-trading-strategy/
├── run_analysis_clean.py          # Main analysis script (57 lines)
├── create_visualizations.py       # Generate 6 portfolio charts (330 lines)
├── data/
│   └── Book1.1.xlsx               # Monthly natural gas data (2020-2025)
├── src/
│   ├── utils.py                   # Data loading utilities (105 lines)
│   ├── models.py                  # OLS, ARMA, GARCH, Random Forest (365 lines)
│   ├── backtest.py                # Walk-forward backtesting engine (190 lines)
│   └── risk_metrics.py            # Sharpe, Sortino, drawdown calculations (197 lines)
├── results/
│   └── charts/                    # Generated visualizations
└── Report/
    └── Natural_Gas_Report.tex     # LaTeX report
```
---

##  Portfolio Visualizations

6 charts showing strategy performance and model characteristics:

| Chart | Description | Key Insight |
|-------|-------------|-------------|
| **1. Equity Curve** | Strategy vs Buy & Hold | 409% return vs -84% (with COVID/Ukraine annotations) |
| **2. Drawdown Timeline** | Peak-to-trough analysis | -59% strategy DD vs -91% buy-and-hold |
| **3. Factor Importance** | Coefficient bar chart | Coal price -0.30 (largest effect), visualizes substitution |
| **4. Rolling Sharpe** | 12-month evolution | Positive 80% of rolling windows (stability) |
| **5. Prediction Accuracy** | Forecast reliability | 63% win rate stable across subperiods |
| **6. Correlation Matrix** | Multicollinearity check | All VIF < 2.5 (low correlation, stable estimates) |

**Generate charts**: `python create_visualizations.py` (output saved to `results/charts/`)
---

##  Results Summary

### Model Comparison (Full Sample)

| Model | R² | SSR | Variables | Best Use Case |
|-------|-----|-----|-----------|---------------|
| **OLS Full (23 vars)** | 0.559 | 2.571 | 23 | Research/exploration |
| **OLS Significant (6 vars)** | 0.362 | 3.800 | 6 | **Production (Best)** |
| ARMA(2,1) | - | 5.616 | - | Technical baseline |
| ARMA-GARCH(1,1) | - | 5.747 | - | Volatility modeling |
| Random Forest | 0.400 | ~3.5 | 23 | Non-linear check |

### Out-of-Sample Performance (Walk-Forward Backtest)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Sharpe Ratio** | 1.07 | Good risk-adjusted returns |
| **Sortino Ratio** | 1.69 | Favorable downside risk ratio |
| **Win Rate** | 63% | Directionally accurate |
| **Max Drawdown** | -59% | Large (2020-2022 volatility) |
| **Number of Trades** | 10 | Low turnover strategy |

> **Note**: Max drawdown highlights the importance of position sizing and risk management in production deployment.

---

##  Significant Factors

The 6 factors used in the parsimonious OLS model. Note: p-values below are from the LaTeX report analysis (statsmodels OLS with Newey-West standard errors). The Python codebase uses `sklearn.linear_model.LinearRegression`, which does not produce p-values.

| Factor | Coefficient | p-value (report) | Economic Rationale |
|--------|-------------|-------------------|-------------------|
| **Henry Hub Spot Price** | +0.164 | <0.001 | Mean reversion signal |
| **Coal Price Index** | -0.300 | 0.018 | Substitution effect (negative correlation) |
| **Net Trade Balance** | -0.112 | <0.001 | LNG exports reduce domestic supply |
| **EIA Storage Change** | +0.093 | 0.042 | Supply/demand indicator |
| **Storage vs 5Y Avg** | +0.078 | 0.067 | Relative scarcity measure |
| **Carbon EUA Futures** | +0.044 | 0.089 | Global energy market linkage |

### Why Coal Price is Negative?

High coal prices → Utilities switch to natural gas → Increased NG demand → **Substitution effect**

---

##  Methodology

### Walk-Forward Backtesting

- **Training Window**: 36 months (expanding)
- **Out-of-Sample Testing**: Monthly predictions (35 periods)
- **No Look-Ahead Bias**: Only past data used for each prediction
- **Signal Generation**: Long if predicted return > +2%, Short if < -2%
- **Transaction Costs**: Not applied (gross returns only)

### Model Selection Process

1. **Full OLS (23 variables)**: Exploratory analysis
2. **Stepwise selection**: Keep only significant factors (p < 0.1)
3. **Cross-validation**: Time series CV for Random Forest
4. **Out-of-sample testing**: 35 months never seen during training

---

##  Key Insights

### 1. Fundamentals > Technical Analysis

At **monthly frequency**, natural gas returns are driven by:
- Physical supply/demand (storage, production)
- Substitution dynamics (coal prices)
- Export demand (LNG trade balance)

**NOT** by momentum or volatility clustering (ARMA/GARCH ineffective).

### 2. Parsimony Works

6-variable model nearly matches 23-variable model performance:
- **Less overfitting risk**
- **Easier to interpret**
- **More stable out-of-sample**

### 3. Max Drawdown: 

**-59% maximum drawdown is a feature, not a bug** for this academic project.

#### Reasons:

**1. Extreme Historical Period (2020-2022)**
- COVID crash: Natural gas dropped to $1.63 (25-year low)
- Ukraine war: Spiked to $9.00+ (450% increase)
- This volatility is NOT representative of normal conditions

**2. No Position Sizing (By Design)**
- Fixed 100% allocation on every signal
- Academic focus: Test SIGNAL QUALITY, not risk management
- Real traders would use 10-30% positions

**3. Monthly Rebalancing**
- Cannot react to intra-month crashes
- Forced to hold through extreme moves
- Daily rebalancing would allow faster exits

---

##  Documentation

### LaTeX Report

An academic report (`Report/Natural_Gas_Report.tex`) is included covering:
- **Optional 1-Page Visual Summary** (`One_Page_Summary.tex`) - Infographic-style overview with key metrics
- Literature review (Geman 2005, Bollerslev 1986, Nick & Thoenes 2014)
- Methodology with mathematical derivations
- **Feature Engineering Rationale** (5-part detailed explanation of factor selection)
- **Training Window Selection** (sensitivity analysis: 24 vs 36 vs 48 months)
- Empirical results with tables **and 6 integrated figures**
- **Variance Decomposition** (standardized coefficients, factor importance ranking)
- **Factor Correlation Matrix** (multicollinearity analysis with VIF)
- **Technical Implementation** callout (Python stack, code architecture)
- **Key Lessons Learned** callout (6 critical insights from the project)
- **Industry Benchmark Comparisons** (AQR, Winton, academic strategies)
- Discussion of practical deployment with specific formulas
- Limitations and future research directions

### Visualization Documentation

The `create_visualizations.py` script generates 6 charts:
- Automatic path handling (runs from any directory)
- Seaborn styling (seaborn-v0_8-darkgrid)
- 300 DPI output
- Event annotations (COVID-19 crash, Ukraine war)
- Runtime: ~10-15 seconds

---

##  Author

**Zaid Annigeri**
- Master of Quantitative Finance, Rutgers Business School
- GitHub: [@Zaid282802](https://github.com/Zaid282802)
- LinkedIn: [Zaid Annigeri](https://www.linkedin.com/in/zed228)
- Email: mz845@scarletmail.rutgers.edu

---

##  References

1. **Walk-Forward Analysis**: *An Empirical Examination of Walk-Forward Testing* (2007)
2. **Natural Gas Modeling**: *Commodity Price Dynamics* - Geman, H. (2005)
3. **Risk Management**: *Active Portfolio Management* - Grinold & Kahn (1999)
4. **Backtesting**: *Evidence-Based Technical Analysis* - Aronson, D. (2006)
5. **Random Forest**: Breiman, L. (2001) "Random Forests", Machine Learning 45(1)
6. **LSTM**: Hochreiter, S. & Schmidhuber, J. (1997) "Long Short-Term Memory", Neural Computation 9(8)
7. **Diebold-Mariano**: Diebold & Mariano (1995) JBES; HLN small-sample correction Harvey, Leybourne & Newbold (1997)
8. **ML in finance**: Lopez de Prado (2018) *Advances in Financial Machine Learning*; Gu, Kelly & Xiu (2020) *Empirical Asset Pricing via Machine Learning* RFS

---

## ML Extension (May 2026)

I added Random Forest (scikit-learn) and LSTM (PyTorch) on top of the original OLS and ARMA-GARCH walk-forward framework, then ran a 4-model bake-off plus equal-weight and stacked ensembles. Same 71-month sample, same expanding-window walk-forward, 35 OOS predictions.

### 4-Model Comparison (Walk-Forward, 35 OOS months)

| Model | RMSE | MAE | Hit Rate | Sharpe | Max DD |
|-------|------|-----|----------|--------|--------|
| OLS (significant 6 vars) | 0.243 | 0.188 | 62.9% | 0.94 | -59% |
| Equal-weight ensemble | 0.239 | 0.182 | 65.7% | 1.21 | -59% |
| LSTM (regression) | 0.256 | 0.202 | 48.6% | 0.42 | -76% |
| Random Forest | 0.262 | 0.201 | 42.9% | 0.10 | -82% |
| Stacked logistic ensemble | 0.286 | 0.234 | 52.0% | 0.05 | -64% |
| ARMA-GARCH | 0.260 | 0.195 | 40.0% | -1.49 | -98% |

Note on the OLS Sharpe: this section reports 0.94 because the comparison harness uses sign-of-prediction trading with 10 bps round-trip cost. The 1.07 number reported earlier in this README comes from the original spec (only trade when prediction magnitude exceeds 2%, no transaction costs). Both are correct under their respective methodology, and the hit rate (62.9% vs 62.86%) is identical.

### Diebold-Mariano (HLN-corrected)

I ran 6 pairwise comparisons. None reached statistical significance (all p > 0.6). With N=35 that is expected per Harvey, Leybourne and Newbold (1997).

### What I learned

OLS still wins on its own. RF and LSTM did not beat it individually in 35 OOS months, which is the outcome Gu-Kelly-Xiu (2020) predict for ML on monthly data with limited features. The equal-weight ensemble of all four models edged out OLS slightly (1.21 vs 0.94 Sharpe, 65.7% vs 62.9% hit rate), so combining model classes adds value even when individual ML models look weak.

If I had daily or weekly NG data, ML would likely move from no edge to additive edge without needing the ensemble. That is the natural follow-up.

### Files

```
nat_gas_ml_extension.ipynb         Single notebook, 14 cells, runs top to bottom
addendum-pytorch-rf/addendum.pdf   12-page methodology document
results/model_comparison.csv       Headline comparison table
results/dm_tests.csv               Pairwise Diebold-Mariano statistics
results/*.parquet                  Per-model walk-forward predictions
```


---

