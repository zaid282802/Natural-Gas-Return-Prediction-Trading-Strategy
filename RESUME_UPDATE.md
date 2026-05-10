# Resume Update — Nat Gas ML Extension

Paste this into `Zaid_Annigeri_Resume.docx`.

## 1. Skills section — add PyTorch + sklearn

**Current:**
```
Programming: Python (Scikit-learn, Statsmodels, Arch, Pandas, NumPy, SciPy), R, MATLAB, SQL
```

**Updated:**
```
Programming: Python (PyTorch, Scikit-learn, Statsmodels, Arch, Pandas, NumPy, SciPy), R, MATLAB, SQL
```

(Added `PyTorch` at the front of the Python list. Scikit-learn was already there.)

Optional sub-line under Skills for ML emphasis:
```
Machine Learning: Random Forest, LSTM, walk-forward CV, ensemble methods, Diebold-Mariano forecast comparison
```

## 2. Nat Gas project bullet — TWO-bullet replacement

**Current (1 bullet, OLS only):**
> Natural Gas Return Prediction & Trading Strategy | Time Series | Python    Sep - Oct 2025
> • Developed OLS framework comparing fundamental factors (storage, coal prices, trade balance) vs ARMA-GARCH models; OLS model achieved ~55% superior accuracy (SSR 2.57 vs 5.68) with 6 significant variables (p<0.1).
> • Validated via 71-month walk-forward backtest yielding 1.07 Sharpe and 63% directional accuracy; identified position sizing critical for managing -59% max drawdown during 2020-2022 volatility. [GitHub]

**Replace with (2 bullets, ML extension added):**

```
Natural Gas Return Prediction & Trading Strategy | Time Series + ML | Python    Sep - Oct 2025; ML extension May 2026
• Developed OLS framework comparing fundamental factors (storage, coal prices, trade balance) vs
  ARMA-GARCH; OLS achieved ~55% superior accuracy (SSR 2.57 vs 5.68) with 6 significant variables
  (p<0.1); 71-month walk-forward backtest yielded 1.07 Sharpe and 63% directional accuracy; -59%
  max drawdown during 2020-2022 volatility highlighted position-sizing as critical.
• Extended with Random Forest (sklearn) and LSTM (PyTorch); 4-model walk-forward bake-off with
  HLN-corrected Diebold-Mariano tests confirmed OLS retains edge in 35-OOS-month sample (Sharpe
  0.94 vs RF 0.10 vs LSTM -0.10); equal-weight ensemble Sharpe 0.68. [GitHub]
```

This is the **honest** version: it shows you tested ML models AND it shows OLS won. That's a stronger story than fake-claiming ML beat OLS — recruiters who know the literature (Gu-Kelly-Xiu 2020) will recognize that 35 OOS months is too small for ML to extract edge over a well-specified linear model, and they'll respect the honesty.

## Character counts

Bullet 1: ~330 chars (within your 300-350 budget)
Bullet 2: ~315 chars (within budget)
Combined: 2 lines visually on resume per bullet, 4 lines total for the project.

## Total resume changes

1. Skills line: add `PyTorch` at front of Python parenthetical
2. Project bullets: replace 1 OLS-only bullet with 2 bullets (OLS + ML extension)
3. Date range header: append "; ML extension May 2026"
4. (Optional) Add ML sub-line under Skills

After these changes, recompile resume to PDF as `Annigeri_DerivativesResearch1.pdf` (or your latest filename convention).
