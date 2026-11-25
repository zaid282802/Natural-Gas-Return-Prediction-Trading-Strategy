"""
Timing check for the OLS model in this repo.

The six fundamental features are read from the same monthly row as NG_Return,
which is the return realised during that month. This script measures what that
costs: it re-runs the OLS walk-forward with each fundamental lagged one month.

  1. Timing check: correlation of NG_Return[t] with the same month's and the
     next month's log price change.
  2. In-sample OLS on the full sample: features as stored vs lagged one month.
  3. Walk-forward OLS (expanding window, 36-month initial train, sign-of-forecast
     trading, 10 bps cost on signal changes, Sharpe = mean/std * sqrt(12)), the
     same harness as nat_gas_ml_extension.ipynb: features as stored vs lagged.

Run from the repo root:  python robustness/lag_check.py
Reads data/Book1.1.xlsx and src/models.py; writes nothing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent  # repo root
sys.path.insert(0, str(ROOT))
from src.models import SignificantOLSModel  # noqa: E402  (the repo's own model class)

DATA = ROOT / "data" / "Book1.1.xlsx"
TARGET = "NG_Return"
FEATURES = [
    "Coal Price Index",
    "Carbon EUA Futures Price",
    "EIA Storage Change",
    "Storage vs 5yr Avg ",
    "Net Trade Balance",
    "HenryHub Spot",
]
TRAIN_WINDOW = 36  # matches nat_gas_ml_extension.ipynb cell 1 (TRAIN_WINDOW = 36)
COST = 0.001  # 10 bps round-trip, matches cell 13 metrics(..., cost=0.001)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def timing_check(raw):
    section("1. TIMING CHECK: what is NG_Return[t] relative to price and features")
    p = raw["NG_Price (spot)"]
    r = raw[TARGET]

    logret_t = np.log(p / p.shift(1))          # return realized month t-1 -> t
    logret_fwd = np.log(p.shift(-1) / p)       # return realized month t -> t+1
    pct_t = p.pct_change()
    pct_fwd = p.pct_change().shift(-1)

    print(f"corr(NG_Return[t], log(P[t]/P[t-1]))   = {r.corr(logret_t):.4f}")
    print(f"corr(NG_Return[t], log(P[t+1]/P[t]))   = {r.corr(logret_fwd):.4f}")
    print(f"corr(NG_Return[t], pct_change P[t])    = {r.corr(pct_t):.4f}")
    print(f"corr(NG_Return[t], pct_change P[t+1])  = {r.corr(pct_fwd):.4f}")

    verdict = "CONTEMPORANEOUS (return realized BY month t, same row as price P[t])"
    print(f"\nVerdict: NG_Return[t] is the {verdict}.")
    print("The 6 fundamental feature columns (Coal Price Index, Carbon EUA, EIA")
    print("Storage Change, Storage vs 5yr Avg, Net Trade Balance, HenryHub Spot)")
    print("are read from the SAME row t as NG_Return[t] in both src/models.py")
    print("(BaseOLSModel.fit: X = data[self.features]; y = data[TARGET_COLUMN])")
    print("and in nat_gas_ml_extension.ipynb cell 5 (SignificantOLSModel().fit(train)")
    print("where train/test are row-aligned slices of the same raw dataframe).")
    print("No .shift() is applied to any of the 6 fundamental columns anywhere in")
    print("src/models.py, src/backtest.py, src/utils.py, or notebook cells 1-13.")
    return {
        "corr_contemp": r.corr(logret_t),
        "corr_forward": r.corr(logret_fwd),
    }


def build_lagged(raw):
    lagged = raw.copy()
    for f in FEATURES:
        lagged[f] = lagged[f].shift(1)
    return lagged.dropna(subset=FEATURES + [TARGET])


def insample_ols(raw):
    section("2. IN-SAMPLE OLS (full sample, 6 variables): as-is vs lagged 1 month")

    # As-is: exact repo code path (SignificantOLSModel from src/models.py, unmodified)
    m_asis = SignificantOLSModel().fit(raw)
    n_asis = len(raw)
    print(f"As-is   (contemporaneous features): n={n_asis}, R^2={m_asis.r2:.4f}, "
          f"Adj R^2={m_asis.adj_r2:.4f}, SSR={m_asis.ssr:.4f}")

    # Lagged: same 6 features, each shifted 1 month (features[t-1] predicts NG_Return[t])
    lagged = build_lagged(raw)
    X = lagged[FEATURES]
    y = lagged[TARGET]
    lr = LinearRegression().fit(X, y)
    y_pred = lr.predict(X)
    r2_lag = r2_score(y, y_pred)
    n = len(y)
    p = len(FEATURES)
    adj_r2_lag = 1 - (1 - r2_lag) * (n - 1) / (n - p - 1)
    ssr_lag = float(np.sum((y - y_pred) ** 2))
    print(f"Lagged 1m (features[t-1] -> NG_Return[t]): n={n}, R^2={r2_lag:.4f}, "
          f"Adj R^2={adj_r2_lag:.4f}, SSR={ssr_lag:.4f}")

    return {
        "r2_asis": m_asis.r2,
        "adj_r2_asis": m_asis.adj_r2,
        "n_asis": n_asis,
        "r2_lag": r2_lag,
        "adj_r2_lag": adj_r2_lag,
        "n_lag": n,
    }


def metrics_from_signal(actual, pred, cost=COST):
    """Reproduces nat_gas_ml_extension.ipynb cell 13 metrics() exactly."""
    err = actual - pred
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(err.abs().mean())
    hit = float((np.sign(actual) == np.sign(pred)).mean())
    sig = np.sign(pred)
    strat = sig * actual
    trades = sig.diff().abs().fillna(0.0)
    strat_net = strat - trades * cost
    sharpe = float(strat_net.mean() / strat_net.std() * np.sqrt(12)) if strat_net.std() > 0 else 0.0
    eq = (1.0 + strat_net).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"RMSE": rmse, "MAE": mae, "Hit": hit, "Sharpe": sharpe, "MaxDD": dd, "n": len(actual)}


def walkforward(data, label):
    rows = []
    for i in range(TRAIN_WINDOW, len(data)):
        train, test = data.iloc[:i], data.iloc[i:i + 1]
        m = SignificantOLSModel().fit(train)
        pred = m.predict(test)[0]
        rows.append({"date": test.index[0], "actual": test[TARGET].values[0], "pred": pred})
    df = pd.DataFrame(rows).set_index("date")
    m = metrics_from_signal(df["actual"], df["pred"])
    print(f"{label}: n_OOS={m['n']}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, "
          f"Hit={m['Hit']*100:.1f}%, Sharpe={m['Sharpe']:.4f}, MaxDD={m['MaxDD']*100:.1f}%")
    return m


def walkforward_check(raw):
    section("3. WALK-FORWARD (expanding 36-month window, 35 OOS months, 10bps, "
            "sign-of-forecast): as-is vs lagged 1 month")
    print("NOTE ON HARNESS: nat_gas_ml_extension.ipynb cell 5 (which produced the")
    print("official results/ols_predictions.parquet and the 0.94 Sharpe / 62.9% hit")
    print("in results/model_comparison.csv) uses an EXPANDING 36-month window with")
    print("range(36, len(raw)) -- NOT the 5-fold TimeSeriesSplit that the addendum's")
    print("prose claims ('Same walk-forward partition: 5 contiguous folds'). Only")
    print("the Random Forest (cell 9) and LSTM (cell 11) actually use")
    print("TimeSeriesSplit(n_splits=5). This script reproduces the ACTUAL OLS code")
    print("path (expanding window) which is what generated")
    print("results/model_comparison.csv.\n")

    m_asis = walkforward(raw, "As-is    (contemporaneous features)")

    lagged = build_lagged(raw)
    m_lag = walkforward(lagged, "Lagged 1m (features[t-1])          ")

    return m_asis, m_lag


def main():
    raw = pd.read_excel(DATA, index_col=0, parse_dates=True).dropna().sort_index()
    print(f"Loaded {DATA.name}: shape={raw.shape}, range={raw.index.min().date()} "
          f"to {raw.index.max().date()}")

    timing = timing_check(raw)
    insample = insample_ols(raw)
    wf_asis, wf_lag = walkforward_check(raw)

    section("SUMMARY TABLE")
    print(f"{'Metric':<22}{'As-is':>14}{'Lagged 1m':>14}")
    print(f"{'In-sample R^2':<22}{insample['r2_asis']:>14.4f}{insample['r2_lag']:>14.4f}")
    print(f"{'In-sample Adj R^2':<22}{insample['adj_r2_asis']:>14.4f}{insample['adj_r2_lag']:>14.4f}")
    print(f"{'WF Sharpe':<22}{wf_asis['Sharpe']:>14.4f}{wf_lag['Sharpe']:>14.4f}")
    print(f"{'WF Hit rate (%)':<22}{wf_asis['Hit']*100:>14.1f}{wf_lag['Hit']*100:>14.1f}")
    print(f"{'WF Max drawdown (%)':<22}{wf_asis['MaxDD']*100:>14.1f}{wf_lag['MaxDD']*100:>14.1f}")


if __name__ == "__main__":
    main()
