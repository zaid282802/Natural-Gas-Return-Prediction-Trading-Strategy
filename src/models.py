import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


class BaseOLSModel:
    def __init__(self, features):
        self.features = features
        self.model = LinearRegression()
        self.fitted = False

    def fit(self, data):
        X = data[self.features]
        y = data['NG_Return']

        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        # Calculate metrics
        self.r2 = r2_score(y, y_pred)
        n = len(y)
        p = len(self.features)
        self.adj_r2 = 1 - (1 - self.r2) * (n - 1) / (n - p - 1)

        residuals = y - y_pred
        self.ssr = np.sum(residuals ** 2)
        self.rmse = np.sqrt(mean_squared_error(y, y_pred))

        self.coefficients = pd.DataFrame({
            'Variable': ['Intercept'] + self.features,
            'Coefficient': [self.model.intercept_] + list(self.model.coef_)
        })

        self.fitted = True
        return self

    def predict(self, data):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(data[self.features])

    def summary(self):
        print(f"{self.__class__.__name__} - OLS REGRESSION")
        print(f"Number of features:  {len(self.features)}")
        print(f"R²:                  {self.r2:.4f}")
        print(f"Adjusted R²:         {self.adj_r2:.4f}")
        print(f"RMSE:                {self.rmse:.4f}")
        print(f"SSR:                 {self.ssr:.4f}")
        print("\nCoefficients:")
        print(self.coefficients.to_string(index=False))
        


class FullOLSModel(BaseOLSModel):# Full OLS with all 23 variables.

    def __init__(self):
        features = [
            'WTI Crude Oil Return ', 'Heating Oil Return', 'Electricity LMP PJM WesternHub',
            'Coal Price Index', 'Carbon EUA Futures Price', 'HDD ', 'NG IV Percentile',
            'VIX ', 'EIA Storage Change', 'Storage vs 5yr Avg ', 'NG Production Volume',
            'LNG Export Volume', 'Net Trade Balance', 'Seasonal Dummy Winter 1 Summer 0 ',
            'Treasury Yield 10yr ', 'Fed Funds Effective Rate', 'Dollar Index',
            'Inflation Expectation', 'Credit Spread ', 'HenryHub Spot',
            'Industrial Production ', 'Propane Price', 'CDD'
        ]
        super().__init__(features)


class SignificantOLSModel(BaseOLSModel):# OLS with 6 significant variables.
    def __init__(self):
        features = [
            'Coal Price Index',
            'Carbon EUA Futures Price',
            'EIA Storage Change',
            'Storage vs 5yr Avg ',
            'Net Trade Balance',
            'HenryHub Spot'
        ]
        super().__init__(features)


class ARMAModel: # ARMA model for time series forecasting.
    def __init__(self, order=(0,0,0)):
        self.order = order
        self.fitted = False

    def fit(self, returns):
        self.model = ARIMA(returns, order=self.order)
        self.fitted_model = self.model.fit()

        residuals = self.fitted_model.resid
        self.ssr = np.sum(residuals ** 2)
        self.aic = self.fitted_model.aic
        self.bic = self.fitted_model.bic

        self.fitted = True
        return self

    def forecast(self, steps=6):
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        return self.fitted_model.forecast(steps=steps)

    def summary(self):
        
        print(f"ARMA{self.order} MODEL")
        
        print(f"SSR: {self.ssr:.4f}")
        print(f"AIC: {self.aic:.4f}")
        print(f"BIC: {self.bic:.4f}")
        
class GARCHModel:# GARCH model for volatility forecasting.
    def __init__(self, mean_model='AR', lags=1, p=1, q=1):
        self.mean_model = mean_model
        self.lags = lags
        self.p = p
        self.q = q
        self.fitted = False

    def fit(self, returns, dist='normal'):
        returns_scaled = returns * 100

        model = arch_model(
            returns_scaled,
            mean=self.mean_model,
            lags=self.lags,
            vol='GARCH',
            p=self.p,
            q=self.q,
            dist=dist
        )

        self.fitted_model = model.fit(disp='off')

        residuals = self.fitted_model.resid / 100
        self.ssr = np.sum(residuals ** 2)

        self.fitted = True
        return self

    def forecast(self, horizon=6):
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")

        forecast = self.fitted_model.forecast(horizon=horizon)
        mean_forecast = forecast.mean.values[-1, :] / 100
        variance_forecast = forecast.variance.values[-1, :] / 10000

        return mean_forecast, variance_forecast

    def summary(self):
        
        print(f"ARMA({self.lags},1)-GARCH({self.p},{self.q}) MODEL")
        
        print(f"SSR: {self.ssr:.4f}")
        


class MLEnsembleModel:# Machine Learning model using Random Forest.
    def __init__(self, n_estimators=100, max_depth=5):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        self.features = [
            'HenryHub Spot',
            'Coal Price Index',
            'EIA Storage Change',
            'Net Trade Balance'
        ]

        self.fitted = False

    def fit(self, data):
        X = data[self.features]
        y = data['NG_Return']

        self.model.fit(X, y)

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=tscv,
            scoring='r2',
            n_jobs=-1
        )

        self.cv_r2_mean = cv_scores.mean()
        self.cv_r2_std = cv_scores.std()

        # In-sample metrics
        y_pred = self.model.predict(X)
        self.r2 = r2_score(y, y_pred)
        self.ssr = np.sum((y - y_pred) ** 2)
        self.rmse = np.sqrt(mean_squared_error(y, y_pred))

        self.feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        self.fitted = True
        return self

    def predict(self, data):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(data[self.features])

    def summary(self):
        
        print("RANDOM FOREST MODEL")
        
        print(f"In-sample R²:        {self.r2:.4f}")
        print(f"CV R² (mean):        {self.cv_r2_mean:.4f}")
        print(f"CV R² (std):         {self.cv_r2_std:.4f}")
        print(f"RMSE:                {self.rmse:.4f}")
        print(f"SSR:                 {self.ssr:.4f}")
        print("\nFeature Importance:")
        for _, row in self.feature_importance.iterrows():
            print(f"  {row['Feature']:<30} {row['Importance']:>10.4f}")
        


def compare_all_models(data):# Comprehensive model comparison function.
    print("COMPREHENSIVE MODEL COMPARISON")
    results = []
    fitted_models = {}

    # OLS Models
    print("\n[1/7] Fitting Full OLS Model (23 variables)...")
    full_ols = FullOLSModel()
    full_ols.fit(data)
    results.append({
        'Model': 'OLS - Full (23 vars)',
        'Type': 'Linear Regression',
        'SSR': full_ols.ssr,
        'R²': full_ols.r2,
        'Adj R²': full_ols.adj_r2,
        'RMSE': full_ols.rmse
    })
    fitted_models['full_ols'] = full_ols

    print("[2/7] Fitting Significant OLS Model (6 variables)...")
    sig_ols = SignificantOLSModel()
    sig_ols.fit(data)
    results.append({
        'Model': 'OLS - Significant (6 vars)',
        'Type': 'Linear Regression',
        'SSR': sig_ols.ssr,
        'R²': sig_ols.r2,
        'Adj R²': sig_ols.adj_r2,
        'RMSE': sig_ols.rmse
    })
    fitted_models['sig_ols'] = sig_ols

    # Machine Learning
    print("[3/7] Fitting Random Forest Model...")
    ml_model = MLEnsembleModel()
    ml_model.fit(data)
    results.append({
        'Model': 'Random Forest',
        'Type': 'Machine Learning',
        'SSR': ml_model.ssr,
        'R²': ml_model.r2,
        'Adj R²': None,
        'RMSE': ml_model.rmse
    })
    fitted_models['ml_rf'] = ml_model

    # Time Series Models
    returns = data['NG_Return']

    print("[4/7] Fitting ARMA(0,0)...")
    arma_00 = ARMAModel(order=(0,0,0))
    arma_00.fit(returns)
    results.append({
        'Model': 'ARMA(0,0)',
        'Type': 'Time Series',
        'SSR': arma_00.ssr,
        'R²': None,
        'Adj R²': None,
        'RMSE': None
    })
    fitted_models['arma_00'] = arma_00

    print("[5/7] Fitting ARMA(1,1)...")
    arma_11 = ARMAModel(order=(1,0,1))
    arma_11.fit(returns)
    results.append({
        'Model': 'ARMA(1,1)',
        'Type': 'Time Series',
        'SSR': arma_11.ssr,
        'R²': None,
        'Adj R²': None,
        'RMSE': None
    })
    fitted_models['arma_11'] = arma_11

    print("[6/7] Fitting ARMA(2,1)...")
    arma_21 = ARMAModel(order=(2,0,1))
    arma_21.fit(returns)
    results.append({
        'Model': 'ARMA(2,1)',
        'Type': 'Time Series',
        'SSR': arma_21.ssr,
        'R²': None,
        'Adj R²': None,
        'RMSE': None
    })
    fitted_models['arma_21'] = arma_21

    print("[7/7] Fitting ARMA(1,1)-GARCH(1,1)...")
    garch = GARCHModel(lags=1, p=1, q=1)
    garch.fit(returns, dist='normal')
    results.append({
        'Model': 'ARMA(1,1)-GARCH(1,1)',
        'Type': 'Volatility',
        'SSR': garch.ssr,
        'R²': None,
        'Adj R²': None,
        'RMSE': None
    })
    fitted_models['garch'] = garch

    # Create comparison table
    comparison_df = pd.DataFrame(results).sort_values('SSR')

    baseline_ssr = comparison_df[comparison_df['Model'] == 'ARMA(0,0)']['SSR'].values[0]
    comparison_df['vs_Baseline'] = (
        (baseline_ssr - comparison_df['SSR']) / baseline_ssr * 100
    ).round(2).astype(str) + '%'

    print("\n" + "="*70)
    print("FINAL MODEL RANKINGS (Sorted by SSR - Lower is Better)")
    
    print(comparison_df.to_string(index=False))
    print("\n" + "="*70)

    # Key findings
    best_model = comparison_df.iloc[0]
    print(f"\nBest Model: {best_model['Model']}")
    r2_str = f"{best_model['R²']:.4f}" if best_model['R²'] is not None and not pd.isna(best_model['R²']) else 'N/A'
    print(f"SSR: {best_model['SSR']:.4f} | R²: {r2_str}")

    ols_ssr = comparison_df[comparison_df['Model'] == 'OLS - Full (23 vars)']['SSR'].values[0]
    best_ts_ssr = comparison_df[comparison_df['Model'] == 'ARMA(2,1)']['SSR'].values[0]
    improvement = (best_ts_ssr - ols_ssr) / best_ts_ssr * 100
    print(f"\nOLS outperforms ARMA(2,1) by {improvement:.1f}%")
    

    return comparison_df, fitted_models
