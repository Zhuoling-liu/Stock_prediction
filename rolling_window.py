import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Import the data engine we built on Day 1
from data_engine import get_cleaned_data

class PredictiveEngine:
    """
    【Predictive Core - Rolling Window Edition】
    The AI brain of the system.
    Implements:
    1. Hybrid Prediction (Ridge + Gradient Boosting)
    2. Walk-Forward Validation (Rolling Window) per Huang et al. (2026) methodology.
    """
    
    def __init__(self):
        # 1. Linear Model (Ridge) - Captures linear trends
        self.ridge_model = Ridge(alpha=1.0)
        
        # 2. Non-linear Model (Gradient Boosting) - Captures complex patterns
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=42
        )
        
        # 3. Regime Classifier - Detects high volatility states
        self.regime_clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        
        self.model_metrics = {}
        self.is_fitted = False

    def _generate_regime_labels(self, X):
        """
        Label High Volatility Regimes (Top 25% volatility).
        """
        volatility = X['vol_20d']
        threshold = volatility.quantile(0.75)
        return (volatility > threshold).astype(int)

    def train_rolling_window(self, X, y, n_splits=5):
        """
        Performs rigorous Walk-Forward Validation (Rolling Window) to calculate RMSE.
        Then retrains on FULL data for final deployment.
        """
        print(f"⚡ Starting Rolling Window Validation ({n_splits} folds)...")
        print("   (This mimics the '3-year train / 1-year test' methodology in the paper)")

        

        # 1. Data Cleaning
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 2. Initialize Rolling Window Validator
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        ridge_scores = []
        gb_scores = []
        hybrid_scores = []
        
        fold = 1
        # 3. Start Walk-Forward Loop
        for train_index, val_index in tscv.split(X):
            # Split data
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            # --- Train Current Window ---
            # A. Ridge
            self.ridge_model.fit(X_train, y_train)
            p_ridge = self.ridge_model.predict(X_val)
            rmse_ridge = np.sqrt(mean_squared_error(y_val, p_ridge))
            ridge_scores.append(rmse_ridge)
            
            # B. Gradient Boosting
            self.gb_model.fit(X_train, y_train)
            p_gb = self.gb_model.predict(X_val)
            rmse_gb = np.sqrt(mean_squared_error(y_val, p_gb))
            gb_scores.append(rmse_gb)
            
            # C. Hybrid Ensemble
            p_hybrid = 0.5 * p_ridge + 0.5 * p_gb
            rmse_hybrid = np.sqrt(mean_squared_error(y_val, p_hybrid))
            hybrid_scores.append(rmse_hybrid)
            
            print(f"   Fold {fold}/{n_splits}: Hybrid RMSE = {rmse_hybrid:.5f}")
            fold += 1
            
        # 4. Calculate Average Out-of-Sample Performance
        avg_ridge_rmse = np.mean(ridge_scores)
        avg_gb_rmse = np.mean(gb_scores)
        avg_hybrid_rmse = np.mean(hybrid_scores)
        
        self.model_metrics = {
            'ridge_rmse': avg_ridge_rmse,
            'gb_rmse': avg_gb_rmse,
            'hybrid_rmse': avg_hybrid_rmse
        }
        
        print(f"✅ Rolling Validation Complete.")
        print(f"   Avg Ridge RMSE    : {avg_ridge_rmse:.5f}")
        print(f"   Avg Hybrid RMSE   : {avg_hybrid_rmse:.5f} (Robust Estimate)")
        
        # 5. [Crucial Step]: Retrain on ALL data for deployment
        # We must do this to predict "tomorrow's" price using the latest data
        print("🔄 Retraining final model on 100% of available data for deployment...")
        self.ridge_model.fit(X, y)
        self.gb_model.fit(X, y)
        
        # Train Regime Classifier
        regime_labels = self._generate_regime_labels(X)
        self.regime_clf.fit(X, regime_labels)
        
        self.is_fitted = True

    def predict(self, X):
        """
        Returns: Expected Returns, Uncertainty, Market Regimes
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet!")
            
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
        # 1. Hybrid Prediction
        p_ridge = self.ridge_model.predict(X)
        p_gb = self.gb_model.predict(X)
        expected_returns = 0.5 * p_ridge + 0.5 * p_gb
        
        # 2. Regime Detection
        regimes = self.regime_clf.predict(X)
        
        # 3. Dynamic Uncertainty (Risk)
        # Use average RMSE from rolling validation as baseline risk
        base_error = self.model_metrics['hybrid_rmse']
        
        # If High Volatility Regime is detected, increase uncertainty estimate by 50%
        uncertainties = np.where(regimes == 1, base_error * 1.5, base_error)
        
        return expected_returns, uncertainties, regimes

# --- Execution Block ---
if __name__ == "__main__":
    # 1. Fetch Data (Day 1)
    X, y, df = get_cleaned_data()
    
    if X is not None:
        agent = PredictiveEngine()
        
        # Train using Rolling Window
        agent.train_rolling_window(X, y, n_splits=5)
        
        print("\n🔮 Generating Forecasts for the most recent date:")
        
        # Get latest data for prediction
        latest_full_data = df.groupby('ticker').tail(1)
        feature_cols = X.columns
        sample_X = latest_full_data[feature_cols]
        sample_X = sample_X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Run prediction
        rets, errs, regimes = agent.predict(sample_X)
        
        results = pd.DataFrame({
            'Ticker': latest_full_data['ticker'].values,
            'Pred_Return': rets,
            'Uncertainty': errs,
            'Regime': regimes
        })
        
        print(results)