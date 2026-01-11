import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error

# Assuming data_engine is imported as before
from data_engine import get_cleaned_data

class PredictiveEngine:
    """
    The AI Brain (Robust & Leakage-Proof)
    
    Key Features:
    1. Global Model Support: Handles mixed tickers via strict time-sorting.
    2. Date-Aligned Splitting: Prevents intra-day leakage (same day tickers split between train/val).
    3. Gap Embargo: Enforces a strict 20-trading-day gap between training and validation sets.
    4. Auto-Tuning: Finds optimal hyperparameters without look-ahead bias.
    """
    
    def __init__(self):
        # Initialize base models (parameters will be optimized by tune_hyperparameters)
        self.ridge_model = Ridge(alpha=1.0)
        
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=3, 
            random_state=42
        )
        
        self.regime_clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        
        self.model_metrics = {}
        self.is_fitted = False

    def _generate_regime_labels(self, X):
        """Label High Volatility Regimes (Top 20% volatility)."""
        volatility = X['vol_20d']
        threshold = volatility.quantile(0.80)
        return (volatility > threshold).astype(int)

    def _date_aligned_split(self, X, n_splits=5, gap_days=20):
        """
        🔥 Core Logic: Custom Time Series Splitter based on DATES, not ROWS.
        
        This solves two critical leakage problems in Panel Data:
        1. Intra-day Leakage: Ensures all tickers for the same date are ALWAYS kept together 
           (either all in Train or all in Val).
        2. Gap Units: 'gap_days' now correctly represents 'trading days', 
           eliminating the need to guess row counts.
        
        Yields:
            (train_indices, val_indices): Generator of integer indices for iloc.
        """
        # 1. Extract unique trading dates and sort them chronologically
        unique_dates = np.sort(X.index.unique())
        
        # 2. Apply TimeSeriesSplit on the DATE array
        tscv_dates = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)
        
        # 3. Iterate through date splits
        for train_date_idx, val_date_idx in tscv_dates.split(unique_dates):
            # Map date indices back to actual date values
            train_dates = unique_dates[train_date_idx]
            val_dates = unique_dates[val_date_idx]
            
            # 4. [CRITICAL STEP] Map dates back to row indices in the original DataFrame
            # X.index.isin(...) creates a boolean mask, which we convert to integer indices.
            # This captures ALL rows (tickers) belonging to the selected dates.
            train_indices = np.where(X.index.isin(train_dates))[0]
            val_indices = np.where(X.index.isin(val_dates))[0]
            
            yield train_indices, val_indices

    def tune_hyperparameters(self, X, y, n_iter=10):
        """
        Performs Hyperparameter Tuning using Date-Aligned Cross-Validation.
        """
        print(f"\n⚙️ Starting Auto-Tuning (Date-Aligned, Gap=20 days)...")

        # 1. Force Sort & Align
        # Essential to prevent index mismatch errors during concatenation
        if not X.index.is_monotonic_increasing:
            print("   ⚠️ Data not sorted! Sorting combined X and y...")
            combined = pd.concat([X, y], axis=1).sort_index()
            X = combined[X.columns]
            y = combined.iloc[:, -1]

        # 2. 🔥 Create the Custom CV Generator
        # We convert it to a list to pass directly into GridSearchCV
        custom_cv = list(self._date_aligned_split(X, n_splits=3, gap_days=20))

        # --- A. Tune Ridge (Grid Search) ---
        print("   🔹 Tuning Ridge Regression...")
        ridge_search = GridSearchCV(
            estimator=Ridge(),
            param_grid={'alpha': [0.1, 1.0, 10.0, 50.0]},
            cv=custom_cv,  # <--- Inject custom splitter here
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        ridge_search.fit(X, y)
        self.ridge_model = ridge_search.best_estimator_
        print(f"      ✅ Best Ridge Alpha: {ridge_search.best_params_}")

        # --- B. Tune Gradient Boosting (Randomized Search) ---
        print("   🔹 Tuning Gradient Boosting...")
        gb_search = RandomizedSearchCV(
            estimator=GradientBoostingRegressor(random_state=42),
            param_distributions={
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 4]
            },
            n_iter=n_iter,
            cv=custom_cv,  # <--- Inject custom splitter here
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42
        )
        gb_search.fit(X, y)
        self.gb_model = gb_search.best_estimator_
        print(f"      ✅ Best GB Params: {gb_search.best_params_}")

    def train_rolling_window(self, X, y, n_splits=5):
        """
        Performs rigorous Walk-Forward Validation using the Date-Aligned Splitter.
        """
        print(f"\n📉 Starting Rolling Window Validation (Date-Aligned)...")
        
        # 1. Ensure Data is Sorted
        if not X.index.is_monotonic_increasing:
            combined = pd.concat([X, y], axis=1).sort_index()
            X = combined[X.columns]
            y = combined.iloc[:, -1]

        # 2. 🔥 Initialize Custom CV Generator
        custom_cv = self._date_aligned_split(X, n_splits=n_splits, gap_days=20)
        
        hybrid_scores = []
        fold = 1
        
        # 3. Iterate through the custom splits
        for train_indices, val_indices in custom_cv:
            # Use iloc to slice by the integer indices we calculated
            X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
            y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
            
            # Train models on the current window
            self.ridge_model.fit(X_train, y_train)
            self.gb_model.fit(X_train, y_train)
            
            # Predict
            p_hybrid = 0.5 * self.ridge_model.predict(X_val) + \
                       0.5 * self.gb_model.predict(X_val)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_val, p_hybrid))
            hybrid_scores.append(rmse)
            
            print(f"   Fold {fold}: Hybrid RMSE = {rmse:.5f}")
            fold += 1
            
        # 4. Store Baseline Uncertainty
        self.model_metrics['hybrid_rmse'] = np.mean(hybrid_scores)
        print(f"✅ Robust RMSE: {self.model_metrics['hybrid_rmse']:.5f}")
        
        # 5. Full Retrain for Deployment
        print("🔄 Retraining final models on ALL historical data...")
        self.ridge_model.fit(X, y)
        self.gb_model.fit(X, y)
        self.is_fitted = True
        
        # Train Regime Classifier (using full dataset)
        regime_labels = self._generate_regime_labels(X)
        self.regime_clf.fit(X, regime_labels)

    def predict(self, X):
        """
        Generates predictions with uncertainty estimation.
        """
        if not self.is_fitted:
            raise ValueError("Model not trained! Call tune_hyperparameters() or train_rolling_window() first.")
            
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 1. Generate Returns Prediction
        p_ridge = self.ridge_model.predict(X)
        p_gb = self.gb_model.predict(X)
        expected_returns = 0.5 * p_ridge + 0.5 * p_gb
        
        # 2. Detect Market Regime
        regimes = self.regime_clf.predict(X)
        
        # 3. Estimate Dynamic Uncertainty (Risk)
        base_error = self.model_metrics.get('hybrid_rmse', 0.02)
        # Inflate uncertainty by 50% if in a high-volatility regime
        uncertainties = np.where(regimes == 1, base_error * 1.5, base_error)
        
        return expected_returns, uncertainties, regimes

# --- Usage Example ---
if __name__ == "__main__":
    # Fetch Data
    X, y, df = get_cleaned_data()
    
    if X is not None:
        agent = PredictiveEngine()
        
        # Step 1: Tune Parameters (finding best alpha / learning_rate)
        agent.tune_hyperparameters(X, y, n_iter=2)
        
        # Step 2: Evaluate Performance (Rolling Window)
        agent.train_rolling_window(X, y, n_splits=5)
        
        print("\n🔮 Forecasting for latest available data:")
        
        # Get latest data for each ticker for final prediction
        latest_full_data = df.sort_index().groupby('ticker').tail(1)
        sample_X = latest_full_data[X.columns]
        
        rets, errs, regimes = agent.predict(sample_X)
        
        results = pd.DataFrame({
            'Ticker': latest_full_data['ticker'].values,
            'Date': latest_full_data.index,
            'Pred_Return': rets,
            'Uncertainty': errs,
            'High_Vol_Regime': regimes
        })
        
        print(results)