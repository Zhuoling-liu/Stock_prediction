import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas_datareader.data as web # Ensure this is installed

# Import modules from previous days
from data_engine import get_cleaned_data
from rolling_window import PredictiveEngine

class MVFOptimizer:
    """
    【Day 3: The Decision Maker】
    Implements the Mean-Variance-Forecast Error (MVF) Model from Huang et al. (2026).
    Objective: Minimize (Risk - Return + Uncertainty)
    """
    
    def __init__(self, alpha1=0.8, alpha2=0.15, alpha3=0.05):
        """
        Parameters defined in the paper (Section 5.2):
        alpha1 (0.80): Aversion to Historical Volatility (Risk)
        alpha2 (0.15): Preference for AI Predicted Returns
        alpha3 (0.05): Aversion to Prediction Uncertainty (RMSE)
        """
        self.a1 = alpha1
        self.a2 = alpha2
        self.a3 = alpha3

    def optimize(self, predictions_df, historical_data):
        """
        Mathematical Optimization Solver.
        """
        print(f"\n⚡ Launching MVF Optimization...")
        print(f"   Parameters: Risk(a1)={self.a1}, Return(a2)={self.a2}, Uncertainty(a3)={self.a3}")
        
        # 1. Prepare Data Matrices
        # Ensure Ticker order consistency
        tickers = predictions_df['Ticker'].values
        n_assets = len(tickers)
        
        # Vector R: AI Predicted Returns (Day 2 Result)
        R_pred = predictions_df['Pred_Return'].values
        
        # Vector E: AI Predicted Uncertainty/RMSE (Day 2 Result)
        E_pred = predictions_df['Uncertainty'].values
        
        # Matrix Sigma: Historical Covariance Matrix (Day 1 Data)
        # --- [Critical Fix] ---
        # Previously caused errors by trying to pivot already aligned data.
        # Now using historical_data directly, ensuring columns match the ticker order in predictions_df.
        pivot_price = historical_data[tickers]
        
        # Calculate Covariance of daily returns and annualize it (x252)
        Sigma = pivot_price.pct_change().cov().values * 252
        
        # 2. Define Objective Function
        # Minimize: 0.5 * a1 * (w @ Sigma @ w) - a2 * (R @ w) + a3 * (E @ w)
        def objective(weights):
            portfolio_risk = 0.5 * self.a1 * np.dot(weights.T, np.dot(Sigma, weights))
            portfolio_return = self.a2 * np.dot(R_pred, weights)
            portfolio_uncertainty = self.a3 * np.dot(E_pred, weights)
            
            # We want to Minimize Risk and Uncertainty, and Maximize Return.
            # Therefore, Objective = Minimize: Risk - Return + Uncertainty
            return portfolio_risk - portfolio_return + portfolio_uncertainty

        # 3. Define Constraints
        # Constraint 1: Sum of all weights must equal 1 (100% capital allocation)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Constraint 2: Individual Asset Bounds
        # Relaxing single stock cap to 30% (0.3) to prevent infeasible solutions
        bounds = tuple((0.0, 0.3) for _ in range(n_assets))
        
        # 4. Solve Optimization Problem
        initial_guess = np.array([1/n_assets] * n_assets) # Initial guess: Equal weights
        
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            print(f"❌ Optimization Failed: {result.message}")
            return None
            
        # 5. Format Output Results
        optimal_weights = result.x
        
        # Clean up negligible weights (e.g., 0.000001 -> 0)
        optimal_weights[optimal_weights < 1e-4] = 0
        
        # Normalize (Ensure sum is strictly 1)
        if np.sum(optimal_weights) > 0:
            optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        final_df = predictions_df.copy()
        final_df['Optimal_Weight'] = optimal_weights
        
        # Print formatted report
        print("\n🏆 Optimization Complete! Optimal Portfolio Allocation:")
        print(final_df[['Ticker', 'Pred_Return', 'Uncertainty', 'Optimal_Weight']].sort_values(by='Optimal_Weight', ascending=False))
        
        return final_df

    def plot_allocation(self, df):
        """
        Plots the asset allocation as a pie chart.
        """
        # Only plot assets with weights > 1%
        plot_data = df[df['Optimal_Weight'] > 0.01]
        
        plt.figure(figsize=(10, 7))
        plt.pie(plot_data['Optimal_Weight'], labels=plot_data['Ticker'], autopct='%1.1f%%', startangle=140)
        plt.title('Final AI Portfolio Allocation (MVF Model)')
        plt.show()

# --- Execution Block ---
if __name__ == "__main__":
    # 1. Fetch Data (Day 1)
    X, y, df = get_cleaned_data()
    
    if X is not None:
        # 2. Run Predictive Model (Day 2)
        agent = PredictiveEngine()
        
        # Use Rolling Window Training (Running 3 splits for demo speed)
        agent.train_rolling_window(X, y, n_splits=3)
        
        # Prepare Prediction Input
        latest_full_data = df.groupby('ticker').tail(1)
        feature_cols = X.columns
        sample_X = latest_full_data[feature_cols]
        sample_X = sample_X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Get Day 2 Predictions
        rets, errs, regimes = agent.predict(sample_X)
        predictions_df = pd.DataFrame({
            'Ticker': latest_full_data['ticker'].values,
            'Pred_Return': rets,
            'Uncertainty': errs
        })
        
        # 3. Run Optimizer (Day 3)
        optimizer = MVFOptimizer(alpha1=0.8, alpha2=0.15, alpha3=0.05)
        
        print("\n📥 Fetching historical prices for Covariance Matrix...")
        # Re-download raw price data for the last year to calculate risk (Covariance)
        price_data = pd.DataFrame()
        
        # Temporary Fix: If Stooq is unstable, retry or check connection
        try:
            for t in predictions_df['Ticker']:
                # print(f"   Downloading price for {t}...")
                tmp = web.DataReader(t, 'stooq', start='2025-01-01', end='2026-01-01')
                # Stooq data is reverse chronological; needs to be reversed!
                price_data[t] = tmp['Close'].iloc[::-1] 
            
            # Simple fill for missing values to prevent covariance errors
            price_data = price_data.fillna(method='ffill').fillna(method='bfill')
            
            # Execute Optimization
            final_portfolio = optimizer.optimize(predictions_df, price_data)
            
            if final_portfolio is not None:
                optimizer.plot_allocation(final_portfolio)
                
        except Exception as e:
            print(f"⚠️ Error during price fetching or optimization: {e}")
            print("Suggest: Check internet connection or try again.")