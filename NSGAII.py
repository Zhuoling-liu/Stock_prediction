import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Import your custom modules ---
# Ensure these files are in the same directory
from data_engine import get_cleaned_data
from rolling import PredictiveEngine 


# 2. Portfolio Optimizer (Pareto Logic)

class PortfolioOptimizer:
    """
    Implements Multi-Objective Optimization logic to find the Pareto Frontier.
    Simulates the result of NSGA-II by generating random portfolios and 
    filtering for non-dominated solutions (Pareto Sort).
    """
    def __init__(self, tickers, exp_returns, uncertainties, cov_matrix):
        self.tickers = tickers
        self.mus = exp_returns       # Expected Returns (from AI)
        self.uncs = uncertainties    # Prediction Uncertainty (from AI)
        self.cov = cov_matrix        # Covariance Matrix (Historical Risk)
        self.num_assets = len(tickers)

    def simulate_and_sort(self, num_portfolios=20000):
        """
        Generates portfolios and extracts the Pareto Frontier.
        """
        print(f" Simulating {num_portfolios} portfolios to find the Efficient Frontier...")
        
        results = np.zeros((4, num_portfolios)) # 0:Ret, 1:Risk, 2:Unc, 3:Sharpe
        weights_record = []

        for i in range(num_portfolios):
            # 1. Generate Random Weights (Dirichlet for sum=1)
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)

            # 2. Calculate Portfolio Metrics
            # Return
            p_ret = np.sum(weights * self.mus)
            # Risk (Volatility)
            p_var = np.dot(weights.T, np.dot(self.cov, weights))
            p_risk = np.sqrt(p_var)
            # Uncertainty (Objective 3: Minimize AI Error)
            p_unc = np.sum(weights * self.uncs) 
            
            # Store
            results[0,i] = p_ret
            results[1,i] = p_risk
            results[2,i] = p_unc
            # Sharpe (assuming 0 risk-free for simplicity)
            results[3,i] = p_ret / (p_risk + 1e-9)

        # 3. Pareto Sort (Non-dominated Sorting)
        # Find points where no other point has (Higher Ret AND Lower Risk)
        is_pareto = self._is_pareto_efficient_simple(results[:2, :].T)
        
        return results, np.array(weights_record), is_pareto

    def _is_pareto_efficient_simple(self, costs):
        """
        Efficiently finds the Pareto frontier for 2D (Risk vs Return).
        costs array structure: [Return, Risk] (Note: typically we minimize risk, maximize return)
        To use standard logic, we negate Return so we minimize both (-Ret, Risk).
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        # Convert to minimization problem: Minimize (-Return) and Minimize (Risk)
        minimize_data = costs.copy()
        minimize_data[:, 0] = -minimize_data[:, 0] # Negate return
        
        for i, c in enumerate(minimize_data):
            if is_efficient[i]:
                # Keep points where no other point is strictly better in BOTH dimensions
                is_efficient[i] = not np.any(
                    np.all(minimize_data < c, axis=1)
                )
        return is_efficient

# ==========================================
# 3. Main Execution Flow
# ==========================================
if __name__ == "__main__":
    # --- Step A: Get Data ---
    print(" Launching System...")
    X, y, df = get_cleaned_data()
    
    if X is None:
        print(" Data download failed.")
        exit()

    # --- Step B: Train AI Model ---
    print("\n Initializing Predictive Engine...")
    agent = PredictiveEngine()
    
    # 1. Auto-tune hyperparameters (optional, time-consuming)
    # agent.tune_hyperparameters(X, y, n_iter=5) 
    
    # 2. Train with Rolling Window
    agent.train_rolling_window(X, y, n_splits=5)

    # --- Step C: Prepare Prediction Inputs ---
    print("\n Generating forecasts for the NEXT period...")
    
    # Get the very last row for each ticker to predict "Tomorrow"
    # Ensure strict sorting to match ticker order
    latest_data = df.sort_index().groupby('ticker').tail(1).sort_values('ticker')
    ordered_tickers = latest_data['ticker'].values
    
    # Extract features for prediction
    X_latest = latest_data[X.columns]
    
    # Predict: Returns, Uncertainty, Regimes
    pred_rets, pred_uncs, pred_regimes = agent.predict(X_latest)
    
    # --- Step D: Calculate Risk Matrix (Covariance) ---
    print("Calculating Risk Matrix (Covariance)...")
    # Pivot: Index=Date, Columns=Ticker, Values=Close
    # We need historical returns for covariance
    price_matrix = df.pivot_table(index=df.index, columns='ticker', values='Close') # use pivot_table to handle dupes if any
    
    # Ensure columns match the prediction order
    price_matrix = price_matrix[ordered_tickers]
    
    # Calculate Annualized Covariance
    cov_matrix = price_matrix.pct_change().cov().values * 252

    # --- Step E: Run Optimization ---
    optimizer = PortfolioOptimizer(ordered_tickers, pred_rets, pred_uncs, cov_matrix)
    
    # Run Simulation
    all_results, all_weights, pareto_mask = optimizer.simulate_and_sort(num_portfolios=25000)
    
    # Extract Pareto Data
    pareto_results = all_results[:, pareto_mask]
    pareto_weights = all_weights[pareto_mask]
    
    print(f" Found {pareto_results.shape[1]} non-dominated solutions (Pareto Frontier).")

    # --- Step F: Identify Key Solutions ---
    # 1. Max Sharpe
    max_sharpe_idx = np.argmax(pareto_results[3])
    ws_sharpe = pareto_weights[max_sharpe_idx]
    
    # 2. Min Risk
    min_risk_idx = np.argmin(pareto_results[1])
    ws_risk = pareto_weights[min_risk_idx]
    
    # 3. Max Return
    max_ret_idx = np.argmax(pareto_results[0])
    ws_ret = pareto_weights[max_ret_idx]

    # --- Step G: Output 1 - Excel-Style Table ---
    print("\n Recommended Portfolio Allocations (Solution Matrix):")
    
    solution_df = pd.DataFrame({
        'Strategy': ordered_tickers,
        'Max Sharpe (Balanced)': ws_sharpe,
        'Min Risk (Conservative)': ws_risk,
        'Max Return (Aggressive)': ws_ret
    }).set_index('Strategy')
    
    # Format as percentage
    pd.set_option('display.float_format', '{:.2%}'.format)
    print(solution_df)
    
    # Export to CSV (Optional)
    # solution_df.to_csv("portfolio_solutions.csv")

    # --- Step H: Output 2 - Scatter Plot ---
    plt.figure(figsize=(12, 8))
    
    # 1. Plot all dominated solutions (Grey cloud)
    plt.scatter(all_results[1], all_results[0], c='lightgray', s=3, alpha=0.3, label='Dominated Portfolios')
    
    # 2. Plot Pareto Frontier (Colored by Uncertainty)
    # This visualizes the "Third Objective": We prefer darker colors (Lower Uncertainty)
    sc = plt.scatter(pareto_results[1], pareto_results[0], c=pareto_results[2], 
                     cmap='viridis_r', s=20, alpha=0.8, label='Pareto Frontier')
    
    # 3. Highlight Max Sharpe
    ms_metrics = pareto_results[:, max_sharpe_idx]
    plt.scatter(ms_metrics[1], ms_metrics[0], c='red', s=200, marker='*', 
                edgecolors='black', zorder=10, 
                label=f'Max Sharpe (Ratio: {ms_metrics[3]:.2f})')
    
    # Formatting
    plt.colorbar(sc, label='Prediction Uncertainty (AI Error)')
    plt.title('Multi-Objective Optimization Landscape', fontsize=16, fontweight='bold')
    plt.xlabel('Expected Volatility (Risk)', fontsize=12)
    plt.ylabel('Predicted Return (Alpha)', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    #Generating Strategy Comparison Chart...
    plt.figure(figsize=(12, 8))

    # 1. Prepare Data for Stacked Bar Chart
    plot_data = solution_df.T 
    
    # 2. Plot Stacked Bar Chart
    ax = plot_data.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='tab20', width=0.7)
    
    # 3. Formatting
    plt.title('Portfolio Composition by Strategy', fontsize=16, fontweight='bold')
    plt.ylabel('Allocation Weight', fontsize=12)
    plt.xlabel('Optimization Strategy', fontsize=12)
    plt.xticks(rotation=0) 
    plt.legend(title='Tickers', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Add percentage labels on each segment
    for c in ax.containers:
        labels = [f'{v.get_height():.1%}' if v.get_height() > 0.02 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.show()
    
    print("Comparison chart generated.")