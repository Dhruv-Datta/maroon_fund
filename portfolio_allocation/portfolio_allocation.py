import numpy as np
import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os

# ========== USER INPUTS ==========
symbols = ['GOOGL', 'AMZN', 'HOOD', 'UBER', 'ASML', 'AAAU', 'HLT', 'UNH', 'BKNG', 'CASH']
expected_returns = np.array([0.1356, 0.1403, -0.0053, 0.096, 0.0967, 0.05, 0.0779, 0.1644, 0.1021, 0.02])
risk_free_rate = 0.042
min_weight = 0.03
max_weight = 0.15
cash_min_weight = 0.01
cash_max_weight = 0.05
num_of_portfolios = 70000

risk_factors = ['Volatility', 'Regulatory', 'Disruption', 'Valuation', 'Earnings Quality']
risk_factor_weights = np.array([0.9, 0.3, 0.4, 0.7, 0.6])

factor_exposures_dict = {
    'GOOGL': [0.50, 0.20, 0.40, 0.20, 0.15],
    'AMZN':  [0.55, 0.10, 0.25, 0.35, 0.20],
    'UNH':   [0.40, 0.60, 0.40, 0.15, 0.30],
    'HLT':   [0.25, 0.10, 0.30, 0.45, 0.25],
    'ASML':  [0.30, 0.15, 0.10, 0.25, 0.20],
    'HOOD':  [0.90, 0.40, 0.10, 0.80, 0.35],
    'AAAU':  [0.20, 0.01, 0.01, 0.50, 0.01],
    'UBER':  [0.70, 0.10, 0.50, 0.45, 0.25],
    'BKNG':  [0.25, 0.10, 0.30, 0.35, 0.20],
}

# ========== BUILD FACTOR EXPOSURE MATRIX ==========
assets = [sym for sym in symbols if sym != 'CASH']
K = len(risk_factors)
N = len(symbols)

F_raw = np.array([factor_exposures_dict[sym] for sym in assets])
F_norm = F_raw / F_raw.sum(axis=0, keepdims=True)  # L1 normalize column-wise
F = np.vstack([F_norm, np.zeros(K)])  # Add row for CASH

# ========== RISK FACTOR WEIGHTS ==========
W = np.diag(risk_factor_weights)

# ========== WEIGHTED COMPOSITE RISK MATRIX ==========
mu = np.mean(F, axis=0)
F_centered = F - mu
Sigma_F = (F_centered.T @ F_centered) / (N - 1)

Sigma_weighted = W @ Sigma_F @ W
Sigma_composite_weighted = F @ Sigma_weighted @ F.T
Sigma_composite_weighted = (Sigma_composite_weighted + Sigma_composite_weighted.T) / 2

# ========== MONTE CARLO SIMULATION ==========
print(f"Generating {num_of_portfolios:,} portfolio simulations...")
all_weights, ret_arr, vol_arr, sharpe_arr = [], [], [], []
samples_generated = 0

while samples_generated < num_of_portfolios:
    weights = np.random.random(N)
    weights /= np.sum(weights)

    cash_weight = weights[symbols.index('CASH')]
    other_weights = np.delete(weights, symbols.index('CASH'))

    if (
        np.all(other_weights >= min_weight) and
        np.all(other_weights <= max_weight) and
        cash_weight >= cash_min_weight and
        cash_weight <= cash_max_weight
    ):
        all_weights.append(weights)
        ret = np.dot(expected_returns, weights)
        vol = np.sqrt(weights.T @ Sigma_composite_weighted @ weights)
        sharpe = (ret - risk_free_rate) / vol if vol != 0 else 0

        ret_arr.append(ret)
        vol_arr.append(vol)
        sharpe_arr.append(sharpe)
        samples_generated += 1
    
    if samples_generated % 10000 == 0:
        print(f"  Progress: {samples_generated:,}/{num_of_portfolios:,}")

print("Simulation complete!\n")

# ========== RESULTS ==========
simulations_df = pd.DataFrame({
    'Returns': ret_arr,
    'Volatility': vol_arr,
    'Sharpe Ratio': sharpe_arr,
    'Portfolio Weights': all_weights
})

simulations_df['Weights_str'] = simulations_df['Portfolio Weights'].apply(
    lambda w: "<br>".join(f"{sym}: {wt:.2%}" for sym, wt in zip(symbols, w))
)

simulations_df['HoverText'] = simulations_df.apply(
    lambda row: f"Composite Ratio: {row['Sharpe Ratio']:.3f}<br>Return: {row['Returns']:.2%}<br>Volatility: {row['Volatility']:.2%}<br><br>{row['Weights_str']}",
    axis=1
)

max_sharpe = simulations_df.loc[simulations_df['Sharpe Ratio'].idxmax()]
min_vol = simulations_df.loc[simulations_df['Volatility'].idxmin()]

# ========== USER-DEFINED PORTFOLIO ==========
user_defined_weights = np.array([0.1848, 0.14, 0.07, 0.0783, 0.1134, 0.0822, 0.0914, 0.12, 0.08, 0.0399])
total_weight = user_defined_weights.sum()
print(f"User Portfolio - Total weight check: {total_weight:.4f}")

# Validate custom weights
assert np.isclose(user_defined_weights.sum(), 1.0), "Weights must sum to 1.0"
assert len(user_defined_weights) == len(symbols), "Custom allocation must match symbol count"

user_ret = np.dot(expected_returns, user_defined_weights)
user_vol = np.sqrt(user_defined_weights.T @ Sigma_composite_weighted @ user_defined_weights)
user_sharpe = (user_ret - risk_free_rate) / user_vol if user_vol != 0 else 0

user_weights_str = "<br>".join(f"{sym}: {wt:.2%}" for sym, wt in zip(symbols, user_defined_weights))
user_hover_text = (
    f"Composite Ratio: {user_sharpe:.3f}<br>"
    f"Return: {user_ret:.2%}<br>"
    f"Volatility: {user_vol:.2%}<br><br>{user_weights_str}"
)

# ========= SCALE SHARPE (Composite Ratio) TO [0, 1] =========
min_sharpe_val = simulations_df['Sharpe Ratio'].min()
max_sharpe_val = simulations_df['Sharpe Ratio'].max()
simulations_df['CompositeRatio'] = (
    (simulations_df['Sharpe Ratio'] - min_sharpe_val) / (max_sharpe_val - min_sharpe_val)
)

# Scale user + stars
user_sharpe_scaled = (user_sharpe - min_sharpe_val) / (max_sharpe_val - min_sharpe_val)
max_sharpe_scaled = (max_sharpe['Sharpe Ratio'] - min_sharpe_val) / (max_sharpe_val - min_sharpe_val)
min_vol_sharpe_scaled = (min_vol['Sharpe Ratio'] - min_sharpe_val) / (max_sharpe_val - min_sharpe_val)

# ========= SCALE COMPOSITE RISK TO [0, 1] =========
min_vol_val = simulations_df['Volatility'].min()
max_vol_val = simulations_df['Volatility'].max()
simulations_df['CompositeRisk'] = (
    (simulations_df['Volatility'] - min_vol_val) / (max_vol_val - min_vol_val)
)

# Scale special portfolios
user_vol_scaled = (user_vol - min_vol_val) / (max_vol_val - min_vol_val)
max_sharpe_vol_scaled = (max_sharpe['Volatility'] - min_vol_val) / (max_vol_val - min_vol_val)
min_vol_vol_scaled = (min_vol['Volatility'] - min_vol_val) / (max_vol_val - min_vol_val)

# ========== PLOT MAIN SCATTER ==========
print("Creating interactive visualization...")
fig = go.Figure()

# Add the main scatter plot first
fig.add_trace(go.Scatter(
    x=simulations_df['CompositeRisk'],
    y=simulations_df['Returns'],
    mode='markers',
    marker=dict(
        color=simulations_df['CompositeRatio'],
        cmin=0, cmax=1,
        colorscale='RdYlBu',
        colorbar=dict(title='Composite Ratio (0 to 1)'),
        size=5
    ),
    name='Portfolio Simulations',
    hovertext=simulations_df['HoverText'],
    hovertemplate="%{hovertext}<extra></extra>",
    showlegend=False
))

# Add Max Sharpe Portfolio
fig.add_trace(go.Scatter(
    x=[max_sharpe_vol_scaled],
    y=[max_sharpe['Returns']],
    mode='markers+text',
    text=["Max Composite Ratio"],
    textposition='top center',
    marker=dict(color='red', size=14, symbol='star'),
    name='Max Composite Portfolio',
    hovertext=[max_sharpe['HoverText']],
    hovertemplate="%{hovertext}<extra></extra>"
))

# Add Min Volatility Portfolio
fig.add_trace(go.Scatter(
    x=[min_vol_vol_scaled],
    y=[min_vol['Returns']],
    mode='markers+text',
    text=["Min Volatility"],
    textposition='top center',
    marker=dict(color='blue', size=14, symbol='star'),
    name='Min Volatility Portfolio',
    hovertext=[min_vol['HoverText']],
    hovertemplate="%{hovertext}<extra></extra>"
))

# Add User Portfolio
fig.add_trace(go.Scatter(
    x=[user_vol_scaled],
    y=[user_ret],
    mode='markers+text',
    text=["Your Portfolio"],
    textposition='top center',
    marker=dict(color='green', size=14, symbol='star'),
    name='User-Defined Portfolio',
    hovertext=[user_hover_text],
    hovertemplate="%{hovertext}<extra></extra>"
))

fig.update_layout(
    title='Efficient Frontier (Weighted Composite Risk)',
    xaxis_title='Composite Risk (0 to 1)',
    yaxis_title='Expected Return',
    clickmode='event+select',
    legend=dict(
        x=0.75,
        y=1.25,
        bgcolor='rgba(255,255,255,0.6)',
        bordercolor='black',
        borderwidth=0.5
    ),
    margin=dict(r=150)
)

# Save and open the HTML file
filename = "efficient_frontier_weighted_composite.html"
fig.write_html(filename)
print(f"Visualization saved to: {filename}")
print("Opening in browser...\n")
webbrowser.open('file://' + os.path.realpath(filename))

# ========== PRINT COVARIANCE AND CORRELATION MATRICES ==========
print("=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

print("\n1. Weighted Factor Covariance Matrix (Σ_F weighted):")
print(pd.DataFrame(Sigma_weighted, index=risk_factors, columns=risk_factors).to_string())

std_factors = np.sqrt(np.diag(Sigma_weighted))
factor_correlation = Sigma_weighted / np.outer(std_factors, std_factors)
print("\n2. Weighted Factor Correlation Matrix:")
print(pd.DataFrame(factor_correlation, index=risk_factors, columns=risk_factors).to_string())

print("\n3. Composite Asset Covariance Matrix (Σ_composite):")
print(pd.DataFrame(Sigma_composite_weighted, index=symbols, columns=symbols).to_string())

with np.errstate(divide='ignore', invalid='ignore'):
    std_assets = np.sqrt(np.diag(Sigma_composite_weighted))
    denom = np.outer(std_assets, std_assets)
    asset_correlation = np.divide(Sigma_composite_weighted, denom, out=np.zeros_like(Sigma_composite_weighted), where=denom!=0)

print("\n4. Composite Asset Correlation Matrix:")
print(pd.DataFrame(asset_correlation, index=symbols, columns=symbols).to_string())

print("\n" + "=" * 80)
print("PORTFOLIO SUMMARY")
print("=" * 80)
print(f"\nMax Composite Ratio Portfolio:")
print(f"  Return: {max_sharpe['Returns']:.2%}")
print(f"  Volatility: {max_sharpe['Volatility']:.2%}")
print(f"  Sharpe Ratio: {max_sharpe['Sharpe Ratio']:.3f}")

print(f"\nMin Volatility Portfolio:")
print(f"  Return: {min_vol['Returns']:.2%}")
print(f"  Volatility: {min_vol['Volatility']:.2%}")
print(f"  Sharpe Ratio: {min_vol['Sharpe Ratio']:.3f}")

print(f"\nYour Portfolio:")
print(f"  Return: {user_ret:.2%}")
print(f"  Volatility: {user_vol:.2%}")
print(f"  Sharpe Ratio: {user_sharpe:.3f}")
print("\nAnalysis complete!")