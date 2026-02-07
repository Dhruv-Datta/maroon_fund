import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from scipy.special import expit



# Load data and model
data = pd.read_csv('test_input.csv')
features = [col for col in data.columns if col not in ['Date', 'Target']]
X = data[features]
y = data['Target']

model = joblib.load('nvda_model.joblib')
data['Dip_Probability'] = model.predict_proba(X)[:, 1]
data['Predicted_Dip'] = data['Dip_Probability'] > 0.90

# Convert Date to datetime (for Plotly hover and axis formatting)
data['Date'] = pd.to_datetime(data['Date'])

# =====================
# SHAP EXPLAINABILITY (IN BETA TESTING)
# =====================

scaler = model.named_steps['scaler']
xgb_model = model.named_steps['xgb']

# Scale the features manually
X_scaled = scaler.transform(X)

# SHAP on XGBoost model only, with feature names
print("Calculating SHAP values...")
explainer = shap.Explainer(xgb_model, X_scaled, feature_names=features)
shap_values = explainer(X_scaled)

# SHAP summary plot with actual feature names
print("Showing SHAP summary plot...")
shap.plots.beeswarm(shap_values, max_display=10)

# SHAP waterfall plot for first prediction
print("Showing SHAP waterfall plot for the first prediction...")
shap.plots.waterfall(shap_values[36], max_display=10)


# Get log-odds prediction for a specific row (e.g., row 4)
log_odds = xgb_model.predict(X_scaled[4].reshape(1, -1), output_margin=True)[0]

# Convert to probability
prob = expit(log_odds)

print(f"\nModel output for row 4:")
print(f"  Log-odds:     {log_odds:.4f}")
print(f"  Probability:  {prob:.4f} → This is the model’s predicted P(dip) for that date.")


# Optional: Top 5 features by SHAP magnitude
shap_vals_df = pd.DataFrame(shap_values.values, columns=features)
mean_abs = shap_vals_df.abs().mean().sort_values(ascending=False)
print("\nTop 5 most impactful features (mean absolute SHAP value):")
print(mean_abs.head(5))

# Plot using Plotly
fig = go.Figure()

# Add price line
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='blue'),
    hoverinfo='skip'
))

# Add dip points with hover tooltips
fig.add_trace(go.Scatter(
    x=data.loc[data['Predicted_Dip'], 'Date'],
    y=data.loc[data['Predicted_Dip'], 'Close'],
    mode='markers+text',
    name='Predicted Dip',
    marker=dict(color='red', size=8),
    text=[f"Prob: {p:.2f}" for p in data.loc[data['Predicted_Dip'], 'Dip_Probability']],
    hovertemplate=
        "Date: %{x}<br>" +
        "Price: %{y:.2f}<br>" +
        "Dip Probability: %{text}<extra></extra>"
))

# Optional: Add hover for all dates (not just predicted dips)
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    mode='markers',
    name='All Points',
    marker=dict(color='rgba(0,0,255,0)', size=5),  # Invisible markers for hover
    text=[f"{p:.2f}" for p in data['Dip_Probability']],
    hovertemplate=
        "Date: %{x}<br>" +
        "Price: %{y:.2f}<br>" +
        "Dip Probability: %{text}<extra></extra>",
    showlegend=False
))

# Update layout
fig.update_layout(
    title="Stock Price with Predicted Dips",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="closest",
    template="plotly_white"
)

fig.show()
