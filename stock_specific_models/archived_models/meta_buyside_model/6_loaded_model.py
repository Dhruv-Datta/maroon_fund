import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load data and model
data = pd.read_csv('test_input.csv')
features = [col for col in data.columns if col not in ['Date', 'Target']]
X = data[features]
y = data['Target']

model = joblib.load('meta_model.joblib')
data['Dip_Probability'] = model.predict_proba(X)[:, 1]
data['Predicted_Dip'] = data['Dip_Probability'] > 0.90

# Convert Date to datetime (for Plotly hover and axis formatting)
data['Date'] = pd.to_datetime(data['Date'])

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
