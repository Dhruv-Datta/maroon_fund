import glob, os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

ticker = "NVDA"                    
pattern = f"{ticker}_sell_analysis_*.csv"
latest_path = max(glob.glob(pattern), key=os.path.getmtime)

print(f"Using file → {latest_path}")

df = pd.read_csv(latest_path)
df['Date'] = pd.to_datetime(df['Date'])

# Create Plotly visualization matching buy side style
fig = go.Figure()

# Add price line
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='blue'),
    hoverinfo='skip'
))

# Add sell signals - red circles like buy side model
sells = df[df['Sell_Signal']]
if len(sells) > 0:
    fig.add_trace(go.Scatter(
        x=sells['Date'],
        y=sells['Close'],
        mode='markers+text',
        name='Predicted Sell',
        marker=dict(color='red', size=8),
        text=[f"Prob {p:.2f}" for p in sells['Sell_Probability']],
        textposition="top center",
        hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>Sell Probability: %{text}<extra></extra>"
    ))

# Update layout - full width like buy side model
fig.update_layout(
    title="Stock Price with Predicted Sell Signals",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="closest",
    template="plotly_white",
    autosize=True,
    width=None,
    height=600
)

fig.show()

num_signals = sells.shape[0]
total_days  = len(df)
latest      = df.iloc[-1]

print("\nSummary")
print("-" * 50)
print(f"File analysed       : {latest_path}")
print(f"Date range          : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Total trading days  : {total_days}")
print(f"Sell signals        : {num_signals} ({num_signals/total_days:.2%})")
print("-" * 50)
print(f"Latest day ({latest['Date'].date()})")
print(f"  • Close Price     : ${latest['Close']:.2f}")
print(f"  • Sell Probability: {latest['Sell_Probability']:.2f}")
print(f"  • Sell Signal     : {'YES' if latest['Sell_Signal'] else 'NO'}")