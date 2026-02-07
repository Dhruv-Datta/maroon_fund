import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('training_input.csv')

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Display the first few rows of the data
print(data[['Date', 'Close', 'Target']].head(10))

# Create a plot to visualize the data
plt.figure(figsize=(15, 10))

# Plot the closing price
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')

# Highlight the areas where Target is 1 (indicating a dip)
plt.fill_between(data['Date'], data['Close'].min(), data['Close'].max(), 
                 where=data['Target'] == 1, color='red', alpha=0.3, 
                 label='Actual Dip')

# Customize the plot
plt.title('Stock Price with Actual Dips')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(data[['Close', 'Target']].describe())

# Print the number of dips
num_dips = data['Target'].sum()
total_days = len(data)
print(f"\nNumber of dips: {num_dips} out of {total_days} days ({num_dips/total_days:.2%})")
