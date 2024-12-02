import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('bakery_sales_revised.csv')

# Convert the 'date_time' column to a datetime object
data['date_time'] = pd.to_datetime(data['date_time'], format='%m/%d/%Y %H:%M')

# Add a new column for the week number
data['Week'] = data['date_time'].dt.isocalendar().week

# Check the dataset shape to ensure all rows are being read
print(f"Dataset shape: {data.shape}")

# Aggregate transactions into weekly sales
weekly_sales = data.groupby(['Week'])['Transaction'].count().reset_index(name='Weekly_Sales')

# Print the first few rows of the aggregated data to confirm
print("Aggregated weekly sales data:")
print(weekly_sales.head())

# Plot weekly sales trends
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales['Week'], weekly_sales['Weekly_Sales'], label='Weekly Sales', marker='o')
plt.title('Weekly Sales Trends')
plt.xlabel('Week')
plt.ylabel('Sales Count')
plt.legend()
plt.grid()
plt.show()

# Smooth weekly sales data using a rolling mean
weekly_sales['Smoothed_Sales'] = weekly_sales['Weekly_Sales'].rolling(window=3, min_periods=1).mean()

# Linear Regression to predict weekly sales
X = weekly_sales[['Week']].values
y = weekly_sales['Weekly_Sales'].values
model = LinearRegression()
model.fit(X, y)

# Print the coefficients of the linear regression model
print(f"Weekly Sales Forecast Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Predict sales for the next 5 weeks
future_weeks = pd.DataFrame({'Week': range(max(weekly_sales['Week']) + 1, max(weekly_sales['Week']) + 6)})
future_weeks['Predicted_Sales'] = model.predict(future_weeks[['Week']])

# Print future predictions
print("Future weekly sales predictions:")
print(future_weeks)

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales['Week'], weekly_sales['Weekly_Sales'], label='Observed Sales', marker='o')
plt.plot(weekly_sales['Week'], weekly_sales['Smoothed_Sales'], label='Smoothed Sales', linestyle='--')
plt.plot(future_weeks['Week'], future_weeks['Predicted_Sales'], label='Predicted Sales', marker='x')
plt.title('Observed, Smoothed, and Predicted Weekly Sales')
plt.xlabel('Week')
plt.ylabel('Sales Count')
plt.legend()
plt.grid()
plt.show()
