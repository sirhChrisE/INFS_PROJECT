import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# --- Data Loading and Preprocessing ---
# Load the dataset
# Reason: Initial data load to begin all analysis
# Load the bakery sales dataset for analysis
data = pd.read_csv('bakery_sales_revised.csv')

# Convert the 'date_time' column to a datetime object
# Reason: Ensure dates are recognized in datetime format for easy manipulation
data['date_time'] = pd.to_datetime(data['date_time'], format='%m/%d/%Y %H:%M')

# Add new columns for the week number and year
# Reason: Extract temporal features that help in aggregating sales data
data['Week'] = data['date_time'].dt.isocalendar().week
data['Year'] = data['date_time'].dt.year

# Check the dataset shape to ensure all rows are being read
print(f"Dataset shape: {data.shape}")

# --- Weekly Sales Aggregation --
# Aggregate transactions into weekly sales
# Reason: Summarize data to see weekly patterns in sales
weekly_sales = data.groupby(['Week'])['Transaction'].count().reset_index(name='Weekly_Sales')

# Print the first few rows of the aggregated data to confirm
print("Aggregated weekly sales data:")
print(weekly_sales.head())

# Plot weekly sales trends
# Reason: Visualize the sales trends over weeks
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales['Week'], weekly_sales['Weekly_Sales'], label='Weekly Sales', marker='o')
plt.title('Weekly Sales Trends')
plt.xlabel('Week')
plt.ylabel('Sales Count')
plt.legend()
plt.grid()
plt.show(block=True)

# -- Smoothing Sales Data ---
# Smooth weekly sales data using a rolling mean
# Reason: Rolling mean helps reduce noise and show the overall trend
weekly_sales['Smoothed_Sales'] = weekly_sales['Weekly_Sales'].rolling(window=3, min_periods=1).mean()

# --- Linear Regression for Weekly Sales Forecasting ---
# Linear Regression to predict weekly sales
# Reason: Use a simple linear model to predict future sales based on historical weekly data
X = weekly_sales[['Week']].values
y = weekly_sales['Weekly_Sales'].values
model = LinearRegression()
model.fit(X, y)

# Print the coefficients of the linear regression model
print(f"Weekly Sales Forecast Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Predict sales for the next 5 weeks
# Reason: Forecasting future sales to plan inventory and operations
future_weeks = pd.DataFrame({'Week': range(max(weekly_sales['Week']) + 1, max(weekly_sales['Week']) + 6)})
future_weeks['Predicted_Sales'] = model.predict(future_weeks[['Week']])

# Print future predictions
print("Future weekly sales predictions:")
print(future_weeks)

# Plot predictions
# Reason: Visualize the observed, smoothed, and predicted sales over weeks
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales['Week'], weekly_sales['Weekly_Sales'], label='Observed Sales', marker='o')
plt.plot(weekly_sales['Week'], weekly_sales['Smoothed_Sales'], label='Smoothed Sales', linestyle='--')
plt.plot(future_weeks['Week'], future_weeks['Predicted_Sales'], label='Predicted Sales', marker='x')
plt.title('Observed, Smoothed, and Predicted Weekly Sales')
plt.xlabel('Week')
plt.ylabel('Sales Count')
plt.legend()
plt.grid()
plt.show(block=True)

# --- K-Means Clustering of Items ---
# Reason: Identify groups of items that have similar sales trends to optimize marketing and inventory
# Prepare data for clustering - aggregating sales by item and week
item_week_sales = data.groupby(['Item', 'Week']).size().reset_index(name='Sales_Count')
item_sales_pivot = item_week_sales.pivot(index='Item', columns='Week', values='Sales_Count').fillna(0)

# Standardize the data before clustering
# Reason: Ensure all features are on the same scale to improve clustering accuracy
scaler = StandardScaler()
item_sales_scaled = scaler.fit_transform(item_sales_pivot)

# Apply k-means clustering to group products with similar sales patterns
# Reason: Segment items into different clusters to understand which products behave similarly over time
kmeans = KMeans(n_clusters=3, random_state=42)
item_sales_pivot['Cluster'] = kmeans.fit_predict(item_sales_scaled)

# Print clustering results
print("Product clusters:")
print(item_sales_pivot[['Cluster']].value_counts())

# Visualize the clusters using a heatmap
# Reason: Provide a visual representation of the sales patterns grouped by clusters
plt.figure(figsize=(12, 8))
sns.heatmap(item_sales_pivot.drop(columns='Cluster'), cmap='viridis', cbar=True)
plt.title('Sales Patterns of Items by Week (Heatmap)')
plt.xlabel('Week')
plt.ylabel('Item')
plt.show(block=True)

# Plot products grouped by clusters
# Reason: Visualize the clusters in terms of sales count and weeks
item_sales_pivot.reset_index(inplace=True)
item_sales_pivot_melted = item_week_sales.merge(item_sales_pivot[['Item', 'Cluster']], on='Item')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=item_sales_pivot_melted, x='Week', y='Sales_Count', hue='Cluster', palette='deep')
plt.title('Product Sales Clusters')
plt.xlabel('Week')
plt.ylabel('Sales Count')
plt.legend()
plt.grid()
plt.show(block=True)

# Analysis and conclusions for clustering
# Clusting helps show which items have similar patterns overtime

# --- Random Forest Regression for Top Item Sales Prediction ---
# Reason: Predict the daily sales for the top-selling item for the first 5 months of 2018 for demonstration
# Filtering data for 2016 and 2017 for predictions
historical_data = data[data['Year'].isin([2016, 2017])]

# Aggregate total sales per item for 2016 and 2017
# Reason: Find the item that sold the most to focus the prediction on that item
total_item_sales = historical_data.groupby('Item')['Transaction'].count().reset_index(name='Total_Sales')

# Get the highest sold item based on total sales
top_item = total_item_sales.sort_values(by='Total_Sales', ascending=False).iloc[0]['Item']

# Aggregate daily sales for the highest sold item
# Reason: Get daily sales data to create a prediction for 2018
daily_item_sales = historical_data[historical_data['Item'] == top_item].groupby(historical_data['date_time'].dt.date).size().reset_index(name='Daily_Sales')
daily_item_sales['DayOfYear'] = pd.to_datetime(daily_item_sales['date_time']).dt.dayofyear

# Use RandomForestRegressor for better trend capturing for 2018 sales prediction
# Reason: RandomForest can model non-linear trends more effectively than linear regression
X_item = daily_item_sales[['DayOfYear']].values
y_item = daily_item_sales['Daily_Sales'].values
model_item = RandomForestRegressor(n_estimators=100, random_state=42)
model_item.fit(X_item, y_item)

# Predict daily sales for the first 5 months of 2018 (assuming 31 days per month)
# Reason: Focus prediction on the initial 5 months for a detailed view
future_days = pd.DataFrame({'DayOfYear': range(1, 152)})
future_days['Predicted_Sales'] = model_item.predict(future_days[['DayOfYear']])
future_days['Item'] = top_item

# Plot the detailed daily sales predictions for the first 5 months of 2018
# Reason: Visual comparison of the past and future prediction to validate model performance
plt.figure(figsize=(14, 8))
plt.plot(daily_item_sales[daily_item_sales['DayOfYear'] <= 151]['DayOfYear'], 
         daily_item_sales[daily_item_sales['DayOfYear'] <= 151]['Daily_Sales'], 
         label='Observed Sales 2016 & 2017', color='green', marker='o', linestyle='--')
plt.plot(future_days['DayOfYear'], future_days['Predicted_Sales'], label='Predicted Sales for 2018', linestyle='-', alpha=0.7)
plt.title(f'Detailed Daily Sales Prediction for {top_item} (First 5 Months of 2018)')
plt.xlabel('Day of Year')
plt.xticks(ticks=np.arange(0, 152, 30), labels=pd.date_range('2018-01-01', periods=6, freq='M').strftime('%b'))
plt.ylabel('Sales Count')
plt.legend()
plt.grid()
plt.show(block=True)

# Create a DataFrame for the prediction and save to CSV
# Reason: Save the predictions for future analysis or reporting
future_days['Date'] = pd.to_datetime('2018-01-01') + pd.to_timedelta(future_days['DayOfYear'] - 1, unit='D')
future_days = future_days[['Item', 'Date', 'Predicted_Sales']]
