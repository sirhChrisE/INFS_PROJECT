import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('bakery_sales_revised.csv')

# Convert the 'date_time' column to a datetime object
data['date_time'] = pd.to_datetime(data['date_time'], format='%m/%d/%Y %H:%M')

# Add a new column for the week number and start date of the week
data['Week'] = data['date_time'].dt.isocalendar().week
data['Year'] = data['date_time'].dt.year

# Aggregate item-level sales into weekly frequency
item_sales = data.groupby(['Year', 'Week', 'Item'])['Transaction'].count().reset_index(name='Weekly_Item_Sales')

# Select a specific item for prediction (e.g., Bread)
selected_item = 'Bread'
item_data = item_sales[item_sales['Item'] == selected_item]

# Create a datetime column for the first day of each week
item_data['Week_Start_Date'] = pd.to_datetime(
    item_data['Year'].astype(str) + '-W' + item_data['Week'].astype(str) + '-1',
    format='%G-W%V-%u'
)

# Sort data by week start date for proper trend visualization
item_data = item_data.sort_values(by='Week_Start_Date')

# Plot observed weekly sales for the selected item
plt.figure(figsize=(12, 6))
plt.plot(item_data['Week_Start_Date'], item_data['Weekly_Item_Sales'], label=f'Observed Sales ({selected_item})', marker='o')
plt.title(f'Weekly Sales Trends for {selected_item}')
plt.xlabel('Date')
plt.ylabel('Sales Count')
plt.grid()

# Prepare data for polynomial regression
X = np.arange(len(item_data)).reshape(-1, 1)  # Use index for modeling
y = item_data['Weekly_Item_Sales'].values

# Create polynomial features
degree = 2  # Reduced for better stability
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict sales for the observed weeks
item_data['Fitted_Sales'] = model.predict(X_poly)

# Predict sales for the next year (52 weeks)
last_week_date = item_data['Week_Start_Date'].max()
future_dates = pd.date_range(last_week_date + pd.Timedelta(weeks=1), periods=52, freq='W-MON')
future_weeks = pd.DataFrame({'Week_Start_Date': future_dates})
X_future = np.arange(len(item_data), len(item_data) + len(future_dates)).reshape(-1, 1)
X_future_poly = poly.transform(X_future)
future_weeks['Predicted_Sales'] = model.predict(X_future_poly)

# Adjust unrealistic predictions (cap to reasonable min and max observed sales)
min_sales = item_data['Weekly_Item_Sales'].min()
max_sales = item_data['Weekly_Item_Sales'].max()
future_weeks['Predicted_Sales'] = future_weeks['Predicted_Sales'].clip(lower=min_sales, upper=max_sales)

# Plot the fitted line and future predictions
plt.plot(item_data['Week_Start_Date'], item_data['Fitted_Sales'], label='Fitted Sales (Polynomial)', linestyle='--', color='green')
plt.plot(future_weeks['Week_Start_Date'], future_weeks['Predicted_Sales'], label='Predicted Sales (Next Year)', color='red', linestyle='--', marker='x')
plt.legend()
plt.show()

# Print predictions for the next year
print("Predicted Sales for the Next Year:")
print(future_weeks)
