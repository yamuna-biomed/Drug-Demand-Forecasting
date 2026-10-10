import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


df = pd.read_csv('C:\\Users\\Divya\\Documents\\GitHub\\Drug-Demand-Forecasting\\data\\salesdaily.csv')

print("=" * 90)
print("DRUG DEMAND FORECASTING")
print("=" * 90)
print(f"Total records: {len(df)}")
print(f"Date range: {df['datum'].min()} to {df['datum'].max()}")
print()


df['datum'] = pd.to_datetime(df['datum'])
df = df.sort_values('datum')


medicine_columns = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

#  total daily sales
df['Total_Sales'] = df[medicine_columns].sum(axis=1)

print("=" * 90)
print("SALES STATISTICS")
print("=" * 90)
print(f"Average daily sales: ${df['Total_Sales'].mean():,.2f}")
print(f"Highest daily sales: ${df['Total_Sales'].max():,.2f}")
print(f"Lowest daily sales: ${df['Total_Sales'].min():,.2f}")
print(f"Total period sales: ${df['Total_Sales'].sum():,.2f}")
print()

#  CHART 1: HISTORICAL DAILY SALES TREND 
print("Creating Chart 1: Historical Daily Sales Trend...")
plt.figure(figsize=(16, 6))
plt.plot(df['datum'], df['Total_Sales'], linewidth=2, color='steelblue', label='Daily Sales')
plt.xlabel('Date')
plt.ylabel('Total Daily Sales ($)')
plt.title('Historical Daily Pharmacy Sales Trend (2014-2017)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:\\Users\\Divya\\Documents\\GitHub\\Drug-Demand-Forecasting\\output\\1_historical_sales_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 1_historical_sales_trend.png\n")

#  PREPARE DATA FOR FORECASTING 
print("Preparing data for forecasting...")

#  features for the model
df['Day_of_Year'] = df['datum'].dt.dayofyear
df['Month'] = df['datum'].dt.month
df['Weekday'] = df['datum'].dt.weekday
df['Days_Elapsed'] = (df['datum'] - df['datum'].min()).dt.days

# Rolling averages (trends)
df['Sales_MA7'] = df['Total_Sales'].rolling(window=7).mean()  # 7-day moving average
df['Sales_MA30'] = df['Total_Sales'].rolling(window=30).mean()  # 30-day moving average

# Remove NaN values created by rolling average
df_train = df.dropna()

# Prepare features (X) and target (y)
feature_columns = ['Days_Elapsed', 'Month', 'Weekday', 'Sales_MA7', 'Sales_MA30']
X = df_train[feature_columns]
y = df_train['Total_Sales']

print(f"Training data points: {len(X)}")
print()

#  PREDICTION MODEL 
print("=" * 90)
print("BUILDING FORECASTING MODEL")
print("=" * 90)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#  predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#  accuracy metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Model Performance:")
print(f"  Training R² Score: {train_r2:.4f}")
print(f"  Testing R² Score: {test_r2:.4f}")
print(f"  Training RMSE: ${train_rmse:,.2f}")
print(f"  Testing RMSE: ${test_rmse:,.2f}")
print()

#  CHART 2: MODEL ACCURACY (ACTUAL VS PREDICTED) 
print("Creating Chart 2: Model Accuracy - Actual vs Predicted...")
plt.figure(figsize=(14, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color='steelblue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Sales ($)')
plt.ylabel('Predicted Sales ($)')
plt.title(f'Model Accuracy: Actual vs Predicted Sales (R² = {test_r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C:\\Users\\Divya\\Documents\\GitHub\\Drug-Demand-Forecasting\\output\\2_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 2_actual_vs_predicted.png\n")

#  FORECAST FUTURE SALES 
print("=" * 90)
print("FORECASTING FUTURE SALES")
print("=" * 90)

# Get the last values for creating future predictions
last_record = df_train.iloc[-1]
last_date = df['datum'].max()
last_sales_ma7 = df_train['Sales_MA7'].iloc[-1]
last_sales_ma30 = df_train['Sales_MA30'].iloc[-1]

#  forecast for next 90 days
forecast_days = 90
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
forecast_data = []

for i, date in enumerate(forecast_dates):
    days_elapsed = (date - df['datum'].min()).days
    month = date.month
    weekday = date.weekday()
    
    # For moving averages, use recent average values
    forecast_ma7 = last_sales_ma7 + (i * 0.1)  # Slight trend
    forecast_ma30 = last_sales_ma30 + (i * 0.05)
    
    forecast_features = np.array([[days_elapsed, month, weekday, forecast_ma7, forecast_ma30]])
    predicted_sales = model.predict(forecast_features)[0]
    
    forecast_data.append({
        'Date': date,
        'Predicted_Sales': max(0, predicted_sales)  # Sales can't be negative
    })

forecast_df = pd.DataFrame(forecast_data)

print(f"\nForecast for next {forecast_days} days:")
print(f"  Average predicted daily sales: ${forecast_df['Predicted_Sales'].mean():,.2f}")
print(f"  Highest predicted sales: ${forecast_df['Predicted_Sales'].max():,.2f}")
print(f"  Lowest predicted sales: ${forecast_df['Predicted_Sales'].min():,.2f}")
print(f"  Total forecasted sales (90 days): ${forecast_df['Predicted_Sales'].sum():,.2f}")
print()

#  CHART 3: SALES FORECAST 
print("Creating Chart 3: Sales Forecast for Next 90 Days...")
plt.figure(figsize=(16, 6))

# Plot historical data
plt.plot(df['datum'], df['Total_Sales'], linewidth=2, color='steelblue', label='Historical Sales')

# Plot forecast
plt.plot(forecast_df['Date'], forecast_df['Predicted_Sales'], linewidth=2, color='red', 
         linestyle='--', label='90-Day Forecast', marker='o', markersize=3)

plt.axvline(x=last_date, color='green', linestyle=':', linewidth=2, label='Forecast Start')
plt.xlabel('Date')
plt.ylabel('Daily Sales ($)')
plt.title('Pharmacy Sales: Historical Data + 90-Day Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('C:\\Users\\Divya\\Documents\\GitHub\\Drug-Demand-Forecasting\\output\\3_sales_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 3_sales_forecast.png\n")

#  CHART 4: MONTHLY FORECAST SUMMARY 
print("Creating Chart 4: Monthly Forecast Summary...")
forecast_df['Month'] = forecast_df['Date'].dt.to_period('M')
monthly_forecast = forecast_df.groupby('Month')['Predicted_Sales'].sum()

plt.figure(figsize=(12, 6))
months = [str(m) for m in monthly_forecast.index]
plt.bar(range(len(monthly_forecast)), monthly_forecast.values, color='coral', alpha=0.7)
plt.xticks(range(len(monthly_forecast)), months, rotation=45)
plt.xlabel('Month')
plt.ylabel('Total Monthly Sales ($)')
plt.title('Forecasted Monthly Sales for Next 90 Days')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('C:\\Users\\Divya\\Documents\\GitHub\\Drug-Demand-Forecasting\\output\\4_monthly_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 4_monthly_forecast.png\n")

#  CHART 5: FORECAST CONFIDENCE BANDS 
print("Creating Chart 5: Forecast with Confidence Intervals...")
plt.figure(figsize=(16, 6))

#  confidence intervals (±10% of prediction)
forecast_df['Upper_Bound'] = forecast_df['Predicted_Sales'] * 1.1
forecast_df['Lower_Bound'] = forecast_df['Predicted_Sales'] * 0.9

# Plot historical data
plt.plot(df['datum'], df['Total_Sales'], linewidth=2, color='steelblue', label='Historical Sales')

# Plot forecast with confidence bands
plt.plot(forecast_df['Date'], forecast_df['Predicted_Sales'], linewidth=2, color='red', 
         linestyle='--', label='Forecast', marker='o', markersize=2)
plt.fill_between(forecast_df['Date'], forecast_df['Lower_Bound'], forecast_df['Upper_Bound'], 
                 alpha=0.2, color='red', label='Confidence Interval (±10%)')

plt.axvline(x=last_date, color='green', linestyle=':', linewidth=2, label='Forecast Start')
plt.xlabel('Date')
plt.ylabel('Daily Sales ($)')
plt.title('Sales Forecast with Confidence Intervals')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('C:\\Users\\Divya\\Documents\\GitHub\\Drug-Demand-Forecasting\\output\\5_forecast_confidence.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 5_forecast_confidence.png\n")

#   FORECAST RESULTS 
print("=" * 90)
print("SAVING FORECAST RESULTS")
print("=" * 90)

# Save forecast to CSV
forecast_df.to_csv('C:\\Users\\Divya\\Documents\\GitHub\\Drug-Demand-Forecasting\\output\\forecast_results.csv', index=False)
print("✓ Saved: forecast_results.csv\n")

# Save detailed report
with open('C:\\Users\\Divya\\Documents\\GitHub\\Drug-Demand-Forecasting\\output\\Forecast_Report.txt', 'w') as f:
    f.write("DRUG DEMAND FORECASTING REPORT\n")
    f.write("=" * 90 + "\n\n")
    
    f.write("MODEL INFORMATION\n")
    f.write("-" * 90 + "\n")
    f.write(f"Model Type: Linear Regression\n")
    f.write(f"Training Data Points: {len(X_train)}\n")
    f.write(f"Testing Data Points: {len(X_test)}\n\n")
    
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("-" * 90 + "\n")
    f.write(f"Training R² Score: {train_r2:.4f}\n")
    f.write(f"Testing R² Score: {test_r2:.4f}\n")
    f.write(f"Training RMSE: ${train_rmse:,.2f}\n")
    f.write(f"Testing RMSE: ${test_rmse:,.2f}\n\n")
    
    f.write("HISTORICAL SALES STATISTICS\n")
    f.write("-" * 90 + "\n")
    f.write(f"Average Daily Sales: ${df['Total_Sales'].mean():,.2f}\n")
    f.write(f"Highest Daily Sales: ${df['Total_Sales'].max():,.2f}\n")
    f.write(f"Lowest Daily Sales: ${df['Total_Sales'].min():,.2f}\n")
    f.write(f"Total Period Sales: ${df['Total_Sales'].sum():,.2f}\n\n")
    
    f.write("FORECAST SUMMARY (Next 90 Days)\n")
    f.write("-" * 90 + "\n")
    f.write(f"Forecast Period: {forecast_df['Date'].min().date()} to {forecast_df['Date'].max().date()}\n")
    f.write(f"Average Predicted Daily Sales: ${forecast_df['Predicted_Sales'].mean():,.2f}\n")
    f.write(f"Highest Predicted Daily Sales: ${forecast_df['Predicted_Sales'].max():,.2f}\n")
    f.write(f"Lowest Predicted Daily Sales: ${forecast_df['Predicted_Sales'].min():,.2f}\n")
    f.write(f"Total Forecasted Sales (90 days): ${forecast_df['Predicted_Sales'].sum():,.2f}\n\n")
    
    f.write("MONTHLY FORECAST BREAKDOWN\n")
    f.write("-" * 90 + "\n")
    for month, sales in monthly_forecast.items():
        f.write(f"{month}: ${sales:,.2f}\n")

print("✓ Saved: Forecast_Report.txt\n")

print("=" * 90)
print("✓ ALL FORECASTING ANALYSES COMPLETED SUCCESSFULLY!")
print("=" * 90)
print("\nOutput files saved in: output/ folder")
print("1. historical_sales_trend.png")
print("2. actual_vs_predicted.png")
print("3. sales_forecast.png")
print("4. monthly_forecast.png")
print("5. forecast_confidence.png")
print("6. forecast_results.csv")
print("7. Forecast_Report.txt")