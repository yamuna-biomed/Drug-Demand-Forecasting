DRUG DEMAND FORECASTING



&#x20;Project Overview:

This project develops a predictive analytics model to forecast future pharmacy drug/medicine demand based on historical sales data. Using linear regression and time series analysis, the model predicts daily medicine sales for the next 90 days with confidence intervals.



&#x20;Dataset Information:

\- File: `salesdaily.csv`

\- Data Period: January 2014 - December 2017

\- Total Records: 1,460 days of sales data

\- Medicines Tracked: 8 different medicine types (M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06)



&#x20;Key Dataset Features:

\- Daily sales amounts for each medicine

\- Temporal data (Year, Month, Day, Weekday)

\- Approximately 4 years of historical data



&#x20;Methodology:



&#x20;Data Preparation

\- Convert date strings to datetime format

\- Calculate daily total sales across all medicines

\- Create engineered features:

&#x20; - Days elapsed

&#x20; - Month

&#x20; - Weekday

&#x20; - 7-day moving average

&#x20; - 30-day moving average



&#x20;Model Building

\- Algorithm: Linear Regression

\- Training/Testing Split: 80% training, 20% testing

\- Features: 5 engineered features capturing temporal patterns and trends



&#x20;Performance Metrics

\- R² Score: Measures how well the model explains sales variance

\- RMSE (Root Mean Squared Error): Measures average prediction error

\- Model Accuracy: Evaluated on both training and testing datasets



&#x20;Visualizations Generated:



1\. \*\*1\_historical\_sales\_trend.png\*\* - Complete historical sales data from 2014-2017

2\. \*\*2\_actual\_vs\_predicted.png\*\* - Model accuracy showing actual vs predicted values

3\. \*\*3\_sales\_forecast.png\*\* - 90-day sales forecast with historical context

4\. \*\*4\_monthly\_forecast.png\*\* - Aggregated monthly forecast

5\. \*\*5\_forecast\_confidence.png\*\* - Forecast with confidence intervals (±10%)



&#x20;Outputs Generated:



\- \*\*forecast\_results.csv\*\* - Detailed predictions for next 90 days

\- \*\*Forecast\_Report.txt\*\* - Comprehensive analysis report with statistics



