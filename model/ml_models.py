
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st 

from prophet import Prophet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error





# Define a function that will predict the stocks price in the next one year or so

def future_prediction(df):
	
	
	# Data Preprocessing
	# Reset index and rename columns for Prophet

	df.reset_index(inplace=True)
	df.rename(columns={'Date': 'ds', 'High': 'y'}, inplace=True)

	df['ds'] = df['ds'].dt.tz_localize(None)

	# Train the model
	model = Prophet()
	model.fit(df)

	# Make future predictions
	future = model.make_future_dataframe(periods=730)  # Predict 1 year into the future
	forecast = model.predict(future)

	# Step 5: Visualization
	# Plot the predictions
	fig = go.Figure()

	# Add historical data
	# fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual', fill='red'))

	# Add forecasted data
	fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))

	# Add prediction intervals
	# fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line_color='lightgrey', name='Lower Confidence Interval'))
	# fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line_color='lightgrey', name='Upper Confidence Interval'))

	# Update layout
	fig.update_layout(
	    title=f'Future Stock Price Prediction',
	    xaxis_title='Date',
	    yaxis_title='Price',
	    legend_title='Legend'
	)

	# Display the plot in Streamlit
	st.plotly_chart(fig)

	return forecast['ds'], forecast['yhat']











def regression_model(df):
	

	df['ds'] = pd.to_datetime(df['ds'])

	df['year'] = df['ds'].dt.year
	df['month'] = df['ds'].dt.month
	df['day'] = df['ds'].dt.day
	

	# features = ['open', 'close', 'low', 'volume', 'year', 'month', 'day']
	features = ['Open', 'Close', 'Low', 'Volume', 'year', 'month', 'day']
	

	x = df[features]
	y = df['y']

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

	# Keep the Date series for plotting
	x_test_dates = df.loc[x_test.index, 'ds']

	# instance of the models
	rfr = RandomForestRegressor(n_estimators=100, random_state=42)
	dtr = DecisionTreeRegressor(random_state=42)
	xgb = XGBRegressor(n_estimators=100, random_state=42)
	gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)

	# fit the models
	rfr.fit(x_train, y_train)
	dtr.fit(x_train, y_train)
	xgb.fit(x_train, y_train)
	gbr.fit(x_train, y_train)

	# predict the target variable 
	rfr_pred = rfr.predict(x_test)
	dtr_pred = dtr.predict(x_test)
	xgb_pred = xgb.predict(x_test)
	gbr_pred = gbr.predict(x_test)

	print(f'Mean Squared Error for Random Forest Regressor Model: {mean_squared_error(y_test, rfr_pred, squared=False)}')
	print(f'Mean Squared Error for Decision Tree Regressor Model: {mean_squared_error(y_test, dtr_pred, squared=False)}')
	print(f'Mean Squared Error for XGBoost Regressor Model: {mean_squared_error(y_test, xgb_pred, squared=False)}')
	print(f'Mean Squared Error for Gradient Boost Regressor Model: {mean_squared_error(y_test, gbr_pred, squared=False)}')
	print('')
	print('The smaller the MSE, the better the models predictive accuracy. Model Selection: In cases where multiple models are considered for a specific problem, the one with the lowest MSE is often preferred as it demonstrates better fitting to the data.')


	# Combine predictions
	final_pred = 0.5 * rfr_pred + 0.5 * dtr_pred

	# Calculate error
	error = mean_squared_error(y_test, final_pred, squared=False)

	# Display error
	print(f"Root Mean Squared Error: {error}")

	# Plot actual vs predicted prices
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual', line_color='white'))
	fig.add_trace(go.Scatter(x=x_test_dates, y=final_pred, mode='lines', name='Predicted', line_color='red', opacity=0.5))

	fig.update_layout(
	    title=f'Stock Price Regression',
	    xaxis_title='Date',
	    yaxis_title='Price',
	    legend_title='Legend'
	)

	# fig.show()
	st.plotly_chart(fig)
	return x_test_dates, final_pred

# data = pd.read_csv('../source/big_tech_stock_prices.csv')
# regression_model(data)








