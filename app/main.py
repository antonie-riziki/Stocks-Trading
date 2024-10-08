import pandas as pd 
import seaborn as sb 
import streamlit as st
import numpy as np 
import json
import os
import csv
import sys 
import time
import warnings 
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import requests
import datetime
import subprocess
import yfinance as yf

from datetime import datetime, timedelta
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.ml_models import regression_model, future_prediction

sb.set()
sb.set_style('darkgrid')
sb.set_palette('viridis')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

warnings.filterwarnings('ignore')


try:
    # check if the key exists in session state
    _ = st.session_state.keep_graphics
except AttributeError:
    # otherwise set it to false
    st.session_state.keep_graphics = False


with st.sidebar:
	selected = option_menu(
		menu_title = 'Menu',
		options = ['Home', 'Live Trading', 'Model'],
		icons = ['speedometer', 'graph-up-arrow', 'computer'],
		menu_icon = 'cast',
		default_index = 0
		)


if selected == 'Home':

	header = st.container()
	local_data = st.container()

	with header:
		st.image('../source/img7.jpg')
		st.title('GLOBAL STOCKS MARKET DATA')
		st.write('**What do stocks mean?**')
		st.write('A stock represents a share in the ownership of a company, including a claim on the companys earnings and assets. As such, stockholders are partial owners of the company. When the value of the business rises or falls, so does the value of the stock.')


	with local_data:
		col1, col2 = st.columns(2)
		
		with col1:
			st.image('../source/img6.png', width=350)

		with col2:
			st.image('../source/img4.webp', width=350)

		
		st.write("<h3 style='text-align: center; color: white;'>One place for your portfolios, <br>metrics and more<h3>", unsafe_allow_html=True)
		st.write('Gain insights, see trends and get real-time updates from well researched and analyzed datasets.')
		st.write('However the developer will integrate the results with Machine Learning algorithms for effecient and predictive output. This will boost accuracy and confidence in investing in stocks.')
		st.write('sorry I wasnt listening.....I was thinking about TRADING')


		df = pd.read_csv('../source/big_tech_stock_prices.csv')
		st.dataframe(df.head())

		# Data Cleaning
		df['date'] = pd.to_datetime(df['date'])

		st.write('### Statistical Representation of Data')

		col1, col2 = st.columns(2)
		
		with col1:
			st.write('Rows ', df.shape[0], 'Columns / Series ', df.shape[1])
			# st.write('Columns / Series ', df.shape[1])
		
			# Capture the output of df.info()
			buffer = io.StringIO()
			df.info(buf=buffer)
			info_str = buffer.getvalue()

			# Display df.info() output in Streamlit
			st.write('Summary of the dataframe')
			st.text(info_str)

		with col2:
			st.write('')
			st.write('')
			st.write('')
			st.write('')
			st.write('Description of the dataframe')
			st.dataframe(df.describe())

		st.write('### Graphical Presentation of Data')

		def get_numerical(df):
			numerical_list = []
			categories = df.select_dtypes(include=['float64', 'int64'])
			for i in categories:
				numerical_list.append(i)
			print(numerical_list)
			return numerical_list

		plot_hist_column = st.selectbox('Select dataframe series', (i for i in get_numerical(df)))

		print(plot_hist_column)

		fig = px.histogram(df[plot_hist_column], 
				title = 'Stock Distribution Plot for ' +  str(plot_hist_column) + ' series'
			)
		
		print(df[plot_hist_column])
		st.plotly_chart(fig)
		
		def get_categories(df):
		    cat = []
		    categories = df.select_dtypes(include=['float64', 'int64'])
		    for i in categories:
		        cat.append(i)
		    print(cat)
		    # fig = sb.heatmap(df[cat].corr(), annot=True, linewidths=0.5)
		    fig = px.imshow(df[cat].corr(), text_auto=True, aspect='auto', 
		    	title = 'Pearsons Correlation of Columns'
		    	)
		    st.plotly_chart(fig)

		get_categories(df)

		col1, col2 = st.columns(2)

		with col1:
			stock_company = st.selectbox('Select Symbol company', df['stock_symbol'].unique())

		with col2:
			stock_clause = st.selectbox('Select Stock Clause', get_numerical(df))

		company_group = df.groupby('stock_symbol').get_group(stock_company)
		
		
		pivot_df = company_group.pivot(index='stock_symbol', columns='date', values=stock_clause)
		print(pivot_df)

		fig = go.Figure(data=go.Heatmap(
			z=pivot_df,
	        x=company_group['date'],
	        y=company_group[stock_clause],
	        colorscale='Viridis'))
		
		fig.update_layout(
			title=f"Daily stocks charts from  {df['date'].dt.date.min()} to {df['date'].dt.date.max()}",
			# xaxis_title='Date',
			yaxis_title=f'{stock_clause} Price',
			legend_title='Company'
			)
		st.plotly_chart(fig)
		

		
		stock_symbol = df.groupby('stock_symbol').get_group(stock_company)

		
		fig = go.Figure()

		# Adding a trace for the company's open stock prices
		fig.add_trace(go.Scatter(x=stock_symbol['date'], y=stock_symbol['open'], mode='lines'))

		# frames = [go.Frame(data=[go.Scatter(x=company_group['date'][:k+1], y=company_group['open'][:k+1])],
	    #                name=str(company_group['date'].iloc[k])) for k in range(len(company_group.head(50)))]

		# fig.frames = frames

		# Setting the title and labels
		fig.update_layout(
			title=f'Open Stocks of the Tech company {stock_company}',
			xaxis_title='Date',
			yaxis_title='Open Price',
			legend_title='Company'
			)



		# Display the figure
		st.plotly_chart(fig)


		def load_lottiefile(filepath: str):
			with open(filepath, 'r') as file:
				return json.load(file)
		
		animation_1 = load_lottiefile('../source/stocks1.json')

		st_lottie(animation_1,
				speed=1,
				reverse=False,
				loop=True,
				quality="high",
				width=500,
				height=450,
				)
		
		# finance_lottie = load_lottieurl("https://app.lottiefiles.com/share/9e58a2cc-e627-4b6a-a0b4-3fa95571236c")


#############################################################################################################################

if selected == 'Live Trading':

# 	st.video('../source/stock_animation.mp4', format='mp4')

# 	import yfinance as yf

# 	tickers = yf.Tickers('msft aapl goog tsla scom coop kcb eqt kq nse bat bamb totl nmg nbk dtk')

# 	# access each ticker 
# 	stock = tickers.tickers[str('nmg').upper()].history(period="max")

# 	df2 = pd.DataFrame(stock).head(1000)

# 	if {'Dividends', 'Stock Splits'}.issubset(df2.columns):
# 		df2.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
	

# 	st.dataframe(df2)
# 	st.write(df2.shape)
# 	st.write(df2.columns)
# 	df2.reset_index(inplace=True)

# 	buffer = io.StringIO()
# 	df2.info(buf=buffer)
# 	info_str = buffer.getvalue()

# 	# Display df.info() output in Streamlit
# 	st.write('Summary of the dataframe')
# 	st.text(info_str)



# 	fig = go.Figure()

# 	fig.add_trace(go.Scatter(x=df2.index, y = df2['High'], mode='lines'))



# 	# Add frames for animation
# 	frames = [go.Frame(data=[go.Scatter(x=df2['Date'][:k+1], y=df2['High'][:k+1])],
# 	                   name=str(df2['Date'].iloc[k])) for k in range(len(df2))]

# 	fig.frames = frames

# 	# Update layout with animation settings
# 	fig.update_layout(
# 	    title=f'High Stocks of the Tech company',
# 	    xaxis_title='Date',
# 	    yaxis_title='High Price',
# 	    legend_title='Company'
# 	    # updatemenus=[dict(type='buttons', showactive=False,
# 	    #                   buttons=[dict(label='Play',
# 	    #                                 method='animate',
# 	    #                                 args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])])],
# 	    # # Automatically start the animation
# 	    # transition={'duration': 100},
# 	    # # frame={'duration': 100, 'redraw': True},
# 	    # sliders=[dict(steps=[dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=50, redraw=True), transition=dict(duration=0))], label=f.name) for f in frames])]
# 	)


# 	st.plotly_chart(fig)


	
##########################################################################################
	

	# Streamlit app
	st.title('Market Security Stocks')

	# User selects a stock symbol
	# stock_symbol = st.selectbox('Select Symbol company', ['AAPL', 'GOOG', 'MSFT'])

	
	with st.form(key="input_parameters"):

		tk = yf.Tickers('msft aapl goog tsla scom coop kcb eqt kq nse bat bamb totl nmg nbk dtk')



		symbol = []
		for i in tk.symbols:
			symbol.append(i)
			symbol.sort(reverse=False)


		ticker = st.selectbox('select ticker symbol', symbol)

		submitted = st.form_submit_button('explore')


		# Fetch the real-time data
		stock = tk.tickers[str(ticker).upper()].history(period="max")

		data = pd.DataFrame(stock)#.head(1000)

		st.write(data.index.max())


		if {'Dividends', 'Stock Splits'}.issubset(data.columns):
			data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)

		today_high = round(data["High"].iloc[0] - data["High"].iloc[1], 2)
		today_open = round(data["Open"].iloc[0] - data["Open"].iloc[1], 2)
		today_high = round(data["High"].iloc[0] - data["High"].iloc[1], 2)

		st.write(f'Market summary > {ticker}')

		trade_col1, trade_col2, trade_col3, trade_col4 = st.columns(4)

		with trade_col1:
			st.metric(label='Net Gain/Loss', value=str(round(data["High"].iloc[0], 2)), delta=str(today_high) + " Today")

		with trade_col2:
			st.metric(label='Open', value=str(round(data["Open"].iloc[0], 2)), delta=str(today_open))

		with trade_col3:
			st.metric(label='Date', value=str(datetime.today().year), delta=str(datetime.today().strftime('%A')))
			# st.write(str(datetime.today().date()))

	
		data.reset_index(inplace=True)

		# Placeholder for the plot
		placeholder = st.empty()


	if submitted or st.session_state.keep_graphics:

		future_prediction(data)
		regression_model(data)

		# Infinite loop to update the plot with real-time data
		while True:
		    # Create the plot
		    fig = go.Figure()
		    
		    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines'))
		    
		    # Update layout
		    fig.update_layout(
		        title=f'{ticker.upper()}',
		        xaxis_title='Time',
		        yaxis_title='Price',
		        legend_title='Stock Symbol'
		    )
		    
		    # Update the plot in the placeholder
		    placeholder.plotly_chart(fig, use_container_width=True)
		    
		    # Wait for a few seconds before updating
		    time.sleep(5)  # Adjust the sleep time as needed

		st.plotly_chart(placeholder)

	
	




if selected ==  'Model':
	pass
	# regression_model(data)
	# subprocess.run([f"{sys.executable}", "../model/regression.py"])
	# st.plotly_chart(fig)

			