import numpy as np
np.float_ = np.float64
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from concurrent.futures import ProcessPoolExecutor, as_completed

# Initialize session state
if 'stocks' not in st.session_state:
    st.session_state.stocks = {'AAPL': None}  # Pre-add Apple stock
if 'years' not in st.session_state:
    st.session_state.years = 1
if 'days' not in st.session_state:
    st.session_state.days = 90

# Function to load data for a single stock
@st.cache_data
def load_data(stock_symbol, start, end):
    try:
        data = yf.download(stock_symbol, start=start, end=end)
        if data.empty:
            return None
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        return data
    except Exception as e:
        return None

# Function to predict stock prices
def predict_stock(stock_symbol, years, days):
    start = datetime.now() - timedelta(days=years*365)
    end = datetime.now()
    data = load_data(stock_symbol, start, end)
    
    if data is None:
        return stock_symbol, None
    
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    
    current_price = data['y'].iloc[-1]
    future_price = forecast['yhat'].iloc[-1]
    lower_bound = forecast['yhat_lower'].iloc[-1]
    upper_bound = forecast['yhat_upper'].iloc[-1]
    
    return stock_symbol, (current_price, future_price, lower_bound, upper_bound, forecast, model)

# Function to calculate and format returns
def calculate_return(current, future):
    return ((future - current) / current) * 100

def color_coded_return(return_value):
    color = 'green' if return_value >= 0 else 'red'
    sign = '+' if return_value >= 0 else '-'
    return f'<span style="color:{color}">{sign}{abs(return_value):.2f}%</span>'

# Streamlit app
st.title("Stock Market Prediction Using Facebook Library Prophet")

# Sidebar inputs
st.sidebar.header("Input Parameters")
new_stock = st.sidebar.text_input("Enter a stock symbol")
add_stock = st.sidebar.button("Add Stock")

if add_stock and new_stock:
    if new_stock not in st.session_state.stocks:
        st.session_state.stocks[new_stock] = None

# Display and allow removal of added stocks
for stock in list(st.session_state.stocks.keys()):
    col1, col2 = st.sidebar.columns([3, 1])
    col1.write(stock)
    if col2.button("❌", key=f"remove_{stock}"):
        del st.session_state.stocks[stock]

# Update parameters
new_years = st.sidebar.slider("Previous Years of data", 1, 15, st.session_state.years)
new_days = st.sidebar.slider("Future Days to predict", 1, 365, st.session_state.days)

# Check if parameters have changed
params_changed = (new_years != st.session_state.years) or (new_days != st.session_state.days)
if params_changed:
    st.session_state.years = new_years
    st.session_state.days = new_days
    # Reset predictions when parameters change
    for stock in st.session_state.stocks:
        st.session_state.stocks[stock] = None

# Predict button
if st.sidebar.button("Predict"):
    stocks_to_predict = [stock for stock, pred in st.session_state.stocks.items() if pred is None]
    
    if stocks_to_predict:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(predict_stock, stock, st.session_state.years, st.session_state.days) 
                       for stock in stocks_to_predict]
            
            for future in as_completed(futures):
                symbol, result = future.result()
                st.session_state.stocks[symbol] = result

# Display results
if st.session_state.stocks:
    data = []
    for symbol, result in st.session_state.stocks.items():
        if result is None:
            data.append([symbol, "❌ Error", "", "", ""])
        else:
            current_price, future_price, lower_bound, upper_bound, _, _ = result
            predicted_return = calculate_return(current_price, future_price)
            lower_return = calculate_return(current_price, lower_bound)
            upper_return = calculate_return(current_price, upper_bound)
            data.append([
                symbol,
                f'${current_price:.2f}',
                color_coded_return(predicted_return),
                color_coded_return(lower_return),
                color_coded_return(upper_return)
            ])

    df = pd.DataFrame(data, columns=['Stock', 'Current Price', 'Predicted Return', 'Lower Bound Return', 'Upper Bound Return'])
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Make rows clickable
    selected_stock = st.selectbox("Select a stock for detailed view:", [""] + list(st.session_state.stocks.keys()))

    if selected_stock != "":
        st.header(f"Detailed View for {selected_stock}")
        result = st.session_state.stocks[selected_stock]
        
        if result is not None:
            _, _, _, _, forecast, model = result

            # Plot forecast
            st.subheader("Forecast Plot")
            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1)

            # Forecast components
            st.subheader("Forecast Components")
            fig2 = model.plot_components(forecast)
            st.write(fig2)
