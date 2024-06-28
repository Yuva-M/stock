import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to load stock data
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

# Function to train ARIMA model and forecast
def train_and_forecast_arima(data, order, forecast_steps):
    model = ARIMA(data['Close'], order=order)
    model_fit = model.fit()
    historical_forecast = model_fit.predict(start=0, end=len(data)-1)
    future_forecast = model_fit.forecast(steps=forecast_steps)
    return historical_forecast, future_forecast

# Streamlit app
st.title("Stock Price Prediction App")

# User input for stock ticker, date range, and forecast days
ticker = st.text_input("Enter Stock Ticker:", "AAPL")
start_date = st.date_input("Start Date:", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date:", value=pd.to_datetime("2023-01-01"))
forecast_steps = st.slider('Select number of days to forecast', 1, 365, 30)

# Load and display data
data_load_state = st.text('Loading data...')
data = load_data(ticker, start_date, end_date)
data_load_state.text('Loading data...done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title='Stock Closing Prices', xaxis_title='Date', yaxis_title='Close Price')
    st.plotly_chart(fig)

plot_raw_data()

# Train ARIMA model and forecast
st.subheader('ARIMA Model Prediction')
order = (7, 0, 6)  # (p,d,q)
historical_forecast, future_forecast = train_and_forecast_arima(data, order, forecast_steps)

# Create DataFrame for historical and forecasted data
forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_steps)
historical_df = pd.DataFrame({'Date': data['Date'], 'Close': data['Close'], 'Forecast': historical_forecast})
future_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': future_forecast})
combined_df = pd.concat([historical_df, future_df], ignore_index=True)

# Plot predictions along with actual data
def plot_forecast():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Plot historical data
    fig.add_trace(go.Scatter(x=historical_df['Date'], y=historical_df['Close'], mode='lines', name='Actual Price'), secondary_y=False)
    
    # Plot historical forecast data
    fig.add_trace(go.Scatter(x=historical_df['Date'], y=historical_df['Forecast'], mode='lines', name='Historical Forecast'), secondary_y=False)
    
    # Plot future forecast data
    fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Forecast'], mode='lines', name='Future Forecast'), secondary_y=False)
    
    fig.update_layout(title='Stock Price Prediction', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

plot_forecast()
