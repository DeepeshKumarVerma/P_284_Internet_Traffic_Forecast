import streamlit as st
import pickle
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA from statsmodels
import warnings
warnings.filterwarnings("ignore") 

st.markdown('''
<style>
.stApp {
    
    background-color:#8DC8ED;
    align:center;\
    display:fill;\
    border-radius: false;\
    border-style: solid;\
    border-color:#000000;\
    border-style: false;\
    border-width: 2px;\
    color:Black;\
    font-size:15px;\
    font-family: Source Sans Pro;\
    background-color:#ffb3b3;\
    text-align:center;\
    letter-spacing:0.1px;\
    padding: 0.1em;">\
}
.sidebar {
    background-color: black;
}
.st-b7 {
    color: #8DC8ED;
}
.css-nlntq9 {
    font-family: Source Sans Pro;
}
</style>
''', unsafe_allow_html=True)

pickle_in = open(r"C:\Users\DELL\Desktop\P_284\arima_model1.pkl", "rb")
arima_model = pickle.load(pickle_in)
daily_data_30 = pd.read_csv(r"C:\Users\DELL\Desktop\P_284\daily_data.csv", header=None)

st.title("Forecast Internet Traffic data")
st.sidebar.subheader("Select the number of days to Forecast from 2022-Mar-12")
days = st.sidebar.number_input('Days', min_value=1, step=1)

# Create future dates
future = pd.date_range(start='2022-03-12', periods=days, tz=None, freq='D')
future_df = pd.DataFrame(index=future)

# Initialize the last 7 days data
z = daily_data_30[0].values

# Forecast for the selected number of days
forecast_values = []
for i in range(0, days):
    # Fit ARIMA model to the existing data
    arima_model = ARIMA(z, order=(2, 1, 1))
 # Replace p, d, and q with your model's order
    arima_fit = arima_model.fit()
    
    # Forecast one step ahead
    forecast = arima_fit.forecast()
    forecast_values.append(forecast[0])
    
    # Update the data for the next iteration
    z = np.append(z, forecast[0])

# Update the future_df with the forecasted data
future_df['Internet Traffic'] = forecast_values

# Display the forecast and data
st.sidebar.write(f"Internet Traffic for {days}th day")
st.sidebar.write(future_df[-1:])
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Internet Traffic Forecasted for {days} days" )
    st.write(future_df)
with col2:
    st.subheader('Forecasted Graph')
    fig, ax = plt.subplots()
    plt.figure(figsize=(8, 3))
    ax.plot(future_df.index, future_df['Internet Traffic'], label='Forecast', color="orange")
    ax.tick_params(axis='x', labelrotation=100)
    plt.legend(fontsize=12, fancybox=True, shadow=True, frameon=True)
    plt.ylabel('Internet Traffic', fontsize=15)
    plt.xlabel('Date', fontsize=15)
    st.pyplot(fig)
