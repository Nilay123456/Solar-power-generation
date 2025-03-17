import streamlit as st
import pandas as pd
import pickle
import numpy as np
import altair as alt

# Load the transformations and the trained model
with open('yeo_johnson_input.pkl', 'rb') as f:
    yj = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler_transformer = pickle.load(f)

with open('Xg_Boost_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load actual values from saved file

    actual_values = pd.read_csv("solarpowergeneration.csv")  # Load CSV file
    actual_median = actual_values['power-generated'].median()  # Compute median
    

# Streamlit app
st.title('Solar Power Generation Predictor')

# Input fields
distance_to_solar_noon = st.number_input('Distance to Solar Noon (rad)')
temperature = st.number_input('Temperature (°C) - Daily Average')
sky_cover = st.selectbox('Sky Cover', [0, 1, 2, 3, 4])
wind_direction = st.number_input('Wind Direction (°) - Daily Average')
wind_speed = st.number_input('Wind Speed (m/s)')
average_wind_speed_period = st.number_input('Average Wind Speed (m/s) - 3 Hour Measurement')
humidity = st.number_input('Humidity (%)')
average_pressure_period = st.number_input('Average Pressure (inches of Hg) - 3 Hour Measurement')

# Create input dataframe
data = {
    'sky-cover': [sky_cover],
    'distance-to-solar-noon': [distance_to_solar_noon],
    'temperature': [temperature],
    'wind-direction': [wind_direction],
    'wind-speed': [wind_speed],
    'humidity': [humidity],
    'average-wind-speed-(period)': [average_wind_speed_period],
    'average-pressure-(period)': [average_pressure_period],
}
input_data = pd.DataFrame(data)

# Apply transformations
input_data[['wind-direction', 'humidity']] = yj.transform(input_data[['wind-direction', 'humidity']])

scaled_features = ['distance-to-solar-noon', 'temperature', 'wind-direction', 'wind-speed',
                   'humidity', 'average-wind-speed-(period)', 'average-pressure-(period)']

input_data[scaled_features] = scaler_transformer.transform(input_data[scaled_features])

# Ensure correct column order
expected_columns = loaded_model.feature_names_in_
input_data = input_data[expected_columns]

# Make Prediction
predicted_value = loaded_model.predict(input_data)[0]

# Display Prediction
if st.button("Show Result"):
    st.subheader("Predicted Power Generated (J) - 3 Hour Measurement")
    st.write(f"**{predicted_value:.2f}**")

    # Create dataframe for visualization
    chart_data = pd.DataFrame({
        'Type': ['Actual (Median)', 'Predicted'],
        'Power (J)': [actual_median, predicted_value]
    })

    # Create bar chart
    chart = alt.Chart(chart_data).mark_bar().encode(
        x='Type',
        y='Power (J)',
        color='Type'
    ).properties(title="Actual vs Predicted Power Generation")

    st.altair_chart(chart, use_container_width=True)
