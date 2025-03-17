import streamlit as st
import pandas as pd
import pickle

# Load the transformations and the trained model
with open('yeo_johnson_input.pkl', 'rb') as f:
    yj = pickle.load(f)
  
with open('scaler.pkl', 'rb') as f:
    scaler_transformer = pickle.load(f)

with open('Xg_Boost_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Streamlit app
st.title('Solar Power Generation Predictor')

distance_to_solar_noon = st.number_input('Distance to Solar Noon (rad)')
temperature = st.number_input('Temperature (°C) - Daily Average')
sky_cover = st.selectbox('Sky Cover', [0, 1, 2, 3, 4])
wind_direction = st.number_input('Wind Direction (°) - Daily Average')
wind_speed = st.number_input('Wind Speed (m/s)')
average_wind_speed_period = st.number_input('Average Wind Speed (m/s) - 3 Hour Measurement')
humidity = st.number_input('Humidity (%)')
average_pressure_period = st.number_input('Average Pressure (inches of Hg) - 3 Hour Measurement')

    
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

input_data = pd.DataFrame(data, index=[0])

st.write("Raw Input Data Before Transformation:")
st.write(input_data)


# Apply Yeo-Johnson transformation on 'wind-direction' and 'humidity'
input_data[['wind-direction', 'humidity']] = yj.transform(input_data[['wind-direction', 'humidity']])

st.write("After Yeo-Johnson Transformation:")
st.write(input_data)

# List of columns to apply scaler (excluding 'sky-cover')
scaled_features = ['distance-to-solar-noon', 'temperature', 'wind-direction', 'wind-speed',
                    'humidity', 'average-wind-speed-(period)', 'average-pressure-(period)']

# Apply scaling
input_data[scaled_features] = scaler_transformer.transform(input_data[scaled_features])


st.write("After Scaling Transformation:")
st.write(input_data[scaled_features])

# input_data = input_data.drop(['wind-direction','humidity'],axis=1)

# ✅ Ensure input_data has the correct column order before making predictions
expected_columns = loaded_model.feature_names_in_
input_data = input_data[expected_columns]  # Reorder columns to match training data

st.write("After Scaling Transformation - colums arrange:")
st.write(input_data)

# Predict 
transformed_prediction = loaded_model.predict(input_data)  # Get transformed target prediction

# Show result
if st.button("Show Result"):
    st.subheader("Predicted Power Generated (J) - 3 Hour Measurement")
    st.write(f"**{transformed_prediction[0]:.2f}**")
