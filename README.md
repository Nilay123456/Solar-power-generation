# Solar Power Generation Prediction

A machine learning project that predicts solar power generation from environmental and time-based features such as sky cover, temperature, humidity, wind speed, pressure, and distance to solar noon.

## Demo

- Live app: https://solar-power-generation-2jcmmsfw6ftjhqzs7xwf5b.streamlit.app/
- Repository: https://github.com/Nilay123456/Solar-power-generation

## Problem

Solar power output changes with weather and sunlight conditions. The goal of this project is to estimate generated power from measurable inputs so the model can support planning, monitoring, and renewable-energy analysis.

This is a supervised regression problem because the target value, power generated, is continuous.

## Features Used

```text
sky-cover
distance-to-solar-noon
temperature
wind-direction
wind-speed
humidity
average-wind-speed-(period)
average-pressure-(period)
```

## Approach

```text
1. Load and inspect the dataset.
2. Clean and preprocess environmental features.
3. Explore relationships between weather/time features and generated power.
4. Apply Yeo-Johnson transformation to skewed inputs where needed.
5. Scale numeric features consistently.
6. Compare regression models.
7. Select XGBoost for non-linear tabular regression performance.
8. Save the trained model and preprocessing artifacts.
9. Deploy a Streamlit app for real-time prediction.
```

## Tech Stack

```text
Python
Pandas
NumPy
scikit-learn
XGBoost
Streamlit
Pickle model artifacts
```

## Model And Preprocessing

The deployed app loads the same preprocessing objects used during training:

```text
yeo_johnson_input.pkl
scaler.pkl
Xg_Boost_model.pkl
```

This matters because model inference must use the same transformations and feature order as training. If preprocessing differs between training and deployment, predictions become unreliable even if the model itself is good.

## How To Run Locally

```powershell
pip install -r requirements.txt
streamlit run solar_main.py
```

## What I Learned

- How to frame a real-world forecasting task as supervised regression.
- Why feature preprocessing must be reused exactly at inference time.
- Why tree-based models such as XGBoost often work well on tabular data.
- How to turn a trained model into a usable deployed app.

## Limitations

- The app is a demo, not a production forecasting platform.
- Prediction uncertainty is not currently shown.
- The README should be updated with exact evaluation metrics from the notebook.
- Input ranges should be constrained to realistic values.
- A production version would add monitoring, model versioning, drift checks, and periodic retraining.

## Production Improvements

```text
1. Add model metrics: R-squared, MAE, RMSE.
2. Add input validation and realistic value ranges.
3. Add feature importance charts.
4. Add model/version metadata in the UI.
5. Add tests for preprocessing and inference.
6. Split the model into a proper API service if used beyond a demo.
```
