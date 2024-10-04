from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
import holidays

# Initialize Flask application
app = Flask(__name__)

# Load the SARIMA model from file
with open('sarima_Exog_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route for time series forecasting
@app.route('/forecast', methods=['POST'])
def forecast():
    # Get data from JSON request
    data = request.json

    # Ensure 'months_ahead' exists and is an integer
    if 'Month_ahead' not in data or not isinstance(data['Month_ahead'], int):
        return jsonify({'error': 'Invalid input format, "date" must be an string'}), 400
    
    Month_ahead = data['Month_ahead']
    
    # Generate future dates for prediction
    last_date = pd.to_datetime('2024-04-30')
    future_dates = [last_date +pd.DateOffset(months=i) for i in range(Month_ahead)]  # Forecast for 'months_ahead' months in days


    # Convert future_dates to a DataFrame
    future_df = pd.DataFrame({'date': future_dates})
    indian_holidays = holidays.India(years=[ 2023, 2024, 2025, 2026, 2027, 2028])
    future_df['holiday'] = future_df['date'].apply(lambda x: 1 if x in indian_holidays else 0)
    print(future_df)
    future_df.set_index('date', inplace=True)
    
    try:
        
        forecast = model.get_forecast(steps=Month_ahead, exog=future_df['holiday'])
        forecast_mean = forecast.predicted_mean
        
       
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_production': forecast_mean.values,
        })
        print('forecast_df :',forecast_df)

    
        forecast_results = forecast_df.to_dict(orient='records')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    return jsonify(forecast_results)

if __name__ == '__main__':
    app.run(debug=True)