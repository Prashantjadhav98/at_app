from flask import Flask, request, jsonify,render_template,send_file,session
import pickle
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
import io
import holidays

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'Prashant@123' 
# Load the SARIMA model from file
with open('sarima_Exog_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    # Get data from JSON request
    return render_template("index.html")

# Define a route for time series forecasting
@app.route('/forecast', methods=['POST'])
def forecast():
    # Get data from JSON request
    Month_ahead = int(request.form['Month_ahead'])

    # Generate future dates for prediction
    last_date = pd.to_datetime('2024-04-30')
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(Month_ahead)]

    # Convert future_dates to a DataFrame
    future_df = pd.DataFrame({'date': future_dates})
    indian_holidays = holidays.India(years=[ 2023, 2024, 2025, 2026, 2027, 2028])
    future_df['holiday'] = future_df['date'].apply(lambda x: 1 if x in indian_holidays else 0)
    
    future_df.set_index('date', inplace=True)

    try:
        forecast = model.get_forecast(steps=Month_ahead, exog=future_df['holiday'])
        forecast_mean = forecast.predicted_mean

        forecast_df = pd.DataFrame({
            'date': [date.strftime('%Y-%m-%d') for date in future_dates],
            'predicted_Price': forecast_mean.values,
        })

        session['forecast_results'] = forecast_df.to_dict(orient='records')
        return render_template('results.html', data=session['forecast_results'])

        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/download_excel')
def download_excel():
    forecast_results = session.get('forecast_results')
    
    if forecast_results is None:
        return jsonify({'error': 'No forecast data available'}), 400
    df = pd.DataFrame(forecast_results)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Forecast', index=False)
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='forecast_results.xlsx'
    )
    

if __name__ == '__main__':
    app.run(debug=True)