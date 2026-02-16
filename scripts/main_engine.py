import os
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from utils import calculate_hri, DATA_PATH, FORECAST_PATH, MODEL_PATH, API_KEY, LAT, LON
from model_trainer import train_model

def run_engine():
    try:
        # A. Fetch Current Data
        aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        
        aqi_res = requests.get(aqi_url).json()
        raw_aqi = aqi_res['list'][0]['components']
        w_res = requests.get(w_url).json()
        
        weather_now = {
            'temp': w_res['main']['temp'],
            'humidity': w_res['main']['humidity'],
            'wind_speed': w_res['wind']['speed'],
            'uv_index': w_res.get('uvi', 3), 
            'precip': w_res.get('rain', {}).get('1h', 0)
        }

    except Exception as e:
        print(f"API Error: {e}")
        return

    # C. Calculate CURRENT HRI (The Truth)
    current_hri = calculate_hri(raw_aqi, weather_now)
    
    # --- NEW: VALIDATION LOGIC (The First Entry Fix) ---
    last_prediction = 0.0
    error_pct = 0.0
    
    if os.path.exists(FORECAST_PATH):
        f_df = pd.read_csv(FORECAST_PATH)
        if not f_df.empty:
            # We look for the prediction meant for this hour
            # For simplicity, we compare against the very first row of the last forecast
            last_prediction = f_df.iloc[0]['predicted_hri']
            error_val = abs(current_hri - last_prediction)
            error_pct = round((error_val / current_hri) * 100, 2) if current_hri != 0 else 0

    # D. Predict FUTURE HRI (24-Hour Timeline)
    forecast_rows = []
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        temp_pollutants = [raw_aqi[p] for p in ['pm2_5', 'pm10', 'no2', 'o3', 'co']]
        
        for i in range(1, 25):
            future_time = datetime.now() + timedelta(hours=i)
            input_data = temp_pollutants + list(weather_now.values()) + [future_time.hour]
            
            preds = model.predict(np.array(input_data).reshape(1, -1))[0]
            pred_aqi_dict = dict(zip(['pm2_5', 'pm10', 'no2', 'o3', 'co'], preds))
            f_hri = calculate_hri(pred_aqi_dict, weather_now)
            
            forecast_rows.append({
                'target_time': future_time.strftime('%Y-%m-%d %H:%M'),
                'predicted_hri': f_hri
            })
            temp_pollutants = preds.tolist()
        
        # Store the full timeline for the website and next-run validation
        pd.DataFrame(forecast_rows).to_csv(FORECAST_PATH, index=False)
        avg_forecast_hri = round(np.mean([r['predicted_hri'] for r in forecast_rows]), 2)
    else:
        avg_forecast_hri = "N/A"

    # E. Data Persistence (Now including Prediction and Error)
    save_data = {
        **raw_aqi, **weather_now, 
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'), 
        'hri': current_hri,
        'predicted_hri': last_prediction,
        'error_pct': error_pct
    }
    
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    pd.DataFrame([save_data]).to_csv(DATA_PATH, mode='a', index=False, header=not os.path.exists(DATA_PATH))

    # Trigger Retrain at Midnight
    if datetime.now().hour == 0 and datetime.now().minute < 60: 
        train_model()

    print(f"\nSTATUS REPORT [{datetime.now().strftime('%H:%M')}]")
    print(f"Current HRI: {current_hri}")
    print(f"Previous Prediction Was: {last_prediction} (Error: {error_pct}%)")
    print(f"New 24-Hour Forecast Avg: {avg_forecast_hri}")

if __name__ == "__main__":
    run_engine()
