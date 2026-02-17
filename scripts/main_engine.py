import os
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from utils import DATA_PATH, FORECAST_PATH, MODEL_PATH, calculate_hri

API_KEY = os.getenv("API_KEY")
LAT, LON = "19.07", "72.87"

def run_engine():
    # --- 1. FETCH CURRENT ACTUAL DATA ---
    try:
        aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        
        aqi_res = requests.get(aqi_url).json()
        raw_aqi = aqi_res['list'][0]['components'] # co, no, no2, o3, so2, pm2_5, pm10, nh3
        
        w_res = requests.get(w_url).json()
        weather_now = {
            'temp': w_res['main']['temp'],
            'humidity': w_res['main']['humidity'],
            'wind_speed': w_res['wind']['speed'],
            'uv_index': w_res.get('uvi', 3),
            'precip': w_res.get('rain', {}).get('1h', 0)
        }
    except Exception as e:
        print(f"API Error: {e}"); return

    # --- 2. SAVE ACTUAL DATA TO HISTORY ---
    current_hri = calculate_hri(raw_aqi, weather_now)
    save_data = {
        **raw_aqi, **weather_now, 
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'), 
        'hri': current_hri
    }
    
    # Append to comprehensive_history.csv
    pd.DataFrame([save_data]).to_csv(DATA_PATH, mode='a', index=False, header=not os.path.exists(DATA_PATH))

    # --- 3. GENERATE 24-HOUR MULTI-OUTPUT FORECAST ---
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        
        # A. Fetch 5-Day/3-Hour Forecast API
        f_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        f_res = requests.get(f_url).json()['list'][:8] # Next 8 blocks = 24 hours
        
        # B. Setup for recursive prediction (5 pollutants used in training)
        pollutant_list = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
        last_pollutants = [raw_aqi[p] for p in pollutant_list]
        forecast_rows = []

        for hour_data in f_res:
            # OpenWeather gives 3-hour chunks; we loop 3 times to get hourly granularity
            for sub_hour in range(3):
                f_time = datetime.fromtimestamp(hour_data['dt']) + timedelta(hours=sub_hour)
                
                f_weather = {
                    'temp': hour_data['main']['temp'],
                    'humidity': hour_data['main']['humidity'],
                    'wind_speed': hour_data['wind']['speed'],
                    'uv_index': 3, # Defaulting UV for forecast
                    'precip': hour_data.get('rain', {}).get('3h', 0) / 3
                }

                # Input Order: [Pollutants_Lag] + [temp, humidity, wind, uv, precip] + [hour]
                X_input = last_pollutants + list(f_weather.values()) + [f_time.hour]
                raw_preds = model.predict(np.array(X_input).reshape(1, -1))[0]
                preds = [round(float(p), 2) for p in raw_preds]
                # Map predictions back to names
                pred_aqi_dict = dict(zip(pollutant_list, preds))
                
                # Calculate HRI for this forecasted hour
                f_hri = calculate_hri(pred_aqi_dict, f_weather)

                # BUILD THE FULL ROW FOR forecast_timeline.csv
                forecast_rows.append({
                    'timestamp': f_time.strftime('%Y-%m-%d %H:%M'),
                    **pred_aqi_dict,
                    **f_weather,
                    'hri': f_hri
                })
                
                # Update lag for the next hour prediction
                last_pollutants = preds

        # C. Save the complete 24-hour timeline
        pd.DataFrame(forecast_rows[:24]).to_csv(FORECAST_PATH, index=False)

    # Trigger Retrain at Midnight
    if datetime.now().hour == 0 and datetime.now().minute < 60:
        from model_trainer import train_model
        train_model()

    print(f"STATUS: History & Full Multi-Output Forecast Updated [{datetime.now().strftime('%H:%M')}]")

if __name__ == "__main__":
    run_engine()
