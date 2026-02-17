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
        raw_aqi = aqi_res['list'][0]['components']
        
        # --- FIX: ROUND ACTUAL TIME TO NEAREST HOUR ---
        # Get raw timestamp from API
        dt_object = datetime.fromtimestamp(aqi_res['list'][0]['dt'])
        # Rounding logic: if > 30 mins, go to next hour, else stay at current hour
        synced_dt = (dt_object + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
        observation_time = synced_dt.strftime('%Y-%m-%d %H:%M')
        
        w_res = requests.get(w_url).json()
        weather_now = {
            'temp': round(w_res['main']['temp'], 2),
            'humidity': w_res['main']['humidity'],
            'wind_speed': round(w_res['wind']['speed'], 2),
            'uv_index': w_res.get('uvi', 3),
            'precip': w_res.get('rain', {}).get('1h', 0)
        }
    except Exception as e:
        print(f"API Error: {e}"); return

    # --- 2. SAVE ACTUAL DATA TO HISTORY (WITH ERROR CALC) ---
    current_hri = calculate_hri(raw_aqi, weather_now)
    
    predicted_hri = 0
    error_pct = 0
    
    # Match with existing forecast using the NEW rounded timestamp
    if os.path.exists(FORECAST_PATH):
        try:
            df_forecast = pd.read_csv(FORECAST_PATH)
            match = df_forecast[df_forecast['timestamp'] == observation_time]
            if not match.empty:
                predicted_hri = round(float(match.iloc[0]['hri']), 2)
                if current_hri > 0:
                    error_pct = round(abs((current_hri - predicted_hri) / current_hri) * 100, 2)
        except: pass

    save_data = {
        **{k: round(v, 2) for k, v in raw_aqi.items()}, 
        **weather_now, 
        'timestamp': observation_time,
        'hri': current_hri,
        'predicted_hri': predicted_hri,
        'error_pct': error_pct
    }
    pd.DataFrame([save_data]).to_csv(DATA_PATH, mode='a', index=False, header=not os.path.exists(DATA_PATH))

    # --- 3. GENERATE 24-HOUR MULTI-OUTPUT FORECAST ---
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        f_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        f_res = requests.get(f_url).json()['list'][:8]
        
        pollutant_list = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
        last_pollutants = [raw_aqi[p] for p in pollutant_list]
        forecast_rows = []

        for hour_data in f_res:
            for sub_hour in range(3):
                # Ensure forecast time is also perfectly at HH:00
                f_dt = (datetime.fromtimestamp(hour_data['dt']) + timedelta(hours=sub_hour)).replace(minute=0, second=0, microsecond=0)
                
                f_weather = {
                    'temp': round(hour_data['main']['temp'], 2),
                    'humidity': hour_data['main']['humidity'],
                    'wind_speed': round(hour_data['wind']['speed'], 2),
                    'uv_index': 3,
                    'precip': round(hour_data.get('rain', {}).get('3h', 0) / 3, 2)
                }

                X_input = last_pollutants + list(f_weather.values()) + [f_dt.hour]
                raw_preds = model.predict(np.array(X_input).reshape(1, -1))[0]
                preds = [round(float(p), 2) for p in raw_preds]
                
                pred_aqi_dict = dict(zip(pollutant_list, preds))
                f_hri = calculate_hri(pred_aqi_dict, f_weather)

                forecast_rows.append({
                    'timestamp': f_dt.strftime('%Y-%m-%d %H:%M'),
                    **pred_aqi_dict,
                    **f_weather,
                    'hri': f_hri
                })
                last_pollutants = preds 

        pd.DataFrame(forecast_rows[:24]).to_csv(FORECAST_PATH, index=False)

    if datetime.now().hour == 0 and datetime.now().minute < 60:
        from model_trainer import train_model
        train_model()

    print(f"STATUS: Engine Run Complete at {datetime.now().strftime('%H:%M')}")

if __name__ == "__main__":
    run_engine()
