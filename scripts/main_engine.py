import os
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from utils import (
    DATA_PATH, FORECAST_PATH, MODEL_PATH,
    calculate_hri, get_metric, now_ist, ts_to_ist, API_KEY, LAT, LON
)

# Explicit, ordered feature list which must match model_trainer.py exactly.
POLLUTANTS = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
WEATHER_COLS = ['temp', 'humidity', 'wind_speed', 'uv_index', 'precip']
FEATURE_COLS = [f'{p}_lag' for p in POLLUTANTS] + WEATHER_COLS + ['hour']


HISTORY_COLS = [
    'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3',
    'temp', 'humidity', 'wind_speed', 'uv_index', 'precip',
    'timestamp', 'hri', 'predicted_hri', 'error_pct', 'metric'
]  # to ensure proper format saving

def run_engine():
    #  FETCH CURRENT ACTUAL DATA 
    try:
        aqi_url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={LAT}&lon={LON}&appid={API_KEY}"
        )
        w_url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        )

        aqi_res = requests.get(aqi_url).json()
        raw_aqi_full = aqi_res['list'][0]['components']

        # Only keep the 5 pollutants we model
        raw_aqi = {p: raw_aqi_full[p] for p in POLLUTANTS}  # used for model/HRI
        raw_aqi_save = {k: round(v, 2) for k, v in raw_aqi_full.items()}  # full data for CSV

        # BUG FIX: Convert Unix timestamp to IST, then round to nearest hour.
        dt_ist = ts_to_ist(aqi_res['list'][0]['dt'])
        synced_dt = (dt_ist + timedelta(minutes=30)).replace(
            minute=0, second=0, microsecond=0
        )
        observation_time = synced_dt.strftime('%Y-%m-%d %H:%M')

        w_res = requests.get(w_url).json()
        # BUG FIX: uvi is NOT reliably in /data/2.5/weather (it's in One Call API).
        # Fall back to 0 at night (hour 20-6) and a moderate 4 during day.
        current_hour = synced_dt.hour
        uv_fallback = 0 if (current_hour >= 20 or current_hour < 6) else 4
        weather_now = {
            'temp':       round(w_res['main']['temp'], 2),
            'humidity':   w_res['main']['humidity'],
            'wind_speed': round(w_res['wind']['speed'], 2),
            'uv_index':   w_res.get('uvi', uv_fallback),
            'precip':     round(w_res.get('rain', {}).get('1h', 0), 2),
        }

    except Exception as e:
        print(f"API Error: {e}")
        return

    #  CALCULATE HRI AND SAVE TO HISTORY 
    current_hri = calculate_hri(raw_aqi, weather_now)
    metric = get_metric(current_hri)

    # BUG FIX: CSV column was named 'hri_predict'/'error', not 'predicted_hri'/'error_pct'.
    # Standardised to 'predicted_hri' and 'error_pct' here and in the CSV.
    predicted_hri = 0.0
    error_pct = 0.0

    if os.path.exists(FORECAST_PATH):
        try:
            df_forecast = pd.read_csv(FORECAST_PATH)
            match = df_forecast[df_forecast['timestamp'] == observation_time]
            if not match.empty:
                predicted_hri = round(float(match.iloc[0]['hri']), 2)
                if current_hri > 0:
                    error_pct = round(
                        abs((current_hri - predicted_hri) / current_hri) * 100, 2
                    )
        except Exception:
            pass

    save_data = {
        **{k: round(v, 2) for k, v in raw_aqi_save.items()},
        **weather_now,
        'timestamp':     observation_time,
        'hri':           current_hri,
        'predicted_hri': predicted_hri,
        'error_pct':     error_pct,
        'metric':        metric,
    }
    history_exists = os.path.exists(DATA_PATH)
    pd.DataFrame([save_data]).reindex(columns=HISTORY_COLS).to_csv(
    DATA_PATH, mode='a', index=False, header=not history_exists
    )

#  GENERATE 24-HOUR MULTI-OUTPUT FORECAST (8 PARTS)
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

        f_url = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        )
        f_slots = requests.get(f_url).json()['list'][:9] 

        start_hour = synced_dt + timedelta(hours=1)
        forecast_hours = [start_hour + timedelta(hours=i) for i in range(24)]

        slot_weather = {}
        for slot in f_slots:
            slot_dt = ts_to_ist(slot['dt']).replace(minute=0, second=0, microsecond=0)
            slot_weather[slot_dt] = {
                'temp': round(slot['main']['temp'], 2),
                'humidity': slot['main']['humidity'],
                'wind_speed': round(slot['wind']['speed'], 2),
                'uv_index': 0,
                'precip': round(slot.get('rain', {}).get('3h', 0) / 3, 2),
            }

        def get_weather_for_hour(dt):
            closest = min(slot_weather.keys(), key=lambda s: abs((s - dt).total_seconds()))
            w = slot_weather[closest].copy()
            h = dt.hour
            if 6 <= h <= 18:
                w['uv_index'] = round(max(0, 6 * np.sin(np.pi * (h - 6) / 12)), 1)
            else:
                w['uv_index'] = 0
            return w

        last_pollutants = [raw_aqi[p] for p in POLLUTANTS]
        hourly_results = {} 

        for i, f_dt in enumerate(forecast_hours, 1):
            f_weather = get_weather_for_hour(f_dt)
            X_input = last_pollutants + [f_weather[c] for c in WEATHER_COLS] + [f_dt.hour]
            
            raw_preds = model.predict(pd.DataFrame([X_input], columns=FEATURE_COLS))[0]
            preds = [round(float(p), 2) for p in raw_preds]

            pred_aqi_dict = dict(zip(POLLUTANTS, preds))
            f_hri = calculate_hri(pred_aqi_dict, f_weather)
            
            hourly_results[i] = {
                'timestamp': f_dt.strftime('%Y-%m-%d %H:%M'),
                'hri': f_hri,
                'metric': get_metric(f_hri)
            }
            last_pollutants = preds 

        divided_forecast = []
        intervals = [3, 6, 9, 12, 15, 18, 21, 24]
        
        for idx, t in enumerate(intervals, 1):
            part_data = {
                'part': idx,
                'interval_hour': t,
                'timestamp_t': hourly_results[t]['timestamp'],
                'hri_prev': hourly_results.get(t-1, {}).get('hri'),
                'hri_target': hourly_results[t]['hri'],
                'hri_next': hourly_results.get(t+1, {}).get('hri') # Returns None for 24+1
            }
            divided_forecast.append(part_data)

        pd.DataFrame(divided_forecast).to_csv(FORECAST_PATH, index=False)
        print(f"8-Part Forecast (24h limit) saved to {FORECAST_PATH}")

    # Retrain model once a day at midnight IST
    now = now_ist()
    if now.hour == 0 and now.minute < 60:
        from model_trainer import train_model
        train_model()

    print(f"STATUS: Engine run complete at {now_ist().strftime('%Y-%m-%d %H:%M IST')}")
    print(f"HRI: {current_hri} ({metric})")


if __name__ == "__main__":
    run_engine()
