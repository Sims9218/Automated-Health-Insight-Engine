import os
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta, datetime
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
] # to ensure proper format saving

def run_engine():
    # FETCH CURRENT ACTUAL DATA
    try:
        aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

        aqi_res = requests.get(aqi_url).json()
        raw_aqi_full = aqi_res['list'][0]['components']
        raw_aqi = {p: raw_aqi_full[p] for p in POLLUTANTS}
        raw_aqi_save = {k: round(v, 2) for k, v in raw_aqi_full.items()}

        dt_ist = ts_to_ist(aqi_res['list'][0]['dt'])
        synced_dt = (dt_ist + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
        observation_time = synced_dt.strftime('%Y-%m-%d %H:%M')

        w_res = requests.get(w_url).json()
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

    # CALCULATE HRI AND SAVE TO HISTORY
    current_hri = calculate_hri(raw_aqi, weather_now)
    metric = get_metric(current_hri)
    predicted_hri, error_pct = 0.0, 0.0

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
        **{k: round(v, 2) for k, v in raw_aqi_save.items()},
        **weather_now,
        'timestamp': observation_time,
        'hri': current_hri,
        'predicted_hri': predicted_hri,
        'error_pct': error_pct,
        'metric': metric,
    }
    history_exists = os.path.exists(DATA_PATH)
    pd.DataFrame([save_data]).reindex(columns=HISTORY_COLS).to_csv(DATA_PATH, mode='a', index=False, header=not history_exists)

    # GENERATE 24-HOUR MULTI-OUTPUT FORECAST 
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        
        f_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        f_res = requests.get(f_url).json()
        f_slots = f_res.get('list', [])[:24] 

        slot_weather = {}
        for slot in f_slots:
            slot_dt = ts_to_ist(slot['dt']).replace(minute=0, second=0, microsecond=0)
            slot_weather[slot_dt] = {
                'temp':       round(slot['main']['temp'], 2),
                'humidity':   slot['main']['humidity'],
                'wind_speed': round(slot['wind']['speed'], 2),
                'uv_index':   0, 
                'precip':     round(slot.get('rain', {}).get('3h', 0) / 3, 2),
            }

        def get_weather_for_hour(dt):
            closest = min(slot_weather.keys(), key=lambda s: abs((s - dt).total_seconds()))
            w = slot_weather[closest].copy()
            h = dt.hour
            w['uv_index'] = round(max(0, 6 * np.sin(np.pi * (h - 6) / 12)), 1) if 6 <= h <= 18 else 0
            return w

        tomorrow_start = (synced_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        key_hours = [0, 3, 6, 9, 12, 15, 18, 21]
        
        target_timestamps = []
        for kh in key_hours:
            base_dt = tomorrow_start + timedelta(hours=kh)
            target_timestamps.extend([
                (base_dt - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M'),
                base_dt.strftime('%Y-%m-%d %H:%M'),
                (base_dt + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')
            ])
        target_timestamps = sorted(list(set(target_timestamps)))

        max_target_dt = datetime.strptime(target_timestamps[-1], '%Y-%m-%d %H:%M').replace(tzinfo=synced_dt.tzinfo)
        hours_to_predict = int((max_target_dt - synced_dt).total_seconds() // 3600)

        last_pollutants = [raw_aqi[p] for p in POLLUTANTS]
        forecast_rows = []

        for i in range(1, hours_to_predict + 1):
            f_dt = synced_dt + timedelta(hours=i)
            f_dt_str = f_dt.strftime('%Y-%m-%d %H:%M')
            f_weather = get_weather_for_hour(f_dt)

            X_input = last_pollutants + [f_weather[c] for c in WEATHER_COLS] + [f_dt.hour]
            raw_preds = model.predict(pd.DataFrame([X_input], columns=FEATURE_COLS))[0]
            preds = [round(float(p), 2) for p in raw_preds]

            if f_dt_str in target_timestamps:
                pred_aqi_dict = dict(zip(POLLUTANTS, preds))
                f_hri = calculate_hri(pred_aqi_dict, f_weather)
                forecast_rows.append({
                    'timestamp': f_dt_str,
                    **pred_aqi_dict,
                    **f_weather,
                    'hri': f_hri,
                    'metric': get_metric(f_hri),
                })
            
            last_pollutants = preds 

        pd.DataFrame(forecast_rows).to_csv(FORECAST_PATH, index=False)
        print(f"Next-day forecast saved: {len(forecast_rows)} specific samples for tomorrow.")

    now = now_ist()
    if now.hour == 0 and now.minute < 60:
        from model_trainer import train_model
        train_model()

    print(f"STATUS: Engine run complete at {now_ist().strftime('%Y-%m-%d %H:%M IST')}")
    print(f"HRI: {current_hri} ({metric})")

if __name__ == "__main__":
    run_engine()
