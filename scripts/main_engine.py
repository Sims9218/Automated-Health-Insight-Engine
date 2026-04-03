import os
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta, datetime
from scripts.utils import (
    MODEL_PATH,
    calculate_hri, calculate_advice, get_metric,   # [ADDED] calculate_advice import
    now_ist, ts_to_ist, API_KEY, supabase
)

POLLUTANTS = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
WEATHER_COLS = ['temp', 'humidity', 'wind_speed', 'uv_index', 'precip']
FEATURE_COLS = [f'{p}_lag' for p in POLLUTANTS] + WEATHER_COLS + ['hour']


def run_engine():
    model = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

    cities = supabase.table("cities").select("*").execute().data
    city_state = {}

    # FETCH CURRENT ACTUAL DATA
    for row in cities:
        CITY = row["city"]
        LAT = row["lat"]
        LON = row["lon"]
        try:
            aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
            w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

            aqi_res = requests.get(aqi_url).json()
            raw_aqi_full = aqi_res['list'][0]['components']
            raw_aqi = {p: raw_aqi_full[p] for p in POLLUTANTS}
            raw_aqi_save = {k: round(v, 2) for k, v in raw_aqi_full.items()}

            # [ADDED] Extract the real OWM AQI index (1–5 scale) from the response
            # OWM returns: 1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor
            owm_aqi_index = aqi_res['list'][0].get('main', {}).get('aqi', None)

            dt_ist = ts_to_ist(aqi_res['list'][0]['dt'])
            synced_dt = (dt_ist + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
            city_state[CITY] = {
                "raw_aqi": raw_aqi,
                "synced_dt": synced_dt
            }
            observation_time = synced_dt.strftime('%Y-%m-%d %H:%M')

            w_res = requests.get(w_url).json()
            current_hour = synced_dt.hour
            uv_fallback = 0 if (current_hour >= 20 or current_hour < 6) else 4
            weather_now = {
                'temp':       round(w_res['main']['temp'], 2),
                'humidity':   w_res['main']['humidity'],
                'wind_speed': round(w_res['wind']['speed'], 2),
                'uv_index':   w_res.get('uyi', uv_fallback),
                'precip':     round(w_res.get('rain', {}).get('1h', 0), 2),
            }
        except Exception as e:
            print(f"API Error for {CITY}: {e}")
            continue

        # CALCULATE HRI
        current_hri = calculate_hri(raw_aqi, weather_now)
        metric = get_metric(current_hri)

        # [ADDED] Generate layered advice using the new calculate_advice function
        # This uses pollutants + weather + hri + today's date for festival detection
        advice = calculate_advice(raw_aqi, weather_now, current_hri, dt=synced_dt)

        predicted_hri, error_pct = 0.0, 0.0

        if supabase:
            try:
                response = supabase.table("forecast") \
                    .select("hri") \
                    .eq("timestamp", observation_time) \
                    .eq("city", CITY) \
                    .execute()

                if response.data:
                    predicted_hri = round(float(response.data[0]["hri"]), 2)
                    if current_hri > 0:
                        error_pct = round(abs((current_hri - predicted_hri) / current_hri) * 100, 2)

            except Exception as e:
                print(f"Forecast lookup failed: {e}")

        save_data = {
            **{k: round(v, 2) for k, v in raw_aqi_save.items()},
            **weather_now,
            'timestamp':     observation_time,
            'hri':           current_hri,
            'predicted_hri': predicted_hri,
            'error_pct':     error_pct,
            'metric':        metric,
            'city':          CITY,
            # [ADDED] Save real OWM AQI index (1–5) to DB
            'aqi':           owm_aqi_index,
        }

        # SAVE TO HISTORY TABLE
        supabase.table("history").insert({
            "city":          save_data['city'],
            "timestamp":     save_data['timestamp'],
            "pm2_5":         save_data['pm2_5'],
            "pm10":          save_data['pm10'],
            "no2":           save_data['no2'],
            "o3":            save_data['o3'],
            "co":            save_data['co'],
            "temp":          save_data['temp'],
            "humidity":      save_data['humidity'],
            "wind_speed":    save_data['wind_speed'],
            "uv_index":      save_data['uv_index'],
            "precip":        save_data['precip'],
            "hri":           save_data['hri'],
            "predicted_hri": save_data['predicted_hri'],
            "error_pct":     save_data['error_pct'],
            "metric":        save_data['metric'],
            # [ADDED] Persist real OWM AQI index into history row
            "aqi":           save_data['aqi'],
        }).execute()

        # [ADDED] Store advice and city_state so /latest-hri can serve it
        # We cache it in city_state so app.py can access it via run_engine return
        city_state[CITY]["advice"] = advice
        city_state[CITY]["weather_now"] = weather_now
        city_state[CITY]["current_hri"] = current_hri
        city_state[CITY]["metric"] = metric
        city_state[CITY]["aqi"] = owm_aqi_index
        city_state[CITY]["save_data"] = save_data

        print(f"HRI: {current_hri} ({metric})")

    # CLEAR OLD FORECASTS
    if supabase:
        try:
            supabase.table("forecast").delete().not_.is_("timestamp", None).execute()
            print("Old forecasts cleared.")
        except Exception as e:
            print(f"Forecast clear failed: {e}")

    # GENERATE NEXT-DAY FORECAST
    for row in cities:
        CITY = row["city"]
        LAT = row["lat"]
        LON = row["lon"]

        if CITY not in city_state:
            continue

        raw_aqi = city_state[CITY]["raw_aqi"]
        synced_dt = city_state[CITY]["synced_dt"]

        if model:
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

                    # [ADDED] Generate advice for each forecast slot as well
                    f_advice = calculate_advice(pred_aqi_dict, f_weather, f_hri, dt=f_dt)

                    forecast_rows.append({
                        'timestamp': f_dt_str,
                        **pred_aqi_dict,
                        **f_weather,
                        'hri':    f_hri,
                        'metric': get_metric(f_hri),
                        'city':   CITY,
                        # [ADDED] Store serialised advice in forecast row
                        # Stored as JSON string; app.py will parse it when serving
                        'advice': str(f_advice),
                    })

                last_pollutants = preds

            if supabase and forecast_rows:
                try:
                    supabase.table("forecast").insert(forecast_rows).execute()
                    print(f"Next-day forecast saved: {len(forecast_rows)} rows.")
                except Exception as e:
                    print(f"Failed to save forecast: {e}")

        print(f"{CITY} run complete at {now_ist().strftime('%Y-%m-%d %H:%M IST')}")

    # Trigger daily retrain at midnight
    now = now_ist()
    if now.hour == 0 and now.minute < 60:
        from model_trainer import train_model
        train_model()

    # [ADDED] Return city_state so app.py can use cached advice without re-fetching
    return city_state


if __name__ == "__main__":
    run_engine()
