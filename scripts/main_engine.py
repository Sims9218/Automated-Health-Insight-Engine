import os
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta # CHANGED: Added timedelta for 24h loop
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
API_KEY = os.getenv("API_KEY")
LAT, LON = "19.07", "72.87"
DATA_PATH = 'data/comprehensive_history.csv'
MODEL_PATH = 'models/specialist_model.pkl'

LIMITS = {'pm2_5': 25, 'pm10': 50, 'no2': 40, 'o3': 100, 'co': 10}
WEIGHTS = {'pm2_5': 0.4, 'pm10': 0.2, 'no2': 0.15, 'o3': 0.15, 'co': 0.1}

# --- 1. THE CALCULATION PART
def calculate_hri(aqi_data, weather_data):
    adjusted = aqi_data.copy()
    
    h = weather_data.get('humidity', 50)
    w = weather_data.get('wind_speed', 3)
    r = weather_data.get('precip', 0)
    uv = weather_data.get('uv_index', 3)

    # Humidity Adjustments
    if h > 75:
        adjusted['pm2_5'] *= 1.10
        adjusted['pm10'] *= 1.08
        adjusted['no2'] *= 1.03
    
    # Wind Adjustments (Dispersion and accumilation)
    if w < 2:
        for p in ['pm2_5', 'pm10', 'no2', 'co']: adjusted[p] *= 1.10
    elif w > 5:
        adjusted['pm2_5'] *= 0.70
        adjusted['pm10'] *= 0.65
        adjusted['no2'] *= 0.85
        adjusted['co'] *= 0.90

    # Precipitation Adjustments (Washout effect)
    if r > 0.5:
        adjusted['pm2_5'] *= 0.75
        adjusted['pm10'] *= 0.70
        if r > 5:
            adjusted['no2'] *= 0.75
            adjusted['co'] *= 0.80

    # UV Adjustments (Photochemical)
    if uv > 6:
        adjusted['o3'] *= 1.20
        adjusted['no2'] *= 0.95

    # Final Weighted Sum
    hri_score = sum((adjusted.get(k, 0) / LIMITS[k]) * WEIGHTS[k] for k in WEIGHTS)
    return round(hri_score * 100, 2)

# --- 2. THE BRAIN BLACK BOX SHIT
def train_model():
    if not os.path.exists(DATA_PATH): return
    df = pd.read_csv(DATA_PATH)
    if len(df) < 48: return 

    # Feature Engineering: Lagged pollutants + Time + Weather
    pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
    weather_cols = ['temp', 'humidity', 'wind_speed', 'uv_index', 'precip']
    
    # CHANGED: Explicitly define feature order to ensure consistency
    for p in pollutants:
        df[f'{p}_lag'] = df[p].shift(1)
    
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df = df.dropna()

    feature_cols = [f'{p}_lag' for p in pollutants] + weather_cols + ['hour']
    X = df[feature_cols]
    y = df[pollutants]

    # CHRONOLOGICAL Split (No Shuffling for Time Series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    # MultiOutput Specialist Regressor  , 100 trees with just 6 depth , using multioutputblah as wrapper such that it makes mini models for individual features
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
    model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True) # CHANGED: Ensure dir exists
    joblib.dump(model, MODEL_PATH)
    print(f"Model Retrained at {datetime.now()}. Data points: {len(df)}")

# --- 3. THE ENGINE ---
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
            'uv_index': w_res.get('uvi', 3), # CHANGED: Default to 3 if missing
            'precip': w_res.get('rain', {}).get('1h', 0)
        }

    except Exception as e:
        print(f"API Error: {e}")
        return

    # C. Calculate CURRENT HRI (The Truth)
    current_hri = calculate_hri(raw_aqi, weather_now)
    
    # D. Predict FUTURE HRI (24-Hour Rolling Forecast)
    # CHANGED: Now loops 24 times to give a full day outlook
    forecast_results = []
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        temp_pollutants = [raw_aqi[p] for p in ['pm2_5', 'pm10', 'no2', 'o3', 'co']]
        
        for i in range(1, 25):
            future_time = datetime.now() + timedelta(hours=i)
            # Input: [Pollutants_Lag] + [Weather] + [Hour]
            input_data = temp_pollutants + list(weather_now.values()) + [future_time.hour]
            
            preds = model.predict(np.array(input_data).reshape(1, -1))[0]
            
            # Map for HRI calc
            pred_aqi_dict = dict(zip(['pm2_5', 'pm10', 'no2', 'o3', 'co'], preds))
            f_hri = calculate_hri(pred_aqi_dict, weather_now) # Using current weather as proxy
            
            forecast_results.append(f_hri)
            temp_pollutants = preds.tolist() # Feed guess back in
        
        avg_forecast_hri = round(np.mean(forecast_results), 2)
    else:
        avg_forecast_hri = "N/A"

    # E. Data Persistence & Retrain Trigger
    save_data = {**raw_aqi, **weather_now, 'timestamp': datetime.now(), 'hri': current_hri}
    os.makedirs('data', exist_ok=True)
    pd.DataFrame([save_data]).to_csv(DATA_PATH, mode='a', index=False, header=not os.path.exists(DATA_PATH))

    # Trigger Retrain at Midnight
    if datetime.now().hour == 0 and datetime.now().minute < 60: # CHANGED: Adjusted window
        train_model()

    print(f"\nSTATUS REPORT [{datetime.now().strftime('%H:%M')}]")
    print(f"Current HRI: {current_hri}")
    print(f"24-Hour Forecast Avg HRI: {avg_forecast_hri}")

if __name__ == "__main__":
    run_engine()
