import os

# --- CONFIGURATION ---
LAT, LON = "19.07", "72.87"
LIMITS = {'pm2_5': 25, 'pm10': 50, 'no2': 40, 'o3': 100, 'co': 10}
WEIGHTS = {'pm2_5': 0.4, 'pm10': 0.2, 'no2': 0.15, 'o3': 0.15, 'co': 0.1}

API_KEY = os.getenv("API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'comprehensive_history.csv')
FORECAST_PATH = os.path.join(BASE_DIR, 'data', 'forecast_timeline.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'specialist_model.pkl')

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
