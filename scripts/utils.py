import os
from datetime import datetime, timezone, timedelta

# Coordinates
LAT, LON = "19.07", "72.87"

# CO limit was 10 (mg/m³) but OWM returns µg/m³.
# WHO 8-hour limit is 10 mg/m³ = 10,000 µg/m³.
# pm2_5, pm10, no2, o3 are already in µg/m³ and their limits are correct.
LIMITS = {'pm2_5': 25, 'pm10': 50, 'no2': 40, 'o3': 100, 'co': 10000}
WEIGHTS = {'pm2_5': 0.4, 'pm10': 0.2, 'no2': 0.15, 'o3': 0.15, 'co': 0.1}

API_KEY = os.getenv("API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'comprehensive_history.csv')
FORECAST_PATH = os.path.join(BASE_DIR, 'data', 'forecast_timeline.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'versions')
LATEST_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'specialist_model.pkl')
REGISTRY_PATH = os.path.join(BASE_DIR, 'models', 'model_registry.csv')

#UTC TO IST
IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    """Returns current datetime in IST."""
    return datetime.now(IST)

def ts_to_ist(unix_timestamp):
    """Converts a Unix timestamp to an IST-aware datetime."""
    return datetime.fromtimestamp(unix_timestamp, tz=IST)


# --- HRI CALCULATION ---
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

    # Wind Adjustments (Dispersion and accumulation)
    if w < 2:
        for p in ['pm2_5', 'pm10', 'no2', 'co']:
            adjusted[p] *= 1.10
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

    # Final Weighted Sum — result is 0-100+ scale
    hri_score = sum((adjusted.get(k, 0) / LIMITS[k]) * WEIGHTS[k] for k in WEIGHTS)
    return round(hri_score * 100, 2)


# --- HRI CATEGORY + SUGGESTIONS ---
# Average day ~270, bad days ~500+.
HRI_LEVELS = [
    (75,  "Good",      "Air quality is healthy. No precautions needed."),
    (150, "Moderate",  "Acceptable air quality. Sensitive individuals should limit prolonged outdoor exposure."),
    (250, "Poor",      "Wear an N95 mask outdoors. Reduce prolonged exertion."),
    (350, "Unhealthy", "Wear an N95 mask. Avoid heavy outdoor exercise like running or cycling."),
    (500, "Severe",    "Minimise all outdoor activity. Keep windows closed."),
    (float('inf'), "Hazardous", "Avoid all outdoor activity. Stay indoors with air purifiers running."),
]

def get_metric(hri):
    """Returns the HRI category label for a given HRI score."""
    for threshold, label, _ in HRI_LEVELS:
        if hri < threshold:
            return label
    return "Hazardous"
