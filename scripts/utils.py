import os
from datetime import datetime, timezone, timedelta
from supabase import create_client
# Coordinates
# CO limit was 10 (mg/m³) but OWM returns µg/m³.
# WHO 8-hour limit is 10 mg/m³ = 10,000 µg/m³.
# pm2_5, pm10, no2, o3 are already in µg/m³ and their limits are correct.
LIMITS = {'pm2_5': 25, 'pm10': 50, 'no2': 40, 'o3': 100, 'co': 10000}
WEIGHTS = {'pm2_5': 0.4, 'pm10': 0.2, 'no2': 0.15, 'o3': 0.15, 'co': 0.1}

API_KEY = os.getenv("API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'versions')
LATEST_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'specialist_model.pkl')
MODEL_PATH=LATEST_MODEL_PATH
REGISTRY_PATH = os.path.join(BASE_DIR, 'models', 'model_registry.csv')

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase connection failed: {e}")

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




HRI_LEVELS = [
    (60,  "Good",      "Great day to be outside! The air is clean and safe for everyone."),
    (120, "Moderate",  "Generally fine, but sensitive groups should take short breaks during outdoor activities."),
    (180, "Poor",      "Air is becoming hazy. Sensitive individuals should avoid prolonged outdoor exposure. Consider a mask if near traffic."),
    (240, "Unhealthy", "Pollution is high. Wear an N95 mask for outdoor commutes. Avoid heavy cardio (running/sports) outside."),
    (300, "Severe",    "Heavy smog detected. Stay indoors as much as possible. If you must go out, use a high-quality N95 mask."),
    (float('inf'), "Hazardous", "Critical air levels. Keep all windows closed, use air purifiers, and strictly avoid outdoor exposure."),
]




def get_metric(hri):
    """Returns the HRI category label for a given HRI score."""
    for threshold, label, _ in HRI_LEVELS:
        if hri < threshold:
            return label
    return "Hazardous"

# [ADDED] --- FESTIVAL CALENDAR ---
# Maps month-day to festival info. Extend this list as needed.
# Each entry: (month, day, name, advice)
FESTIVAL_CALENDAR = [
    (1, 14,  "Makar Sankranti", "Kite flying day — watch for kite string hazards outdoors. Avoid rooftops if windy."),
    (1, 26,  "Republic Day",    "Heavy traffic and parade crowds expected. Avoid central areas if possible."),
    (3, 25,  "Holi",           "Air may carry colour powder — wear a mask outdoors. Avoid eye contact with colours."),
    (10, 2,  "Gandhi Jayanti", "Public holiday — reduced traffic, lighter pollution expected."),
    (10, 24, "Dussehra",       "Effigy burning causes short-term smoke spikes. Stay indoors during evening events."),
    (11, 1,  "Diwali",         "Fireworks significantly raise PM2.5. Wear N95 mask, keep windows shut, run air purifier."),
    (11, 2,  "Diwali",         "Post-Diwali air may still be heavily polluted. Limit outdoor exposure."),
    (11, 15, "Guru Nanak",     "Processions may cause traffic. Plan routes in advance."),
    (12, 25, "Christmas",      "Relatively calm. Light traffic and lower pollution expected."),
    (8, 15,  "Independence Day","Celebrations and traffic. Fireworks in evening — sensitive groups stay indoors."),
    (8, 26,  "Ganesh Chaturthi","Processions and crowd gatherings. Noise and traffic will be high."),
    (9, 5,   "Ganesh Visarjan","Immersion processions — heavy crowds and noise. Coastal areas crowded."),
]

def get_festival_advice(dt=None):
    """
    [ADDED] Returns festival advice if today matches a festival, else None.
    dt: a datetime object (defaults to today IST).
    Returns dict with 'name' and 'advice', or None.
    """
    if dt is None:
        dt = now_ist()
    for month, day, name, advice in FESTIVAL_CALENDAR:
        if dt.month == month and dt.day == day:
            return {"name": name, "advice": advice}
    return None


# [ADDED] --- LAYERED ADVICE ENGINE ---
def calculate_advice(aqi_data, weather_data, hri_score, dt=None):
    """
    [ADDED] Generates layered, context-aware advice based on:
    - AQI / pollutant levels (air quality layer)
    - Temperature (heat/hydration layer)
    - UV index (sun protection layer)
    - Wind speed (pollution dispersion/trapping layer)
    - Precipitation (rain layer)
    - Festival calendar (event-based layer)

    Returns a dict of advice layers. Each layer is always present;
    irrelevant ones return None so the frontend can choose to hide them.

    Args:
        aqi_data    : dict of pollutant readings (pm2_5, pm10, no2, o3, co)
        weather_data: dict with temp, humidity, wind_speed, uv_index, precip
        hri_score   : computed HRI float (0–500+)
        dt          : datetime for festival lookup (defaults to now IST)
    """
    advice = {}

    # --- Layer 1: Air Quality ---
    # Based on the new 60-point interval thresholds
    if hri_score < 60:
        advice["air"] = {
            "label": "Good",
            "text": "Great day to be outside! The air is clean and safe for everyone.",
            "mask": False
        }
    elif hri_score < 120:
        advice["air"] = {
            "label": "Moderate",
            "text": "Generally fine, but sensitive groups should take short breaks during outdoor activities.",
            "mask": False
        }
    elif hri_score < 180:
        advice["air"] = {
            "label": "Poor",
            "text": "Air is becoming hazy. Sensitive individuals should avoid prolonged outdoor exposure. Consider a mask if near traffic.",
            "mask": "Optional"
        }
    elif hri_score < 240:
        advice["air"] = {
            "label": "Unhealthy",
            "text": "Pollution is high. Wear an N95 mask for outdoor commutes. Avoid heavy cardio (running/sports) outside.",
            "mask": True
        }
    elif hri_score < 300:
        advice["air"] = {
            "label": "Severe",
            "text": "Heavy smog detected. Stay indoors as much as possible. If you must go out, use a high-quality N95 mask.",
            "mask": True
        }
    else:
        advice["air"] = {
            "label": "Hazardous",
            "text": "Critical air levels. Keep all windows closed, use air purifiers, and strictly avoid outdoor exposure.",
            "mask": True
        }

    # --- Layer 2: Temperature ---
    temp = weather_data.get('temp', 25)
    if temp >= 40:
        advice["temp"] = "Extreme heat   stay hydrated, avoid outdoor activity between 11AM–4PM, wear light clothing."
    elif temp >= 35:
        advice["temp"] = "Very hot   drink water frequently, wear sunscreen and light clothing. Limit outdoor exertion."
    elif temp >= 30:
        advice["temp"] = "Warm day   stay hydrated. Take breaks if exercising outdoors."
    elif temp <= 10:
        advice["temp"] = "Cold weather   dress in layers, keep extremities warm. Elderly and children take extra care."
    elif temp <= 15:
        advice["temp"] = "Cool weather   carry a light jacket, especially in the evening."
    else:
        # [ADDED] Return None for comfortable temps — frontend hides this layer
        advice["temp"] = None

    # --- Layer 3: UV Index ---
    uv = weather_data.get('uv_index', 0)
    if uv >= 11:
        advice["uv"] = "Extreme UV   stay indoors midday. Wear SPF 50+, hat, UV-blocking sunglasses and full-sleeve clothing."
    elif uv >= 8:
        advice["uv"] = "Very high UV   apply SPF 30+ every 2 hours, wear a hat and sunglasses. Avoid midday sun."
    elif uv >= 6:
        advice["uv"] = "High UV   wear sunscreen and a hat if going out between 10AM–3PM."
    elif uv >= 3:
        advice["uv"] = "Moderate UV   sunscreen recommended for prolonged outdoor exposure."
    else:
        advice["uv"] = None  # low UV at night or overcast, no advice needed

    # --- Layer 4: Wind Speed ---
    wind = weather_data.get('wind_speed', 3)
    if wind < 2:
        advice["wind"] = "Very low wind   pollutants are trapped near ground level. Avoid busy main roads and high-traffic areas."
    elif wind > 10:
        advice["wind"] = "Strong winds   good for dispersing pollution but avoid exposed areas. Secure loose items outdoors."
    elif wind > 6:
        advice["wind"] = "Moderate-high winds   air dispersion is good. Outdoor conditions generally fine."
    else:
        advice["wind"] = None  # normal wind, no specific advice

    # --- Layer 5: Precipitation ---
    precip = weather_data.get('precip', 0)
    humidity = weather_data.get('humidity', 50)
    if precip > 5:
        advice["precip"] = "Heavy rain   carry an umbrella. Roads may be flooded. Avoid waterlogged areas."
    elif precip > 0.5:
        advice["precip"] = "Light rain expected   carry an umbrella. Air quality will improve as rain washes out pollutants."
    elif humidity > 85:
        advice["precip"] = "High humidity   feels muggy. Rain possible. Keep an umbrella handy."
    else:
        advice["precip"] = None  # dry conditions, no advice needed

    # --- Layer 6: Festival ---
    # [ADDED] Checks today's date against festival calendar
    advice["festival"] = get_festival_advice(dt)

    return advice 
