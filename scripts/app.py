import json
from fastapi import FastAPI
from scripts.main_engine import run_engine
from scripts.utils import supabase, calculate_advice, get_metric
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "HRI Engine Running"}


@app.get("/run-engine")
def run():
    run_engine()
    return {"status": "Engine executed successfully"}


@app.get("/latest-hri/{city}")
def latest_hri(city: str):
    """
    Returns latest history row for a city plus a computed advice object.
    - hri    : actual score (0–500+ scale)
    - aqi    : real OWM index (1–5)
    - metric : label (Good / Moderate / Poor etc.)
    - advice : layered advice object (air, temp, uv, wind, precip, festival)
    """
    try:
        data = (
            supabase.table("history")
            .select("*")
            .eq("city", city)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )

        if not data.data:
            return {"error": "No data found for city"}

        row = data.data[0]

        # [UNCHANGED] Recalculate advice from stored fields — avoids storing it in DB
        pollutant_data = {k: row.get(k, 0) for k in ["pm2_5", "pm10", "no2", "o3", "co"]}
        weather_data   = {k: row.get(k, 0) for k in ["temp", "humidity", "wind_speed", "uv_index", "precip"]}
        advice = calculate_advice(pollutant_data, weather_data, row.get("hri", 0))

        return {**row, "advice": advice}

    except Exception as e:
        return {"error": str(e)}


@app.get("/forecast/{city}")
def get_forecast(city: str):
    """
    Returns next-day forecast rows for a city.
    [FIXED] No longer tries to parse advice strings — was crashing the endpoint.
    Advice is only served via /latest-hri which recalculates it cleanly.
    """
    try:
        data = (
            supabase.table("forecast")
            .select("timestamp, hri, metric, pm2_5, pm10, no2, o3, co, temp, humidity, wind_speed, uv_index, precip, city")
            # [FIXED] Explicitly select columns — excludes the 'advice' string column
            # that was causing ast.literal_eval crashes
            .eq("city", city)
            .order("timestamp", desc=True)
            .limit(24)
            .execute()
        )

        return list(reversed(data.data))

    except Exception as e:
        return {"error": str(e)}


# Lightweight advice-only endpoint
@app.get("/advice/{city}")
def get_advice(city: str):
    try:
        data = (
            supabase.table("history")
            .select("pm2_5, pm10, no2, o3, co, temp, humidity, wind_speed, uv_index, precip, hri")
            .eq("city", city)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )

        if not data.data:
            return {"error": "No data found for city"}

        row = data.data[0]
        pollutant_data = {k: row.get(k, 0) for k in ["pm2_5", "pm10", "no2", "o3", "co"]}
        weather_data   = {k: row.get(k, 0) for k in ["temp", "humidity", "wind_speed", "uv_index", "precip"]}
        advice = calculate_advice(pollutant_data, weather_data, row.get("hri", 0))

        return {"city": city, "advice": advice}

    except Exception as e:
        return {"error": str(e)}
