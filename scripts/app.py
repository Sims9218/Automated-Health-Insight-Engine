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
    Returns the latest history row for a city, with:
    - hri        : actual HRI score (0–500+ scale, NOT divided by 10)
    - aqi        : real OWM AQI index (1–5 scale)
    - metric     : HRI label (Good / Moderate / Poor etc.)
    - advice     : layered advice object (air, temp, uv, wind, precip, festival)
    - all pollutants and weather fields
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

        # [ADDED] Build advice from the stored pollutant + weather fields
        # Recalculated here so we don't need to store the full advice object in DB
        pollutant_data = {
            "pm2_5": row.get("pm2_5", 0),
            "pm10":  row.get("pm10", 0),
            "no2":   row.get("no2", 0),
            "o3":    row.get("o3", 0),
            "co":    row.get("co", 0),
        }
        weather_data = {
            "temp":       row.get("temp", 25),
            "humidity":   row.get("humidity", 50),
            "wind_speed": row.get("wind_speed", 3),
            "uv_index":   row.get("uv_index", 0),
            "precip":     row.get("precip", 0),
        }
        advice = calculate_advice(
            pollutant_data,
            weather_data,
            row.get("hri", 0)
        )

        # [ADDED] Return the full row plus the advice object
        # 'aqi' is now the real OWM 1–5 index saved by main_engine
        return {
            **row,
            "advice": advice,
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/forecast/{city}")
def get_forecast(city: str):
    """
    Returns next-day forecast rows for a city.
    Each row now includes an 'advice' field (parsed from stored string).
    """
    try:
        data = (
            supabase.table("forecast")
            .select("*")
            .eq("city", city)
            .order("timestamp", desc=True)
            .limit(24)
            .execute()
        )

        rows = list(reversed(data.data))

        # [ADDED] Parse advice string back to dict if it was stored as a string
        # This happens because Supabase doesn't have a native dict column type
        for row in rows:
            if "advice" in row and isinstance(row["advice"], str):
                try:
                    # ast.literal_eval is safer than eval for Python dict strings
                    import ast
                    row["advice"] = ast.literal_eval(row["advice"])
                except Exception:
                    row["advice"] = None

        return rows

    except Exception as e:
        return {"error": str(e)}


# [ADDED] Dedicated advice endpoint — useful if frontend wants advice independently
@app.get("/advice/{city}")
def get_advice(city: str):
    """
    Returns only the advice object for the latest reading of a city.
    Lightweight alternative to /latest-hri if only advice is needed.
    """
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
        weather_data = {k: row.get(k, 0) for k in ["temp", "humidity", "wind_speed", "uv_index", "precip"]}

        advice = calculate_advice(pollutant_data, weather_data, row.get("hri", 0))
        return {"city": city, "advice": advice}

    except Exception as e:
        return {"error": str(e)}
