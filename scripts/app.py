from fastapi import FastAPI
from scripts.main_engine import run_engine
from utils import supabase

app = FastAPI()


@app.get("/")
def home():
    return {"message": "HRI Engine Running 🚀"}


@app.get("/run-engine")
def run():
    run_engine()
    return {"status": "Engine executed successfully"}


@app.get("/latest-hri")
def latest_hri():
    try:
        data = supabase.table("history") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(10) \
            .execute()
        return data.data
    except Exception as e:
        return {"error": str(e)}