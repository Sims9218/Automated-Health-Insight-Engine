import sqlite3
from utils import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # HISTORY TABLE
    cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        timestamp TEXT,
        pm2_5 REAL,
        pm10 REAL,
        no2 REAL,
        o3 REAL,
        co REAL,
        temp REAL,
        humidity REAL,
        wind_speed REAL,
        uv_index REAL,
        precip REAL,
        hri REAL,
        predicted_hri REAL,
        error_pct REAL,
        metric TEXT
    )
    """)

    # FORECAST TABLE
    cur.execute("""
    CREATE TABLE IF NOT EXISTS forecast (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        timestamp TEXT,
        pm2_5 REAL,
        pm10 REAL,
        no2 REAL,
        o3 REAL,
        co REAL,
        temp REAL,
        humidity REAL,
        wind_speed REAL,
        uv_index REAL,
        precip REAL,
        hri REAL,
        metric TEXT,
        model_version TEXT,
        created_at TEXT
    )
    """)

    # MODEL REGISTRY
    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT,
        model_path TEXT,
        trained_at TEXT,
        data_points INTEGER,
        mae REAL,
        promoted INTEGER
    )
    """)

    conn.commit()
    conn.close()
    print("Database initialized.")

if __name__ == "__main__":
    init_db()