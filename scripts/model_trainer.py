import os
import pandas as pd
import joblib
from utils import (DATA_PATH,MODEL_DIR,LATEST_MODEL_PATH,REGISTRY_PATH,now_ist)
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sqlite3
from utils import DB_PATH
from utils import supabase

# --- MODEL TRAINER ---
def train_model():
    # Fetch training data from Supabase
    response = supabase.table("history").select("*").order("timestamp").execute()
    df = pd.DataFrame(response.data)

    if df.empty:
        print("No data in Supabase, skipping training.")
        return
    
    
    # Normalise all to YYYY-MM-DD HH:MM before parsing.
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, format='mixed')

    if len(df) < 48:
        print(f"Not enough data to train ({len(df)} rows, need 48). Skipping.")
        return

    pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
    # consistency between training and prediction.
    weather_cols = ['temp', 'humidity', 'wind_speed', 'uv_index', 'precip']
    
    df = df.sort_values(["city", "timestamp"])
    for p in pollutants:
        df[f'{p}_lag'] = df.groupby("city")[p].shift(1)
        

    df['hour'] = df['timestamp'].dt.hour
    df = df.dropna()

    feature_cols = [f'{p}_lag' for p in pollutants] + weather_cols + ['hour']
    X = df[feature_cols]
    y = df[pollutants]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False
    )

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    )
    model.fit(X_train, y_train)
    
    # Evaluate NEW model
    new_preds = model.predict(X_test)
    new_score = mean_absolute_error(y_test, new_preds)

    # Create folders
    os.makedirs(os.path.dirname(LATEST_MODEL_PATH), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Create unique model version name
    model_version = now_ist().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(
        MODEL_DIR,
        f"model_{model_version}.pkl"
    )

    # Save versioned model
    joblib.dump(model, version_path)
    
    # Compare with existing (production) model
    if os.path.exists(LATEST_MODEL_PATH):
        old_model = joblib.load(LATEST_MODEL_PATH)
        old_preds = old_model.predict(X_test)
        old_score = mean_absolute_error(y_test, old_preds)
    else:
        old_score = float("inf")

    # Promote only if better
    if new_score < old_score:
        print(f"New model improved MAE: {old_score:.4f} → {new_score:.4f}")
        joblib.dump(model, LATEST_MODEL_PATH)
    else:
        print(f"Model NOT promoted. Old MAE: {old_score:.4f} | New MAE: {new_score:.4f}")

    print(
        f"Model retrained at {now_ist().strftime('%Y-%m-%d %H:%M IST')}. "
        f"Data points: {len(df)}"
    )

    # Save registry entry
    registry_row = {
        "version": model_version,
        "model_path": version_path,
        "trained_at": now_ist().strftime('%Y-%m-%d %H:%M'),
        "data_points": len(df),
        "mae": new_score,
        "promoted":new_score<old_score
    }

    pd.DataFrame([registry_row]).to_csv(
        REGISTRY_PATH,
        mode="a",
        index=False,
        header=not os.path.exists(REGISTRY_PATH)
    )
if __name__ == "__main__":
    train_model()
