import os
import pandas as pd
import joblib
from utils import DATA_PATH, MODEL_PATH, now_ist
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# --- MODEL TRAINER ---
def train_model():
    if not os.path.exists(DATA_PATH):
        print("No history data found, skipping training.")
        return

    df = pd.read_csv(DATA_PATH)

    # BUG FIX: History had mixed timestamp formats (DD-MM-YYYY and YYYY-MM-DD).
    # Normalise all to YYYY-MM-DD HH:MM before parsing.
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, format='mixed')

    if len(df) < 48:
        print(f"Not enough data to train ({len(df)} rows, need 48). Skipping.")
        return

    pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
    # BUG FIX: Explicit, ordered weather columns to guarantee feature order
    # consistency between training and prediction.
    weather_cols = ['temp', 'humidity', 'wind_speed', 'uv_index', 'precip']

    for p in pollutants:
        df[f'{p}_lag'] = df[p].shift(1)

    df['hour'] = df['timestamp'].dt.hour
    df = df.dropna()

    feature_cols = [f'{p}_lag' for p in pollutants] + weather_cols + ['hour']
    X = df[feature_cols]
    y = df[pollutants]

    # Chronological split — no shuffling for time series
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False
    )

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    )
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model retrained at {now_ist().strftime('%Y-%m-%d %H:%M IST')}. Data points: {len(df)}")

if __name__ == "__main__":
    train_model()
