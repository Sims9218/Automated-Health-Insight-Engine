import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from utils import DATA_PATH, MODEL_PATH

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

    # We train ONLY on real data features
    feature_cols = [f'{p}_lag' for p in pollutants] + weather_cols + ['hour']
    X = df[feature_cols]
    y = df[pollutants]

    # CHRONOLOGICAL Split (No Shuffling for Time Series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    # MultiOutput Specialist Regressor
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model Retrained at {datetime.now()}. Data points: {len(df)}")

if __name__ == "__main__":
    train_model()
