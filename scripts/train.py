import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/pollutant_data.csv'
MODEL_PATH = 'models/model.pkl'
ERROR_THRESHOLD = 0.15

def load_data():
    if not os.path.exists(DATA_PATH):
        print("No data found to train on.")
        return None
    df = pd.read_csv(DATA_PATH)
  
    if len(df) < 24:
        print("Not enough data points to train (need at least 24).")
        return None
    return df

def prepare_features(df):
    """
    Simple feature engineering: Use previous hour's values to predict current.
    Targets: PM2.5, PM10, NO2, O3, CO
    """
    pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
    
    for p in pollutants:
        df[f'{p}_lag1'] = df[p].shift(1)
    
    df = df.dropna()
    
    X = df[[f'{p}_lag1' for p in pollutants]]
    y = df[pollutants]
    
    return X, y

def train_and_save():
    df = load_data()
    if df is None: return

    X, y = prepare_features(df)
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        
        relative_error = mae / y.mean().mean()
        
        print(f"Current Model Relative Error: {relative_error:.2%}")
        
        if relative_error <= ERROR_THRESHOLD:
            print("Model accuracy is still within limits. Skipping retrain.")
            return
        else:
            print("Error threshold exceeded. Retraining model...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    new_model = RandomForestRegressor(n_estimators=100, random_state=42)
    new_model.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(new_model, MODEL_PATH)
    print("New model saved successfully to models/model.pkl")

if __name__ == "__main__":
    train_and_save()
