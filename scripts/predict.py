import pandas as pd
import joblib
import os
from datetime import datetime

LIMITS = {
    'pm2_5': 60,
    'pm10': 100,
    'no2': 80,
    'o3': 100,
    'co': 2000
}

def get_precautions(hri):
    if hri <= 1.0:
        return "LOW RISK: Air quality is safe. Perfect for outdoor exercise and activities."
    elif hri <= 2.0:
        return "MODERATE RISK: Sensitive individuals (asthma, heart conditions) should limit heavy outdoor exertion."
    elif hri <= 3.0:
        return "HIGH RISK: Wear an N95 mask if outdoors for long periods. Close windows during peak traffic hours."
    else:
        return "VERY HIGH RISK: Avoid all outdoor activities. Use air purifiers indoors and keep physical activity minimal."

def run_prediction():
    if not os.path.exists('models/model.pkl') or not os.path.exists('data/pollutant_data.csv'):
        print("Model or Data not found. Ensure fetch_data.py and train.py have run successfully.")
        return

    model = joblib.load('models/model.pkl')
    df = pd.read_csv('data/pollutant_data.csv')
    
    last_row = df.iloc[-1]
    pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'co']
    current_values = last_row[pollutants].values.reshape(1, -1)
    
    current_hour = datetime.now().hour
    hours_to_predict = 24 - current_hour
    
    predictions = []
    temp_input = current_values
    
    for _ in range(hours_to_predict):
        pred = model.predict(temp_input)
        predictions.append(pred[0])
        temp_input = pred
    
    forecast_df = pd.DataFrame(predictions, columns=pollutants)
    daily_avg = forecast_df.mean()
    
    hri_components = [daily_avg[p] / LIMITS[p] for p in pollutants]
    hri = sum(hri_components) / len(pollutants)
    
    print("PUBLIC HEALTH INSIGHT REPORT")
    print(f"Prediction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Calculated Daily HRI: {hri:.2f}")
    print(f"Health Precaution: {get_precautions(hri)}")

if __name__ == "__main__":
    run_prediction()
