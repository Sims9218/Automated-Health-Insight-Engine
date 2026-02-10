import requests
import pandas as pd
from datetime import datetime
import os

API_KEY = os.getenv('AIR_QUALITY_API_KEY')

LAT = "19.0760" 
LON = "72.8777"
DATA_PATH = 'data/pollutant_data.csv'

def fetch_air_quality():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['list'][0]
        
        components = data['components']
        new_row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pm2_5': components['pm2_5'],
            'pm10': components['pm10'],
            'no2': components['no2'],
            'o3': components['o3'],
            'co': components['co']
        }
        
        df = pd.DataFrame([new_row])
        
        file_exists = os.path.isfile(DATA_PATH)
        df.to_csv(DATA_PATH, mode='a', index=False, header=not file_exists)
        
        print(f"Successfully fetched data at {new_row['timestamp']}")
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_air_quality()
