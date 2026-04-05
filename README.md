# Automated Public Health Insight Engine

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![MLOps](https://img.shields.io/badge/MLOps-Pipeline-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Status](https://img.shields.io/badge/Status-Active-success)

An end-to-end **MLOps pipeline** for real-time air quality monitoring, health risk analysis, and pollutant forecasting.

---

## Overview

This project implements a fully automated system that:

* Collects **real-time air quality + weather data**
* Computes a custom **Health Risk Index (HRI)**
* Predicts **future pollutant levels**
* Continuously **monitors and retrains** the model
* Serves results via **FastAPI endpoints**

---

## MLOps Pipeline

```text
API Data → Processing → HRI → ML Model → Storage → Monitoring → Retraining
```

* Runs **every hour** (GitHub Actions)
* Retrains **daily at midnight**
* Promotes model **only if performance improves**

---

## Key Features

*  Real-time data ingestion (OpenWeather API)
*  Custom **Health Risk Index (HRI)**
*  Multi-output ML model (RandomForest)
*  Automated pipeline (GitHub Actions)
*  Model monitoring (error %)
*  Daily retraining + model versioning
*  FastAPI backend
*  Supabase database
*  Docker deployment

---

## System Architecture

```text
Frontend (React)
        ↓
FastAPI Backend
        ↓
Supabase (History + Forecast)
        ↓
ML Pipeline (run_engine)
        ↓
Model Training (train_model)
        ↓
GitHub Actions (Automation)
```

---

## Machine Learning

* **Model:** RandomForest (MultiOutputRegressor)
* **Inputs:**

  * Lag pollutant values
  * Weather data (temp, humidity, wind, UV, precipitation)
  * Hour of day
* **Output:**

  * Predicted pollutant levels → converted to HRI
* **Metric:** Mean Absolute Error (MAE)

---

## Health Risk Index (HRI)

### Formula:

```text
HRI = Σ (Pollutant / Limit × Weight) × 100
```

### Adjustments:

* Humidity
* Wind speed
* Precipitation
* UV index

### Categories:

🟢 Good | 🟡 Moderate | 🟠 Poor | 🔴 Unhealthy | 🟣 Severe | ⚫ Hazardous

---

## Model Monitoring & Retraining

* Compare **actual vs predicted HRI**
* Compute **error %**
* Retrain model **daily**
* Promote model **only if performance improves**

---

## API Endpoints

| Endpoint             | Description      |
| -------------------- | ---------------- |
| `/`                  | Health check     |
| `/run-engine`        | Trigger pipeline |
| `/latest-hri/{city}` | Latest HRI       |
| `/forecast/{city}`   | 24-hour forecast |

---

## Database (Supabase)

* **cities** → city, lat, lon
* **history** → AQI, HRI, predictions, error
* **forecast** → future predictions

---

## Docker

```bash
docker build -t hri-engine .
docker run -p 8000:8000 hri-engine
```

---

## ⚙️ GitHub Actions Automation

* Runs every hour
* Executes full pipeline
* Saves updated models
* Pushes changes automatically

---

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```env
API_KEY=your_openweather_api_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
```

## 🧪 Run Frontend

```bash
cd frontend
npm install
npm start
```

### 3. Run backend

```bash
uvicorn scripts.app:app --reload
```

### 4. Trigger pipeline

```
http://localhost:8000/run-engine
```

---

## Output

* Real-time HRI
* Forecasted HRI
* Pollutant trends
* Health categories

---

## Future Scope

* Mobile application
* Multi-city scaling
* Advanced drift detection
* Improved prediction accuracy

---

## Authors

-**Vijval Nair**  
-**Rajvardhan Nalawade**  
-**Akshit Singh**  
-**Simarjit Banka**



---

##  Project Highlights

* End-to-end MLOps pipeline
* Automated retraining + model promotion
* Real-time + predictive system
* Fully containerized and deployable

---
