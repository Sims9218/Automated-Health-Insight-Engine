# Automated-Public-Health-Insight-Engine
An end-to-end MLOps pipeline that fetches real-time air quality data, calculates a custom Health Risk Index (HRI), and uses a self-correcting Machine Learning model to forecast health risks and suggest precautions.

Key Features
1. Real-time Data Ingestion: Automatically fetches PM2.5, PM10, NO_2, O_3, and CO hourly via OpenWeatherMap API.
2. Self-Correcting ML Model: Uses XGBoost for time-series forecasting. The model evaluates its own error and automatically retrains if accuracy drops below a specific threshold.
3. Automated Workflow: Powered entirely by GitHub Actions—no dedicated server required.
4. Live Dashboard: A Streamlit interface to visualize HRI trends and performance logs.
5. Health Notifications: Real-time alerts sent via Telegram when hazardous air quality is detected.

Tech Stack
Language: Python 3.9+
ML Framework: #edit it
Automation: GitHub Actions (Cron scheduling)
Data Handling: Pandas, Joblib
