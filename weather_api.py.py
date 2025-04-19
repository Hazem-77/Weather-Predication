import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from datetime import datetime
import pytz
import random
from pathlib import Path
import os
from typing import Optional

app = FastAPI(
    title="Weather Prediction API",
    description="API for predicting weather conditions using XGBoost models",
    version="1.0.0"
)

# Configuration
API_KEY = os.getenv("API_KEY", "39357f117056c535298fe0df516ce3e3")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class CityInput(BaseModel):
    city: str
    historical_data_path: Optional[str] = "weather.csv"

class WeatherResponse(BaseModel):
    city: str
    country: str
    current_temp: float
    feels_like: float
    humidity: int
    wind_speed: float
    wind_direction: str
    rain_prediction: str
    description: str
    prediction_confidence: Optional[float]

@app.get("/", tags=["Root"])
async def root():
    """Health check endpoint"""
    return {"status": "API is running"}

def get_current_weather(city: str) -> dict:
    """Fetch current weather data from OpenWeatherMap API"""
    url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            'city': data.get('name', city),
            'current_temp': round(data['main'].get('temp', 0), 1),
            'feels_like': round(data['main'].get('feels_like', 0), 1),
            'temp_min': round(data['main'].get('temp_min', 0), 1),
            'temp_max': round(data['main'].get('temp_max', 0), 1),
            'humidity': data['main'].get('humidity', 0),
            'description': data['weather'][0]['description'] if data.get('weather') else 'N/A',
            'country': data['sys'].get('country', 'N/A'),
            'wind_speed': data['wind'].get('speed', 0),
            'wind_deg': data['wind'].get('deg', 0),
            'pressure': data['main'].get('pressure', 0)
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather API request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing weather data: {str(e)}")

def load_data(filename: str) -> pd.DataFrame:
    """Load and preprocess historical weather data"""
    try:
        if not Path(filename).exists():
            raise FileNotFoundError(f"Data file {filename} not found")
            
        df = pd.read_csv(filename)
        return df.dropna().drop_duplicates()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")

def prepare_features(df: pd.DataFrame, le: LabelEncoder = None) -> tuple:
    """Prepare features for model prediction"""
    if le is None:
        le = LabelEncoder()
    
    df = df.copy()
    if 'WindGustDir' in df.columns:
        df['WindGustDir'] = le.fit_transform(df['WindGustDir'].astype(str))
    
    required_cols = ['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure']
    available_cols = [col for col in required_cols if col in df.columns]
    
    return df[available_cols], le

def get_compass_direction(degrees: float) -> str:
    """Convert wind degrees to compass direction"""
    compass_points = [
        ("N", 0, 22.5), ("NE", 22.5, 67.5), ("E", 67.5, 112.5), 
        ("SE", 112.5, 157.5), ("S", 157.5, 202.5), ("SW", 202.5, 247.5), 
        ("W", 247.5, 292.5), ("NW", 292.5, 337.5), ("N", 337.5, 360)
    ]
    degrees = degrees % 360
    return next((point for point, start, end in compass_points if start <= degrees < end), "Unknown")

def save_model(model, model_type: str):
    """Save trained model to disk"""
    model_path = Path(MODEL_DIR) / f"{model_type}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model_path

def load_model(model_type: str):
    """Load trained model from disk"""
    model_path = Path(MODEL_DIR) / f"{model_type}_model.pkl"
    if not model_path.exists():
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_model(model_type: str, X: pd.DataFrame = None, y: pd.Series = None):
    """Get or train model"""
    model = load_model(model_type)
    
    if model is None and X is not None and y is not None:
        if model_type == "rain":
            model = XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            model = XGBRegressor(random_state=42)
        
        model.fit(X, y)
        save_model(model, model_type)
    
    return model

@app.post("/predict", response_model=WeatherResponse, tags=["Prediction"])
async def predict_weather(city_input: CityInput):
    """
    Predict weather conditions for a given city
    
    - **city**: City name to get prediction for
    - **historical_data_path**: Path to historical data CSV (default: weather.csv)
    """
    try:
        # 1. Get current weather data
        current_weather = get_current_weather(city_input.city)
        
        # 2. Load and prepare historical data
        historical_data = load_data(city_input.historical_data_path)
        X, le = prepare_features(historical_data)
        y = historical_data['RainTomorrow'] if 'RainTomorrow' in historical_data.columns else None
        
        # 3. Get or train model
        rain_model = get_model("rain", X, y)
        if rain_model is None:
            raise HTTPException(status_code=500, detail="Model not available and no training data provided")
        
        # 4. Prepare input features
        wind_dir = get_compass_direction(current_weather['wind_deg'])
        wind_dir_encoded = le.transform([wind_dir])[0] if wind_dir in le.classes_ else 0
        
        input_data = pd.DataFrame([{
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': wind_dir_encoded,
            'WindGustSpeed': current_weather['wind_speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure']
        }])
        
        # 5. Make prediction
        rain_pred = rain_model.predict(input_data)[0]
        rain_prob = rain_model.predict_proba(input_data)[0][1] if hasattr(rain_model, 'predict_proba') else None
        
        return {
            "city": current_weather['city'],
            "country": current_weather['country'],
            "current_temp": current_weather['current_temp'],
            "feels_like": current_weather['feels_like'],
            "humidity": current_weather['humidity'],
            "wind_speed": current_weather['wind_speed'],
            "wind_direction": wind_dir,
            "rain_prediction": "Yes" if rain_pred == 1 else "No",
            "description": current_weather['description'],
            "prediction_confidence": round(rain_prob, 2) if rain_prob is not None else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "weather_api:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        reload_dirs=["."],
        log_level="info"
    )
