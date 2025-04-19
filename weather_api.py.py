import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from datetime import datetime, timedelta
import pytz
import random
from pathlib import Path
import os
from flask import Flask, request, jsonify
app = Flask(__Weather.Predication_)
app = FastAPI()

API_KEY = "39357f117056c535298fe0df516ce3e3"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

class CityInput(BaseModel):
    city: str
    historical_data_path: str = "weather.csv"

def get_current_weather(city):
    url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            raise Exception(f"Error: {data.get('message', 'Unknown error')}")
        return {
            'city': data.get('name', 'Unknown'),
            'current_temp': round(data['main'].get('temp', 0)),
            'feels_like': round(data['main'].get('feels_like', 0)),
            'temp_min': round(data['main'].get('temp_min', 0)),
            'temp_max': round(data['main'].get('temp_max', 0)),
            'humidity': round(data['main'].get('humidity', 0)),
            'description': data['weather'][0]['description'] if 'weather' in data else 'N/A',
            'country': data['sys'].get('country', 'N/A'),
            'WindGustSpeed': data['wind'].get('speed', 0),
            'WindGustDir': data['wind'].get('deg', 0),
            'Pressure': data['main'].get('pressure', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Weather API error: {str(e)}")

def load_data(filename):
    try:
        df = pd.read_csv(filename)
        df = df.dropna().drop_duplicates()
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading data: {str(e)}")

def prepare(data):
    le = LabelEncoder()
    if 'WindGustDir' in data.columns:
        data['WindGustDir'] = le.fit_transform(data['WindGustDir'].astype(str))
    if 'RainTomorrow' in data.columns:
        data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'].astype(str))
    required_columns = ['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure']
    available_columns = [col for col in required_columns if col in data.columns]
    X = data[available_columns]
    y = data['RainTomorrow'] if 'RainTomorrow' in data.columns else None
    return X, y, le

def get_compass_direction(degrees):
    compass_points = [
        ("N", 0, 22.5), ("NE", 22.5, 67.5), ("E", 67.5, 112.5), ("SE", 112.5, 157.5),
        ("S", 157.5, 202.5), ("SW", 202.5, 247.5), ("W", 247.5, 292.5), ("NW", 292.5, 337.5), ("N", 337.5, 360)
    ]
    return next((point for point, start, end in compass_points if start <= degrees < end), "Unknown")

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def train_or_load_model(model_type, X, y, model_path):
    if Path(model_path).exists():
        print(f"Loading saved {model_type} model...")
        return load_model(model_path)
    else:
        print(f"Training new {model_type} model...")
        if model_type == "classifier":
            model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        else:
            model = XGBRegressor(random_state=42)
        model.fit(X, y)
        save_model(model, model_path)
        return model

@app.post("/predict")
async def predict_weather(city_input: CityInput):
    try:
        current_weather = get_current_weather(city_input.city)
        
        historical_data = load_data(city_input.historical_data_path)
        
        X, y, le = prepare(historical_data)
        
        rain_model = train_or_load_model("classifier", X, y, "rain_model.pkl")
        
        wind_dir = get_compass_direction(current_weather['WindGustDir'] % 360)
        wind_dir_encoded = le.transform([wind_dir])[0] if wind_dir in le.classes_ else -1
        
        input_data = pd.DataFrame([{
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': wind_dir_encoded,
            'WindGustSpeed': current_weather['WindGustSpeed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['Pressure']
        }])
        
        rain_pred = rain_model.predict(input_data)[0]
        
        return {
            "city": current_weather['city'],
            "country": current_weather['country'],
            "current_temp": current_weather['current_temp'],
            "feels_like": current_weather['feels_like'],
            "humidity": current_weather['humidity'],
            "wind_speed": current_weather['WindGustSpeed'],
            "wind_direction": wind_dir,
            "rain_prediction": "Yes" if rain_pred == 1 else "No",
            "description": current_weather['description']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

if __Weather.Predication__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
