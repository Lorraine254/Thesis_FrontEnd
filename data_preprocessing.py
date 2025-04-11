import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import load
import os
import requests
import math
import random
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from dotenv import load_dotenv
load_dotenv()

import lime
import lime.lime_tabular
from lime import lime_tabular

NAIROBI_BOUNDS = {
    'min_lat': -1.47,
    'max_lat': -1.15,
    'min_lon': 36.65,
    'max_lon': 37.05
}

@st.cache_data
def read_data():
    historical_data = pd.read_csv(
        'https://raw.githubusercontent.com/Lorraine254/Data/refs/heads/main/nairobi_pm25_weather_historical.csv',
        parse_dates=['Unnamed: 0']
    )
    historical_data.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp']).dt.tz_localize(None)
    historical_data['hour'] = historical_data['timestamp'].dt.hour
    historical_data['year'] = historical_data['timestamp'].dt.year
    historical_data['week'] = historical_data['timestamp'].dt.isocalendar().week
    return historical_data

@st.cache_data
def create_lag_features(df, lag=5):
    for i in range(1, lag+1):
        df[f'lag_{i}'] = df['pm2.5'].shift(i)
    df.dropna(inplace=True)
    return df

@st.cache_data
def train_test_split_data(df):
    df_lagged = create_lag_features(df, lag=5)
    # Only keep the features that were used in training
    required_features = [
        'lag_1','lag_2','hour','year','lag_5','lag_3','lag_4',
        'dew_point','wind_speed','wind_deg','pressure','week',
        'temperature','humidity','temp_max'
    ]
    X = df_lagged[required_features]
    y = df_lagged['pm2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@st.cache_resource
@st.cache_resource
@st.cache_resource
def load_models():
    """Load all LightGBM models (main + quantile)"""
    try:
        models = load('lightgbm_models.joblib')
        if not isinstance(models, dict):
            st.error("Invalid model format - expected dictionary")
            return None
        
        main_model = models.get('main_model')
        upper_model = models.get('upper_model')
        lower_model = models.get('lower_model')

        if main_model is None:
            st.error("Main model not found in the model file")
            return None
        
        return {
            'main': models['main_model'],
            'upper': upper_model,
            'lower': lower_model
        }
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None
    
def fetch_weather(lat, lon, target_time):
    """Fetch weather data with robust error handling"""
    # Fixed Nairobi coordinates
    NAIROBI_COORDS = (-1.286389, 36.817223)
    
    # Always use Nairobi coordinates regardless of input
    lat, lon = NAIROBI_COORDS
    
    try:
        # For current weather (remove time parameter)
        if target_time.date() == datetime.now().date():
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    'lat': lat,
                    'lon': lon,
                    'appid': os.getenv('OPENWEATHER_API_KEY'),
                    'units': 'metric'
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                'dew_point': data.get('main', {}).get('dew_point', 15),
                'wind_speed': data.get('wind', {}).get('speed', 3),
                'wind_deg': data.get('wind', {}).get('deg', 0),
                'pressure': data.get('main', {}).get('pressure', 1013),
                'humidity': data.get('main', {}).get('humidity', 50),
                'temperature': data.get('main', {}).get('temp', 20),
                'temp_max': data.get('main', {}).get('temp_max', 23),
                'data_source': 'OpenWeatherMap Current'
            }
        # For forecast data (up to 5 days)
        else:
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={
                    'lat': lat,
                    'lon': lon,
                    'appid': os.getenv('OPENWEATHER_API_KEY'),
                    'units': 'metric',
                    'cnt': 40  # Number of timestamps (5 days in 3-hour intervals)
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Find the closest forecast time to our target
            closest = min(data['list'], key=lambda x: abs(datetime.fromtimestamp(x['dt']) - target_time))
            
            return {
                'dew_point': closest.get('main', {}).get('dew_point', 15),
                'wind_speed': closest.get('wind', {}).get('speed', 3),
                'wind_deg': closest.get('wind', {}).get('deg', 0),
                'pressure': closest.get('main', {}).get('pressure', 1013),
                'humidity': closest.get('main', {}).get('humidity', 50),
                'temperature': closest.get('main', {}).get('temp', 20),
                'temp_max': closest.get('main', {}).get('temp_max', 23),
                'data_source': 'OpenWeatherMap Forecast'
            }
            
    except Exception as api_error:
        error_msg = str(api_error)
        if hasattr(api_error, 'response'):
            error_msg += f" (Status: {api_error.response.status_code})"
        
        st.warning(f"⚠ Using simulated Nairobi data (API error: {error_msg[:150]})")
        
        # Generate realistic Nairobi-like simulated data
        hour = target_time.hour
        return {
            'dew_point': 12.0 + 4*math.sin(hour/24*2*math.pi),
            'wind_speed': 2.5 + random.uniform(-0.5, 0.5),
            'wind_deg': random.randint(90, 270),
            'pressure': 1015 + random.randint(-5, 5),
            'humidity': 60 + int(15*math.sin(hour/12*math.pi)),
            'temperature': 18.0 + 6*math.sin(hour/24*2*math.pi),
            'temp_max': 22.0 + 4*math.sin(hour/24*2*math.pi),
            'data_source': 'Simulated Nairobi Data'
        }
    
@st.cache_data
def validate_nairobi_location(lat, lon):
    return (NAIROBI_BOUNDS['min_lat'] <= lat <= NAIROBI_BOUNDS['max_lat'] and
            NAIROBI_BOUNDS['min_lon'] <= lon <= NAIROBI_BOUNDS['max_lon'])

@st.cache_data
def geocode_location(location_name):
    geolocator = Nominatim(user_agent="nairobi_pm25_app")
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(location_name + ", Nairobi, Kenya")
            if location:
                return location.latitude, location.longitude
            return None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt == max_retries - 1:
                st.error(f"Geocoding failed: {str(e)}")
                return None
            time.sleep(retry_delay)
    return None

@st.cache_resource
def get_explainer(X_train):
    """Create and cache a LIME explainer for the model"""
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns,
            class_names=['PM2.5'],
            mode='regression',
            verbose=False,
            discretize_continuous=True,
            random_state=42
        )
        return explainer
    except Exception as e:
        st.error(f"Could not create explainer: {str(e)}")
        return None        