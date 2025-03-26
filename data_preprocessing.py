import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load
import os
import pickle
from datetime import datetime
import plotly.express as px


from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

# Required imports for model unpickling
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import lime
import lime.lime_tabular
from lime import lime_tabular


# loading the data and caching
@st.cache_data
def read_data():
    # Load and preprocess the data
    historical_data = pd.read_csv('https://raw.githubusercontent.com/Lorraine254/Data/refs/heads/main/nairobi_pm25_weather_historical.csv', parse_dates=['Unnamed: 0'])
    
    # Renaming the column
    historical_data.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    
    # First convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(historical_data['timestamp']):
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
    
    # Now safely remove timezone information
    historical_data['timestamp'] = historical_data['timestamp'].dt.tz_localize(None)
    
    # Extract temporal features
    historical_data['hour'] = historical_data['timestamp'].dt.hour
    historical_data['year'] = historical_data['timestamp'].dt.year
    historical_data['week'] = historical_data['timestamp'].dt.isocalendar().week
    
    return historical_data

# Function to create lag features
@st.cache_data
def create_lag_features(df, lag=5):
    for i in range(1, lag+1):
        df[f'lag_{i}'] = df['pm2.5'].shift(i)
    df.dropna(inplace=True)
    return df

# # Function to split data
@st.cache_data
def train_test_split_data(df):

    # Create lag features
    df_lagged = create_lag_features(df, lag=5)
    
    # Defining X and y
    X = df_lagged.drop(columns=['pm2.5'])
    y = df_lagged['pm2.5']

    # First split: Separate training + validation (80%) and testing (20%)
    X_train_val, X_val, y_train_val, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Second split: Separate training (64%) and validation (16%)
    X_train, X_test, y_train, y_test = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Select specific columns for each split
    selected_features = ['lag_1', 'lag_2', 'hour', 'year', 'lag_5', 'lag_3', 'lag_4', 
                         'dew_point', 'wind_speed', 'wind_deg', 'pressure', 'week', 
                         'temperature', 'humidity', 'temp_max']
    
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_val = X_val[selected_features]
    
    # Verify the shapes of the datasets
    #st.write("Training data shape:", X_train.shape)
    #st.write("Testing data shape:", X_test.shape)
    #st.write("Validation data shape:", X_val.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# loading the model pickle file
@st.cache_resource
def load_models():
    """Load all models and metadata with proper test data handling"""
    try:
        with open('all_models.pkl', 'rb') as f:
            models_data = pickle.load(f)

        # Backward compatibility check
        if isinstance(models_data, tuple):  # Old format
            models, results = models_data
            models_data = {
                'models': models,
                'lgbm_metadata': {
                    'ci_models': {'upper': None, 'lower': None},
                    'test_metrics': {},
                    'forecast_decay': None
                },
                'metadata': {'created_at': datetime.now(), 'version': '1.0'}
            }

        # Load and split the data
        data = read_data()
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_data(data)
        
        # Prepare model results
        formatted_results = []
        for model_name, model in models_data['models'].items():
            # Calculate fresh metrics for all models
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, model.predict(X_train))),
                'train_r2': r2_score(y_train, model.predict(X_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, model.predict(X_test))),
                'test_r2': r2_score(y_test, model.predict(X_test))
            }

            feature_names = list(X_train.columns)

            # Get feature importance
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, 'get_score'):
                importance = model.get_score(importance_type='weight')
            
            formatted_results.append({
                'model_name': model_name,
                **metrics,
                'feature_importance': importance
            })

        # Return three consistent values
        return (
            models_data['models'],  # Original models dict
            formatted_results,      # Formatted metrics
            {                       # Structured metadata
                'features': feature_names,
                'lgb_upper': models_data['lgbm_metadata'].get('ci_models', {}).get('upper'),
                'lgb_lower': models_data['lgbm_metadata'].get('ci_models', {}).get('lower'),
                'lgbm_metadata': models_data['lgbm_metadata']
            }
        )
            
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return {}, [], {}

def display_model_results(model, model_results, X_test, y_test):

    """Enhanced version with better visualization and error handling"""
    with st.container():
        # Header with model type
        st.subheader(f"Model Performance: {type(model).__name__}")
        
        # Metrics columns
        cols = st.columns(3)
        
        # Calculate metrics if not provided
        if not model_results or 'test_rmse' not in model_results:
            y_pred = model.predict(X_test)
            model_results = {
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'test_mae': mean_absolute_error(y_test, y_pred),
                'test_r2': r2_score(y_test, y_pred)
            }
        
        # Display metrics
        cols[0].metric("RMSE", f"{model_results['test_rmse']:.2f}", 
                      help="Root Mean Squared Error (lower is better)")
        cols[1].metric("MAE", f"{model_results['test_mae']:.2f}" if 'test_mae' in model_results else "N/A",
                      help="Mean Absolute Error (lower is better)")
        cols[2].metric("R²", f"{model_results['test_r2']:.2f}",
                      help="Coefficient of Determination (closer to 1 is better)")
        
        # Actual vs Predicted plot
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pred = model.predict(X_test)
        sns.regplot(x=y_test, y=y_pred, ax=ax, line_kws={'color': 'red'})
        ax.set_xlabel("Actual PM2.5")
        ax.set_ylabel("Predicted PM2.5")
        ax.set_title("Actual vs Predicted Values")
        st.pyplot(fig)
        
        # Feature importance
        st.subheader("Feature Importance")
        try:
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plotly interactive chart
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                with st.expander("View Detailed Importance Scores"):
                    st.dataframe(importance.style.format({'Importance': '{:.3f}'}))
                    
            elif hasattr(model, 'get_score'):  # For XGBoost
                importance = pd.DataFrame(
                    model.get_score(importance_type='weight').items(),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Could not display feature importance: {str(e)}")

# Add these constants for Nairobi bounds
NAIROBI_BOUNDS = {
    'min_lat': -1.47,
    'max_lat': -1.15,
    'min_lon': 36.65,
    'max_lon': 37.05
}

# Add this new function for location validation
@st.cache_data
def validate_nairobi_location(lat, lon):
    """Validate coordinates are within Nairobi bounds"""
    return (NAIROBI_BOUNDS['min_lat'] <= lat <= NAIROBI_BOUNDS['max_lat'] and
            NAIROBI_BOUNDS['min_lon'] <= lon <= NAIROBI_BOUNDS['max_lon'])

# Add this function for geocoding
@st.cache_data
def geocode_location(location_name):
    """Convert location name to coordinates with retry logic"""
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

# Create a button to download the model
def download_objects(file_path):
    """Handle file downloads"""
    with open(file_path, "rb") as file:
        btn = st.sidebar.download_button(
            label=f"Download {os.path.basename(file_path)}",
            data=file,
            file_name=os.path.basename(file_path)
        )  
