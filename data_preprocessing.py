import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load
import os

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
    
    # Remove timezone information from the timestamp column
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
                         'feels_like', 'humidity', 'temp_max']
    
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    X_val = X_val[selected_features]
    
    # Verify the shapes of the datasets
    st.write("Training data shape:", X_train.shape)
    st.write("Testing data shape:", X_test.shape)
    st.write("Validation data shape:", X_val.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# loading the model pickle file
@st.cache_resource
def model_load():
    try:
        models, results = load('all_models_metadata.pkl')
        
        # Load training data
        data = read_data()
        X_train, _, _, y_train, _, _ = train_test_split_data(data)
        
        # Calculate training metrics if not already in results
        formatted_results = []
        for model_name, model in models.items():
            # Find corresponding results or initialize empty
            result = next((r for r in results if r['model_name'] == model_name), {})
            
            # Calculate training metrics if not present
            if 'training_metrics' not in result:
                y_train_pred = model.predict(X_train)
                result['training_metrics'] = {
                    'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                    'r2': r2_score(y_train, y_train_pred)
                }
            
            # Ensure test metrics exist
            if 'test_metrics' not in result:
                result['test_metrics'] = {'rmse': None, 'r2': None}
            
            # Format for consistent DataFrame structure
            formatted = {
                'model_name': model_name,
                'train_rmse': result['training_metrics']['rmse'],
                'test_rmse': result['test_metrics']['rmse'],
                'train_r2': result['training_metrics']['r2'],
                'test_r2': result['test_metrics']['r2'],
                'feature_importance': result.get('feature_importance')
            }
            formatted_results.append(formatted)
            
        return models, formatted_results
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return {}, []

def display_model_results(model, model_results, X_test, y_test):
    # Show saved metrics if available
    if model_results and 'test_metrics' in model_results:
        st.subheader("Saved Performance Metrics")
        cols = st.columns(3)
        cols[0].metric("RMSE", f"{model_results['test_metrics']['rmse']:.2f}")
        cols[1].metric("MAE", f"{model_results['test_metrics']['mae']:.2f}")
        cols[2].metric("R²", f"{model_results['test_metrics']['r2']:.2f}")
    else:
        # Calculate live metrics
        y_pred = model.predict(X_test)
        cols = st.columns(3)
        cols[0].metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        cols[1].metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        cols[2].metric("R²", f"{r2_score(y_test, y_pred):.2f}")
        
        # Plot actual vs predicted
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        st.pyplot(fig)
    
    # Show feature importance if available
    if model_results and 'feature_importance' in model_results:
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame(
            model_results['feature_importance'].items(),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
    elif hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

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
