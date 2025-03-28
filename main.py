import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from data_preprocessing import (read_data, create_lag_features, 
                              train_test_split_data, load_models,
                              validate_nairobi_location, geocode_location,fetch_weather)
from visuals import (get_aqi_category, get_aqi_color, display_forecast_results)
import numpy as np
from datetime import datetime, timedelta, time
import math
import random
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Nairobi PM2.5 Prediction Tool", page_icon="🪂", layout="wide")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(None, ["Home","Methodology", "Prediction"], 
                         icons=['house', 'book','gear'], 
                         menu_icon="cast", default_index=0,
                         styles={
                             "icon": {"color": "orange", "font-size": "16px"},
                             "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                             "nav-link-selected": {"background-color": "green"},
                         })
    
    with st.expander("❓ How to Use This Tool"):
        st.markdown("""
        1. **Navigate to the prediction tab.**
        2. **Choose your forecast start date (default: today)**  
        3. **Select start hour (24-hour format)** 
        4. **Enter PM₂.₅ measurements from the last 5 hours (in µg/m³)**  
        5. Click **Generate Forecast**  
        """)
        st.table(pd.DataFrame({
            "AQI Category": ["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"],
            "PM2.5 Range": ["0-12", "12-35", "35-55", "55-150", "250+"],
            "Color": ["🟢", "🟡", "🟠", "🔴", "🟤"]
        }))
        

if selected == "Home":
    st.title("PM2.5 Forecasting System")
    st.markdown("""
    ### Nairobi Air Quality Prediction
    This system provides 24-hour PM2.5 forecasts with confidence intervals.
    """)
    
    
    st.subheader(":orange[Air Quality Sensor Network]")
    
    # Sensor coordinates data
    sensor_locations = {
        'Latitude': [-1.33, -1.327, -1.322, -1.32, -1.316, -1.316, -1.316, -1.316,
                    -1.306, -1.306, -1.303, -1.303, -1.301, -1.3, -1.298, -1.297,
                    -1.297, -1.296, -1.295, -1.295, -1.293, -1.292, -1.291, -1.291,
                    -1.291, -1.29, -1.289, -1.288, -1.287, -1.283, -1.27, -1.267,
                    -1.265, -1.261, -1.261, -1.26, -1.259, -1.253, -1.251, -1.239,
                    -1.235, -1.22, -1.218, -1.215],
        'Longitude': [36.866, 36.882, 36.797, 36.885, 36.79, 36.793, 36.87, 36.872,
                     36.733, 36.773, 36.789, 36.829, 36.754, 36.785, 36.791, 36.743,
                     36.755, 36.776, 36.777, 36.86, 36.769, 36.821, 36.725, 36.733,
                     36.781, 36.777, 36.825, 36.841, 36.811, 36.828, 36.801, 36.8,
                     36.857, 36.772, 36.782, 36.793, 36.799, 36.854, 36.923, 36.791,
                     36.854, 36.879, 36.887, 36.862]
    }
    
    sensor_df = pd.DataFrame(sensor_locations)
    st.map(sensor_df, latitude='Latitude', longitude='Longitude', color='#FF4B4B', size=15, zoom=10)
    st.caption("Figure 1: Distribution of air quality sensors used in this study")

elif selected == "Prediction":
    st.markdown("### :orange[24-Hour PM2.5 Forecast with Confidence Intervals]")
    
    # Initialize session state for predictions if it doesn't exist
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
    
    # Load models
    models = load_models()
    if not models:
        st.stop()

    # Fixed Nairobi coordinates
    NAIROBI_COORDS = (-1.286389, 36.817223)
    today = datetime.now().date()  # Define 'today' variable
    
    # Compact form with border
    with st.form('pm25_form', border=True):
        # Date and Hour Input section (formerly in expander)
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced date input with tomorrow as default
            min_date = today
            max_date = today + timedelta(days=5)  # 5-day forecast limit
            default_date = today   # Default to today
            
            forecast_date = st.date_input(
                "Choose your forecast start date (any date from today through [today + 5 days]:",
                min_value=min_date,
                max_value=max_date,
                value=default_date,
                help="Weather data available for up to 5 days in advance"
            )
            
            # Show warning if trying to forecast beyond API limits
            if forecast_date > today + timedelta(days=5):
                st.warning("Note: Forecast limited to 5 days ahead")

            start_hour = st.slider("Choose the starting hour:", 0, 23, 8, help="Use the 24-hour clock system")  # Default to 8 AM
        
        with col2: 
            # Weather data handling
            if forecast_date == today + timedelta(days=1):  # If forecasting tomorrow
                st.info("""
                **Weather Data Note:**  
                Using today's latest observed weather as baseline,
                with tomorrow's forecasted weather patterns
                """)
        
        # Location display (fixed to Nairobi)
        # st.markdown(f"**Location:** Nairobi (Lat: {NAIROBI_COORDS[0]:.4f}, Lon: {NAIROBI_COORDS[1]:.4f})")
        
        # PM2.5 history inputs (formerly in expander)
        st.markdown("**Enter PM2.5 measurements from the last 5 hours (in µg/m³)**")
        pm_cols = st.columns(5)
        pm_history = []
        for i, col in enumerate(pm_cols, 1):
            with col:
                pm_history.append(
                    st.number_input(f"{i} hour ago", 
                                min_value=0.0, 
                                max_value=100.0, 
                                value=15.0,
                                key=f"pm_{i}")
                )
        
        submitted = st.form_submit_button("Generate Forecast")

    if submitted:
        try:
            # Get weather data (either current or forecast)
            if forecast_date == today + timedelta(days=1):  # Tomorrow's forecast
                current_weather = fetch_weather(*NAIROBI_COORDS, datetime.now())
                forecast_weather = fetch_weather(*NAIROBI_COORDS, 
                                              datetime.combine(forecast_date, time(start_hour, 0)))
                weather = {
                    **current_weather,
                    **{k: forecast_weather[k] for k in ['temperature', 'wind_speed', 'humidity']}
                }
            else:  # Today or other dates
                weather = fetch_weather(*NAIROBI_COORDS, datetime.combine(forecast_date, time(start_hour, 0)))
                if weather.get('data_source') == 'Simulated Data':
                    st.error("⚠️ Using simulated weather data (API unavailable)")
                else:
                    st.success("✅ Live weather data loaded")
            
            # API status feedback
            if weather.get('data_source', '').startswith('OpenWeatherMap'):
                st.toast("🌐 Data successfully retrieved from API", icon="✅")
            
            base_time = datetime.combine(forecast_date, time(start_hour, 0))
            hours = [base_time + timedelta(hours=i) for i in range(24)]
            
            predictions = []
            current_lags = pm_history.copy()
            
            with st.spinner("Generating forecast..."):
                progress_bar = st.progress(0)

                for i, hour in enumerate(hours):
                    # Use the weather data we already fetched
                    features = {
                        'lag_1': current_lags[-1],
                        'lag_2': current_lags[-2],
                        'lag_3': current_lags[-3],
                        'lag_4': current_lags[-4],
                        'lag_5': current_lags[-5],
                        'hour': hour.hour,
                        'week': hour.isocalendar()[1],
                        'year': hour.year,
                        'dew_point': weather['dew_point'],
                        'wind_speed': weather['wind_speed'],
                        'wind_deg': weather['wind_deg'],
                        'pressure': weather['pressure'],
                        'humidity': weather['humidity'],
                        'temperature': weather['temperature'],
                        'temp_max': weather['temp_max']
                    }
                    
                    input_df = pd.DataFrame([features])
                    
                    pred = models['main'].predict(input_df)[0]
                    upper = models['upper'].predict(input_df)[0]
                    lower = models['lower'].predict(input_df)[0]
                    
                    predictions.append({
                        'timestamp': hour.strftime('%Y-%m-%d %H:%M'),
                        'prediction': pred,
                        'upper_95': upper,
                        'lower_05': lower,
                        'aqi_category': get_aqi_category(pred),
                        'data_source': weather.get('data_source', 'OpenWeatherMap API')
                    })
                    
                    current_lags.pop(0)
                    current_lags.append(pred)

                    # Update progress after each hour is processed
                    progress_bar.progress((i + 1) / len(hours)) 
                     
            progress_bar.empty()
            st.session_state.forecast_results = pd.DataFrame(predictions)
            
        except Exception as e:
            st.error(f"Forecast generation failed: {str(e)}")

    # Display results and download button OUTSIDE the form
    if st.session_state.forecast_results is not None:
        display_forecast_results(st.session_state.forecast_results)
        
        # Download button
        csv = st.session_state.forecast_results.to_csv(index=False)
        st.download_button(
            label="Download Forecast Data",
            data=csv,
            file_name=f"pm25_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

elif selected == "Methodology":
    st.markdown("""
    ### Model Architecture
    - **Algorithm**: LightGBM Ensemble with Quantile Regression  
    - **Features**:  
    - Temporal: Hour of day, week of year  
    - PM2.5 Lags: 1-5 hour history  
    - Weather: Temperature, humidity, wind speed  
    - **Confidence Intervals**: 5th-95th percentiles  
    """)
    #st.image("model_architecture.png")