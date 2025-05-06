import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from data_preprocessing import (read_data, create_lag_features, 
                              train_test_split_data, load_models,
                              validate_nairobi_location, geocode_location,fetch_weather,get_explainer,get_today)
from visuals import (get_aqi_category, get_aqi_color, display_forecast_results)
import numpy as np
from datetime import datetime, timedelta, time
import math
import random
import plotly.graph_objects as go
import geopandas as gpd
from libpysal import weights
from esda import G_Local
import plotly.express as px
import calendar 
import streamlit.components.v1 as components



# Page configuration
st.set_page_config(page_title="Nairobi PM2.5 Prediction Tool", page_icon="🪂", layout="wide")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(None, ["Home","Methodology", "Prediction"], #,"Interpretation"
                         icons=['house', 'book','gear'], #,'play'
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
    # Add this debug check at the start of your Prediction tab
    #st.write("API key present:", "OPENWEATHER_API_KEY" in st.secrets)
    
    # Initialize session state for predictions if it doesn't exist
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
    
    # Load models
    models = load_models()
    if not models:
        st.stop()

    # Fixed Nairobi coordinates
    NAIROBI_COORDS = (-1.286389, 36.817223)
    today = get_today()  # Define 'today' variable
    
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

elif selected == "EDA":
    st.header(":orange[PM2.5 Air Quality Analysis]", divider=True)
    
    # Load data with preprocessing pipeline
    @st.cache_data
    def load_eda_data():
        df = read_data()  # Handles timestamp conversion and feature creation
        df = create_lag_features(df, lag=5)  # Adds lag_1 through lag_5
        return df

    df = load_eda_data()

    # Create tabs for different EDA views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Time Trends", "🔍 Feature Relationships" ,"♨️ Hotspot Analysis"])
    
    with tab1:
        st.subheader("Processed Data Preview")
        st.dataframe(df.head(3))
        
        # Enhanced statistics display
        st.subheader("Basic Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numeric Features**")
            numeric_stats = df.select_dtypes(include=['float64']).describe().T
            st.dataframe(numeric_stats.style.format("{:.2f}"))
            
        with col2:
            st.markdown("**Temporal Features**")
            time_stats = df[['hour', 'year', 'week']].describe().T
            st.dataframe(time_stats.style.format("{:.2f}"))
        
        st.subheader("Data Quality Check")
        missing = df.isna().sum().to_frame("Missing Values")
        st.dataframe(missing[missing["Missing Values"] > 0])
        #st.caption("Note: Lag features may introduce missing values at start of time series")


    with tab2:
        st.subheader("Temporal Patterns")
        
        # Time series analysis
        df_time = df.set_index('timestamp').resample('D').mean()
        
        # Create basic line plot
        fig = px.line(df_time, x=df_time.index, y='pm2.5',
                    title="Daily PM2.5 Concentration with Reference Lines",
                    labels={'pm2.5': 'PM2.5 (μg/m³)'})
        
        # Add the three reference lines only
        # fig.add_hline(y=9, line_dash="dot", line_color="green",
        #             annotation_text="Annual Standard (9 μg/m³)", 
        #             annotation_position="bottom right")
        
        fig.add_hline(y=35, line_dash="dash", line_color="red",
                    annotation_text="24-hour Standard (35 μg/m³)",
                    annotation_position="top left")
        
        st.plotly_chart(fig, use_container_width=True)

            # Monthly analysis - matching daily plot style
        #st.subheader("Monthly PM2.5 Concentration")
        df_time = df.set_index('timestamp')
        df_time['month'] = df_time.index.month
        monthly_avg = df_time.groupby('month')['pm2.5'].mean().reset_index()
        
        # Convert to month names and order correctly
        monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: calendar.month_abbr[x])  # Using abbreviations like "Jan"
        month_order = [calendar.month_abbr[i] for i in range(1,13)]
        monthly_avg['month_name'] = pd.Categorical(monthly_avg['month_name'], categories=month_order, ordered=True)
        monthly_avg = monthly_avg.sort_values('month_name')

        fig_monthly = px.line(monthly_avg,
                            x='month_name',
                            y='pm2.5',
                            title="Monthly PM2.5 Concentration",
                            labels={'pm2.5': 'PM2.5 (μg/m³)', 'month_name': 'Month'})
        
        # Exact same styling as daily plot
        #fig_monthly.update_traces(line=dict(width=2, color='blue')  # No markers to match daily)
        
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Day of week analysis (similarly styled)
        #st.subheader("Weekly PM2.5 Pattern")
        df_time = df.set_index('timestamp')
        df_time['day_name'] = df_time.index.day_name()
        day_of_week_avg = df_time.groupby('day_name')['pm2.5'].mean()
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week_avg = day_of_week_avg.reindex(day_order)
        
        fig_weekly = px.line(day_of_week_avg,
                            x=day_of_week_avg.index,
                            y='pm2.5',
                            title="Weekly PM2.5 Pattern",
                            labels={'pm2.5': 'PM2.5 (μg/m³)', 'x': 'Day of Week'})
        
        fig_weekly.update_xaxes(tickangle=45)
        fig_weekly.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        
        st.plotly_chart(fig_weekly, use_container_width=True)

         #Add hourly average line plot
         #Hourly average line plot (styled to match daily plot)
        #st.subheader("Hourly PM2.5 Concentration")
        hourly_avg = df.groupby('hour')['pm2.5'].mean().reset_index()
        
        fig_hourly = px.line(hourly_avg,
                            x='hour',
                            y='pm2.5',
                            title="Hourly PM2.5 Concentration",
                            labels={'pm2.5': 'PM2.5 (μg/m³)', 'hour': 'Hour of Day'})
        
        # Match the daily plot's styling
        #fig_hourly.update_traces(line=dict(width=2, color='blue')) # Simpler line style)
        
        # Format x-axis to show all hours
        fig_hourly.update_xaxes(
            tickvals=list(range(24)),
            ticktext=[f"{h}:00" for h in range(24)],
            tickangle=45
        )
        
        # Add grid matching the daily plot
        fig_hourly.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        
        st.plotly_chart(fig_hourly, use_container_width=True)



    with tab3:
        st.subheader("Feature Analysis")
        
        # Focused correlation matrix
        focus_features = ['pm2.5'] + [f'lag_{i}' for i in [1,2,3,4,5]] + \
                        ['humidity','dew_point', 'wind_speed','temp','wind_deg']
        corr_matrix = df[focus_features].corr()
        
        fig5 = px.imshow(corr_matrix,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale='RdBu',
                        title="Selected Feature Correlations")
        st.plotly_chart(fig5, use_container_width=True)
        
        # Lag feature visualization
        st.subheader("Lag Feature Impact")
        fig6 = px.scatter(df, x='lag_1', y='pm2.5',
                         trendline="ols",
                         title="Current vs Previous Hour PM2.5")
        st.plotly_chart(fig6, use_container_width=True)



    with tab4:
        st.subheader("PM2.5 Hotspot Analysis (All Years Combined)")
        # Hypothesis Testing Explanation - First expander
        with st.expander("🔍 About the Hotspot Analysis Method", expanded=True):
                st.markdown("""
                **Spatial Hypothesis Test:**  
                *Null Hypothesis (H₀):* PM2.5 concentrations are randomly distributed across space.  
                *Alternative (H₁):* PM2.5 concentrations show statistically significant clustering (hotspots or coldspots).  

                We use the **Getis-Ord Gi* statistic** to test this hypothesis, which identifies:
                """)
                
                cols = st.columns(3)
                with cols[0]:
                    st.markdown("""
                    **🔥 Hotspots**  
                    Areas with *significantly higher* PM2.5 values than expected by chance:  
                    - Z-score > +1.96 **AND** p < 0.05  
                    - Indicates **pollution clustering**  
                    """)
                with cols[1]:
                    st.markdown("""
                    **❄️ Coldspots**  
                    Areas with *significantly lower* PM2.5 values than expected: 
                    - Z-score < -1.96 **AND** p < 0.05  
                    - Indicates **clean air clustering**  
                    """)
                with cols[2]:
                    st.markdown("""
                    **⚪ Normal Areas**  
                    *No significant clustering*:  
                    - |Z-score| ≤ 1.96 **OR** p ≥ 0.05  
                    - PM2.5 is spatially random  
                    """)
                
                st.markdown(f"""
                **Key Notes for Interpretation:**  
                - **Z-score** → Strength/Direction of clustering (🔥 +/❄️ -).  
                - **p-value < 0.05** → 95% confidence to reject H₀.
                """)

        try:
            # Create sensor locations DataFrame
            sensor_locations = {
                'latitude': [-1.33, -1.327, -1.322, -1.32, -1.316, -1.316, -1.316, -1.316,
                            -1.306, -1.306, -1.303, -1.303, -1.301, -1.3, -1.298, -1.297,
                            -1.297, -1.296, -1.295, -1.295, -1.293, -1.292, -1.291, -1.291,
                            -1.291, -1.29, -1.289, -1.288, -1.287, -1.283, -1.27, -1.267,
                            -1.265, -1.261, -1.261, -1.26, -1.259, -1.253, -1.251, -1.239,
                            -1.235, -1.22, -1.218, -1.215],
                'longitude': [36.866, 36.882, 36.797, 36.885, 36.79, 36.793, 36.87, 36.872,
                            36.733, 36.773, 36.789, 36.829, 36.754, 36.785, 36.791, 36.743,
                            36.755, 36.776, 36.777, 36.86, 36.769, 36.821, 36.725, 36.733,
                            36.781, 36.777, 36.825, 36.841, 36.811, 36.828, 36.801, 36.8,
                            36.857, 36.772, 36.782, 36.793, 36.799, 36.854, 36.923, 36.791,
                            36.854, 36.879, 36.887, 36.862],
                'pm2.5_2018': [24.08, 62.11, 13.99, 72.08, 10.5, 20.31, 11.57, 12.33, 
                            12.23, 12.22, 9.81, 12.93, 14.83, 9.57, 9.82, 9.17, 11.58, 
                            7.9, 7.9, 17.05, 6.69, 10.09, 10.48, 13.99, 7.1, 7.63, 
                            12.34, 20.18, 14.78, 13.05, 9.1, 8.67, 10.39, 9.93, 
                            9.41, 6.52, 10.72, 4.38, 7.48, 9.85, 3.66, 0.56, 5.02, 6.88],
                'pm2.5_2019': [4.4, 5.74, 8.89, 15.44, 8.45, 8.9, 5.24, 48.08, 
                            8.75, 7.95, 7.04, 10.17, 7.88, 7.52, 16.12, 7.28, 7.14, 
                            6.12, 6.12, 15.05, 1.56, 8.66, 16.82, 4.55, 5.3, 6.46, 
                            9.66, 11.24, 15.75, 10.1, 8.0, 8.0, 10.65, 2.14, 
                            4.33, 6.57, 6.89, 9.74, 9.33, 7.13, 10.08, 2.91, 1.1, 8.02],
                'pm2.5_2020': [7.58, 6.43, 14.61, 7.4, 14.99, 13.71, 6.85, 6.9, 
                            12.41, 29.08, 7.56, 13.49, 13.61, 9.58, 13.82, 8.86, 
                            12.83, 10.34, 10.34, 15.16, 13.43, 8.78, 11.46, 11.46, 
                            10.34, 8.83, 12.5, 24.13, 14.45, 14.82, 25.85, 25.28, 
                            17.76, 19.83, 22.52, 21.86, 21.86, 16.82, 13.88, 19.03, 
                            16.57, 15.67, 15.43, 16.17],
                'pm2.5_2021': [9.87, 9.82, 19.87, 9.23, 18.61, 19.43, 5.28, 5.44, 
                            34.16, 7.27, 17.98, 18.86, 25.3, 17.98, 16.25, 26.83, 
                            26.49, 24.65, 25.19, 14.18, 27.0, 18.34, 52.64, 48.88, 
                            24.65, 24.65, 18.78, 18.88, 20.83, 19.41, 24.53, 24.52, 
                            20.32, 50.56, 21.52, 25.53, 25.53, 22.26, 21.97, 28.76, 
                            23.78, 26.8, 27.21, 25.24],
                'pm2.5_2022': [17.17, 17.96, 17.12, 17.9, 16.45, 16.87, 16.54, 16.57, 
                            17.92, 14.57, 16.04, 20.04, 16.59, 16.04, 16.11, 19.3, 
                            16.19, 13.54, 13.54, 21.74, 9.32, 11.1, 18.19, 18.19, 
                            13.54, 13.54, 22.16, 21.88, 17.6, 20.02, 24.74, 24.17, 
                            38.62, 21.0, 14.31, 18.82, 18.82, 31.55, 22.26, 10.17, 
                            24.9, 22.89, 22.75, 22.58],
                'pm2.5_2023': [47.81, 42.01, 18.78, 43.04, 15.68, 17.01, 54.92, 54.52, 
                            22.54, 11.33, 9.52, 24.62, 20.63, 9.52, 5.91, 24.62, 
                            21.23, 16.59, 16.04, 18.99, 16.27, 24.26, 23.28, 23.28, 
                            16.04, 16.04, 20.71, 28.67, 23.74, 41.93, 12.51, 13.51, 
                            32.9, 23.65, 20.99, 22.38, 22.38, 31.04, 29.79, 61.54, 
                            30.01, 29.56, 29.49, 29.77],
                'pm2.5_2024': [41.99, 38.03, 18.78, 38.71, 15.67, 16.31, 46.88, 46.6, 
                            25.4, 17.28, 6.55, 19.71, 24.41, 6.55, 2.69, 28.02, 
                            24.41, 16.84, 16.23, 18.39, 16.99, 25.81, 26.13, 26.13, 
                            16.23, 16.23, 24.57, 29.12, 25.67, 40.07, 23.89, 23.89, 
                            41.0, 22.43, 22.93, 23.01, 23.01, 36.11, 30.24, 23.79, 
                            31.9, 30.32, 30.21, 30.2]
            }
            
            hotspot_df = pd.DataFrame(sensor_locations)
            
            # Calculate average PM2.5 across all years
            pm_columns = [f'pm2.5_{year}' for year in range(2018, 2025)]
            hotspot_df['pm2.5_avg'] = hotspot_df[pm_columns].mean(axis=1)
            hotspot_df = hotspot_df.dropna(subset=['pm2.5_avg'])

            # Convert to GeoDataFrame and reproject to UTM
            gdf = gpd.GeoDataFrame(
                hotspot_df,
                geometry=gpd.points_from_xy(hotspot_df.longitude, hotspot_df.latitude),
                crs="EPSG:4326"
            ).to_crs(epsg=32637)  # UTM Zone 37S for Kenya

            # Create spatial weights matrix (1km threshold in meters)
            w = weights.DistanceBand.from_dataframe(
                gdf, 
                threshold=1000,  # 1km in meters
                binary=False,
                silence_warnings=True
            )

            # Perform Getis-Ord Gi* analysis with fixed random seed
            np.random.seed(123)  # For reproducibility
            gi = G_Local(
                gdf['pm2.5_avg'].values, 
                w, 
                star=True,
                permutations=999
            )

            # Add results (drop NaNs instead of filling with 0)
            gdf['gi_zscore'] = gi.z_sim
            gdf['gi_pvalue'] = gi.p_sim
            gdf = gdf[~np.isnan(gdf['gi_zscore'])]

            # Classify hotspots/coldspots
            gdf['hotspot'] = np.where(
                (gdf['gi_zscore'] > 1.96) & (gdf['gi_pvalue'] < 0.05),
                "Hotspot",
                np.where(
                    (gdf['gi_zscore'] < -1.96) & (gdf['gi_pvalue'] < 0.05),
                    "Coldspot",
                    "Normal"
                )
            )

            # Scale marker sizes for visualization
            gdf['marker_size'] = np.where(
                gdf['hotspot'] != "Normal",
                np.abs(gdf['gi_zscore']),
                5  # Base size for normal points
            )
            min_size, max_size = 5, 20
            if gdf['marker_size'].max() > min_size:
                gdf['marker_size'] = min_size + (max_size - min_size) * (
                    (gdf['marker_size'] - gdf['marker_size'].min()) / 
                    (gdf['marker_size'].max() - gdf['marker_size'].min())
                )

            # Convert back to WGS84 for visualization
            gdf_vis = gdf.to_crs(epsg=4326)

            # Create layout
            col_map, col_stats = st.columns([2, 1])
            
            with col_map:
                st.markdown("**Hotspot Map (All Years Average)**")
                fig = px.scatter_mapbox(
                    gdf_vis,
                    lat="latitude",
                    lon="longitude",
                    color="hotspot",
                    color_discrete_map={
                        "Hotspot": "red",
                        "Coldspot": "blue",
                        "Normal": "gray"
                    },
                    size="marker_size",
                    hover_data=["pm2.5_avg", "gi_zscore", "gi_pvalue"],
                    zoom=11,
                    height=500
                )
                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r":0,"t":0,"l":0,"b":0}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_stats:
                st.markdown("**Analysis Results**")
                
                hotspots = gdf_vis[gdf_vis['hotspot'] == "Hotspot"]
                coldspots = gdf_vis[gdf_vis['hotspot'] == "Coldspot"]
                
                st.metric("Significant Hotspots", len(hotspots))
                st.metric("Significant Coldspots", len(coldspots))
                
                if not hotspots.empty:
                    st.markdown("**Top Hotspot Locations**")
                    st.dataframe(
                        hotspots.nlargest(3, 'gi_zscore')[
                            ['latitude', 'longitude', 'pm2.5_avg', 'gi_zscore']
                        ],
                        column_config={
                            "gi_zscore": st.column_config.NumberColumn(format="%.2f"),
                            "pm2.5_avg": st.column_config.NumberColumn(format="%.2f μg/m³")
                        },
                        hide_index=True
                    )
                
                if not coldspots.empty:
                    st.markdown("**Top Coldspot Locations**")
                    st.dataframe(
                        coldspots.nsmallest(3, 'gi_zscore')[
                            ['latitude', 'longitude', 'pm2.5_avg', 'gi_zscore']
                        ],
                        column_config={
                            "gi_zscore": st.column_config.NumberColumn(format="%.2f"),
                            "pm2.5_avg": st.column_config.NumberColumn(format="%.2f μg/m³")
                        },
                        hide_index=True
                    )
            
            # Second expander for complete results (not nested)
            with st.expander("View Complete Analysis Data"):
                st.dataframe(
                    gdf_vis.sort_values('gi_zscore', ascending=False)[
                        ['latitude', 'longitude', 'pm2.5_avg', 'hotspot', 'gi_zscore', 'gi_pvalue']
                    ],
                    column_config={
                        "gi_zscore": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                        "gi_pvalue": st.column_config.NumberColumn("P-Value", format="%.4f"),
                        "pm2.5_avg": st.column_config.NumberColumn("PM2.5", format="%.2f μg/m³")
                    },
                    height=300
                )

        except ImportError as e:
            st.error(f"Required packages not found: {str(e)}")
            st.info("Install with: pip install geopandas libpysal esda plotly")
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

elif selected == "Interpretation":
    st.markdown("### :orange[Model Interpretation with LIME]")
    
    # Load data and models
    data = read_data()
    models = load_models()

    if not models or 'main' not in models:
        st.error("Main model not available for interpretation")
        st.stop()
    
    # Get just the main model
    model = models['main']
    X_train, X_test,y_train, y_test = train_test_split_data(data)
    
    # Get cached explainer
    explainer = get_explainer(X_train)
    if not explainer:
        st.stop()
    
     # Combine features and target for display
    display_data = X_test.join(y_test.rename('Actual PM2.5'))
    
    # Instance selection
    data_instance = st.sidebar.selectbox(
        "Select a Data Instance",
        options=display_data.index.to_list()
    )
    
    # Display full data
    st.data_editor(
        display_data,
        use_container_width=True,
        height=250,
        column_config={
            "Actual PM2.5": st.column_config.NumberColumn(format="%.1f μg/m³")
        }
    )
    st.markdown('👈 Please select a Data Instance')
    
    if data_instance:  
        # Display selected instance
        data_picked = display_data.loc[[data_instance]]
        st.write('### Selected Instance Details')
        st.data_editor(
            data_picked,
            use_container_width=True,
            disabled=True,
            column_config={
                "Actual PM2.5": st.column_config.NumberColumn(format="%.1f μg/m³")
            }
        )
        
        # Show prediction
        prediction = model.predict(X_test.loc[[data_instance]])[0]
        st.metric(
            "Model Prediction", 
            f"{prediction:.1f} μg/m³",
            delta=f"{(prediction - data_picked['Actual PM2.5'].iloc[0]):.1f} vs actual"
        )
        
        # Interpretation toggle
        on = st.toggle("Show Interpretability", value=False)
        if on:
            with st.container(border=True):
                # Custom CSS to change background color
                st.markdown("""
                <style>
                    .lime-container {
                        background-color: #f0f0f0 !important;
                        padding: 20px;
                        border-radius: 10px;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                with st.spinner("Generating explanation..."):
                    try:
                        exp = explainer.explain_instance(
                            X_test.loc[data_instance].values,
                            model.predict,
                            num_features=15
                        )
                        components.html(
                            exp.as_html(), 
                            height=800, 
                            width=900, 
                            scrolling=True
                        )
                    except Exception as e:
                        st.error(f"Explanation failed: {str(e)}")
