import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from data_preprocessing import (read_data,create_lag_features,train_test_split_data,load_models,display_model_results,download_objects,get_explainer,validate_nairobi_location,geocode_location)
from visuals import (plot_model_results,get_aqi_category,get_aqi_color,display_forecast_results,get_aqi_category)
import numpy as np
import lime
import lime.lime_tabular
from lime import lime_tabular
from datetime import datetime
import plotly.express as px
import calendar 
import geopandas as gpd
from libpysal import weights
from esda import G_Local
import joblib
from datetime import timedelta, time
import math
import random
import plotly.graph_objects as go


# Page configuration
st.set_page_config(page_title="Nairobi PM2.5 Prediction Tool", page_icon="🪂", layout="wide")


model_path = 'all_models.pkl'
note_book_path = 'model_building.html'


with st.sidebar:
    selected = option_menu(None, ["Home","EDA", "Models", "Prediction",'Interpretation'], 
    icons     =['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon ="cast", default_index=0, orientation="vertical",
    
    styles={
    "icon"              : {"color": "orange", "font-size": "16px"}, 
    "nav-link"          : {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected" : {"background-color": "green"},
    })

if selected == "Home":

    st.header(":orange[Nairobi PM2.5 Prediction Tool]", divider=True)
    st.write("Welcome to the PM2.5 prediction tool for Nairobi.") 
    st.write("The dataset used in this study was obtained from Sensors Africa and Openweather.")
    #st.write("Distribution of the sensors used in the study:")

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
    
    # Create DataFrame
    sensor_df = pd.DataFrame(sensor_locations)
    
    # Create the map - all points visible
    st.map(sensor_df,
          latitude='Latitude',
          longitude='Longitude',
          color='#FF4B4B',  # Nairobi orange
          size=15,  # Visible point size
          zoom=10)  # Nairobi city view
    
    # Minimal description
    st.caption("Figure 1: Distribution of 43 air quality sensors used in this study")
    st.write("Data source: Sensors Africa")

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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Time Trends", "🔍 Feature Relationships" ,"♨️ Hotspot Analysis", ])
    
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

        # Hypothesis Testing Explanation
        with st.expander("🔍 About the Hotspot Analysis Method", expanded=True):
         st.markdown("""
        **Spatial Hypothesis Being Tested:**
        
        *Null Hypothesis (H₀):* PM2.5 concentrations are randomly distributed across space  
        *Alternative Hypothesis (H₁):* PM2.5 concentrations show statistically significant spatial clustering
        
        We use the **Getis-Ord Gi* statistic** to test this hypothesis, which identifies:
        """)
        
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
            **🔥 Hotspots**  
            Areas with *significantly higher* PM2.5 values than expected by chance:
            - Z-score > +1.96 (p < 0.05)
            - Indicates pollution clustering
            """)
        with cols[1]:
            st.markdown("""
            **❄️ Coldspots**  
            Areas with *significantly lower* PM2.5 values than expected:
            - Z-score < -1.96 (p < 0.05)
            - Indicates clean air clustering
            """)
        with cols[2]:
            st.markdown("""
            **⚪ Normal Areas**  
            No significant spatial pattern:
            - -1.96 ≤ Z-score ≤ +1.96  
            - PM2.5 levels are spatially random
            """)
        
        st.markdown("""
        **Interpretation Guide:**
        - Z-scores measure standard deviations from the mean spatial pattern
        - p-values < 0.05 indicate statistical significance (95% confidence)
        - Analysis uses a 1km neighborhood radius (threshold=0.01 decimal degrees)
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
            
            # Calculate average PM2.5 across all years for each location
            pm_columns = [f'pm2.5_{year}' for year in range(2018, 2025)]
            hotspot_df['pm2.5_avg'] = hotspot_df[pm_columns].mean(axis=1)
            
            # Remove any rows with NaN values
            hotspot_df = hotspot_df.dropna(subset=['pm2.5_avg'])
            
            # Convert to GeoDataFrame
            gdf = gpd.GeoDataFrame(
                hotspot_df,
                geometry=gpd.points_from_xy(hotspot_df.longitude, hotspot_df.latitude),
                crs="EPSG:4326"
            )
            
            # Create spatial weights matrix (1km neighborhood)
            w = weights.DistanceBand.from_dataframe(gdf, threshold=0.01)
            
            # Perform Getis-Ord Gi* analysis
            gi = G_Local(gdf['pm2.5_avg'].values, w, star=True)
            
            # Add results to DataFrame
            gdf['gi_zscore'] = gi.z_sim
            gdf['gi_pvalue'] = gi.p_sim
            
            # Replace NaN values in z-scores with 0
            gdf['gi_zscore'] = gdf['gi_zscore'].fillna(0)
            
            # Classify hotspots and coldspots
            gdf['hotspot'] = np.where(
                (gdf['gi_zscore'] > 1.96) & (gdf['gi_pvalue'] < 0.05),
                "Hotspot",
                np.where(
                    (gdf['gi_zscore'] < -1.96) & (gdf['gi_pvalue'] < 0.05),
                    "Coldspot",
                    "Normal"
                )
            )
            
            # Create marker sizes - ensure no NaN values and scale appropriately
            gdf['marker_size'] = np.abs(gdf['gi_zscore']).replace(np.nan, 0)
            min_size, max_size = 5, 20
            if gdf['marker_size'].max() > 0:
                gdf['marker_size'] = min_size + (max_size - min_size) * (
                    (gdf['marker_size'] - gdf['marker_size'].min()) / 
                    (gdf['marker_size'].max() - gdf['marker_size'].min())
                )    
            
            # Create layout
            col_map, col_stats = st.columns([2, 1])
            
            with col_map:
                # Interactive hotspot map
                st.markdown("**Hotspot Map (All Years Average)**")
                fig = px.scatter_mapbox(
                    gdf,
                    lat="latitude",
                    lon="longitude",
                    color="hotspot",
                    color_discrete_map={
                        "Hotspot": "red",
                        "Coldspot": "blue",
                        "Normal": "black"
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
                # Statistics section
                st.markdown("**Analysis Results**")
                
                hotspots = gdf[gdf['hotspot'] == "Hotspot"]
                coldspots = gdf[gdf['hotspot'] == "Coldspot"]
                
                st.metric("Significant Hotspots", len(hotspots))
                st.metric("Significant Coldspots", len(coldspots))
                
                if not hotspots.empty:
                    st.markdown("**Top Hotspot Locations**")
                    st.dataframe(
                        hotspots.nlargest(3, 'gi_zscore')[['latitude', 'longitude', 'pm2.5_avg']],
                        hide_index=True
                    )
                
                if not coldspots.empty:
                    st.markdown("**Top Coldspot Locations**")
                    st.dataframe(
                        coldspots.nsmallest(3, 'gi_zscore')[['latitude', 'longitude', 'pm2.5_avg']],
                        hide_index=True
                    )
            
            # Full results expander
            with st.expander("View Complete Analysis Data"):
                st.dataframe(
                    gdf.sort_values('gi_zscore', ascending=False)[
                        ['latitude', 'longitude', 'pm2.5_avg', 'hotspot', 'gi_zscore', 'gi_pvalue']
                    ],
                    column_config={
                        "gi_zscore": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                        "gi_pvalue": st.column_config.NumberColumn("P-Value", format="%.4f"),
                        "pm2.5_avg": st.column_config.NumberColumn("Avg PM2.5", format="%.2f μg/m³")
                    },
                    height=300
                )
        
        except ImportError as e:
            st.error(f"Required packages not found: {str(e)}")
            st.info("Install with: pip install geopandas libpysal esda plotly")
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")


    
elif selected == "Models":
    st.subheader(':orange[Trained Models Information]', divider=True)
    
    # Load data and models
    with st.expander('Training Data (X_train)'):
        data = read_data()
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_data(data)
        st.write("Features (X_train):")
        st.dataframe(X_train)
        st.write("Target (y_train):")
        st.dataframe(y_train)
    
    loaded_models, loaded_model_results,models_metadata = load_models()
    
    if not loaded_models:
        st.warning("No models loaded successfully")
        st.stop()
    
    df = pd.DataFrame(loaded_model_results)
    
    # Metric comparison
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.plotly_chart(plot_model_results(df, 'rmse'), use_container_width=True)
            st.caption("Lower RMSE values indicate better performance")
    with col2:
        with st.container(border=True):
            st.plotly_chart(plot_model_results(df, 'r2'), use_container_width=True)
            st.caption("Higher R² values indicate better fit (1.0 is perfect)")
    
    # # Individual model details
    # for model_name, model in loaded_models.items():
    #     with st.expander(f"{model_name} Details", expanded=False):
    #         # Find corresponding results
    #         model_result = next((r for r in loaded_model_results if r['model_name'] == model_name), None)
            
    #         # Display metrics
    #         if model_result:
    #             cols = st.columns(2)
    #             with cols[0]:
    #                 st.subheader("Training Performance")
    #                 st.metric("RMSE", f"{model_result['train_rmse']:.3f}",
    #                          help="Root Mean Squared Error on training data")
    #                 st.metric("R²", f"{model_result['train_r2']:.3f}",
    #                          help="Variance explained on training data (0-1 scale)")
                    
    #                 # Training data predictions plot
    #                 y_train_pred = model.predict(X_train)
    #                 fig_train = px.scatter(
    #                     x=y_train,
    #                     y=y_train_pred,
    #                     labels={'x': 'Actual', 'y': 'Predicted'},
    #                     title=f"{model_name} Training Predictions"
    #                 )
    #                 fig_train.add_shape(type='line', x0=y_train.min(), y0=y_train.min(),
    #                                   x1=y_train.max(), y1=y_train.max())
    #                 st.plotly_chart(fig_train, use_container_width=True)
                
    #             with cols[1]:
    #                 st.subheader("Test Performance")
    #                 if model_result['test_rmse'] is not None:
    #                     st.metric("RMSE", f"{model_result['test_rmse']:.3f}",
    #                              help="Root Mean Squared Error on test data")
    #                     st.metric("R²", f"{model_result['test_r2']:.3f}",
    #                              help="Variance explained on test data (0-1 scale)")
                        
    #                     # Test data predictions plot
    #                     y_test_pred = model.predict(X_test)
    #                     fig_test = px.scatter(
    #                         x=y_test,
    #                         y=y_test_pred,
    #                         labels={'x': 'Actual', 'y': 'Predicted'},
    #                         title=f"{model_name} Test Predictions"
    #                     )
    #                     fig_test.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
    #                                       x1=y_test.max(), y1=y_test.max())
    #                     st.plotly_chart(fig_test, use_container_width=True)
    #                 else:
    #                     st.warning("Test metrics not available")
            
    #         # Feature importance
    #         if model_result and model_result.get('feature_importance'):
    #             st.subheader("Feature Importance")
    #             importance_df = pd.DataFrame(
    #                 model_result['feature_importance'].items(),
    #                 columns=['Feature', 'Importance']
    #             ).sort_values('Importance', ascending=False)
                
    #             tab1, tab2 = st.tabs(["Chart", "Table"])
    #             with tab1:
    #                 st.bar_chart(importance_df.set_index('Feature'))
    #             with tab2:
    #                 st.dataframe(importance_df.style.format({'Importance': '{:.3f}'}))
    
    # # Notebook display
    # with st.expander("Learn how the model was trained?", expanded=False):
    #     with open(note_book_path, 'r', encoding='utf-8') as f:
    #         html_data = f.read()
    #     components.html(html_data, height=1000, width=800, scrolling=True)
    
    # Download section
    st.sidebar.markdown("### Download")
    download_choice = st.sidebar.selectbox(
        label='Select what to download 👇', 
        options=["Serialized Model", "Notebook"]
    )
    
    if download_choice == 'Serialized Model':
        with open(model_path, "rb") as f:
            st.sidebar.download_button(
                label="Download Model",
                data=f,
                file_name="trained_models.pkl"
            )
    elif download_choice == 'Notebook':
        with open(note_book_path, "rb") as f:
            st.sidebar.download_button(
                label="Download Notebook",
                data=f,
                file_name="model_training.ipynb"
            )



elif selected == "Prediction":
    st.markdown("### :orange[24-Hour PM2.5 Forecast]")
    
    # Load models
    loaded_models, loaded_model_results, models_metadata  = load_models()
    
    # Display metrics based on model choice
    model_choice = st.selectbox("Select model", ["XGBoost", "LightGBM", "Hybrid"])

    if model_choice == "LightGBM":
            # Use the pre-calculated metrics from lgbm_metadata
            if 'lgbm_metadata' in models_metadata and 'test_metrics' in models_metadata['lgbm_metadata']:
                st.metric("Test RMSE", f"{models_metadata['lgbm_metadata']['test_metrics']['rmse']:.2f}")
            else:
                # Fallback to our freshly calculated metrics
                lgb_result = next((r for r in loaded_model_results if r['model_name'] == 'LightGBM'), None)
                if lgb_result:
                    st.metric("Test RMSE", f"{lgb_result['test_rmse']:.2f}")

    elif model_choice == "XGBoost":
            xgb_result = next((r for r in loaded_model_results if r['model_name'] == 'XGBoost'), None)
            if xgb_result:
                st.metric("Test RMSE", f"{xgb_result['test_rmse']:.2f}")

    elif model_choice == "Hybrid":
            hybrid_result = next((r for r in loaded_model_results if r['model_name'] == 'Stacked_model'), None)
            if hybrid_result:
                st.metric("Test RMSE", f"{hybrid_result['test_rmse']:.2f}")

    with st.form("forecast_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Date and time selection
            forecast_date = st.date_input("Forecast date", 
                                       min_value=datetime.now().date(),
                                       max_value=datetime.now().date() + timedelta(days=7))
            start_hour = st.slider("Starting hour", 0, 23, datetime.now().hour)
            
            # Location input options
            location_method = st.radio("Location input method:",
                                     ["Coordinates", "Place Name"],
                                     horizontal=True)
            
            if location_method == "Coordinates":
                lat = st.number_input("Latitude (-1.47 to -1.15)", 
                                    min_value=-1.47, max_value=-1.15, value=-1.2864)
                lon = st.number_input("Longitude (36.65 to 37.05)", 
                                   min_value=36.65, max_value=37.05, value=36.8172)
            else:
                location_name = st.text_input("Enter place name in Nairobi", "Westlands")
                if st.button("Geocode Location"):
                    coords = geocode_location(location_name)
                    if coords:
                        lat, lon = coords
                        st.success(f"Found coordinates: {lat:.4f}, {lon:.4f}")
                    else:
                        st.error("Could not find location. Please try different name or use coordinates.")
                        st.stop()
        
        with col2:
            # PM2.5 history input
            st.markdown("#### Last 5 Hours PM2.5 Readings")
            pm_history = [
                st.number_input(f"{i} hour ago (µg/m³)", 
                              min_value=0.0, 
                              max_value=50.0, 
                              value=15.0,
                              key=f"pm_{i}")
                for i in range(5, 0, -1)
            ]
        
        if st.form_submit_button("Generate 24-Hour Forecast"):
            try:
                # Validate location
                if not validate_nairobi_location(lat, lon):
                    st.error("Coordinates must be within Nairobi boundaries")
                    st.stop()
                
                # Validate PM history
                if any(pm is None for pm in pm_history):
                    st.warning("Please provide all historical PM2.5 values")
                    st.stop()
                
                # Generate timeline
                base_time = datetime.combine(forecast_date, time(start_hour, 0))
                hours = [base_time + timedelta(hours=i) for i in range(24)]
                
                # Get weather forecast - implement your actual API call here
                def fetch_nairobi_weather(lat, lon, target_time):
                    """Mock weather data - replace with real API call"""
                    return {
                        'dew_point': 15.0 + 5*math.sin(target_time.hour/24*2*math.pi),
                        'wind_speed': 3.0 + random.uniform(-1, 1),
                        'wind_deg': random.randint(0, 360),
                        'pressure': 1013 + random.randint(-10, 10),
                        'temperature': 20.0 + 10*math.sin(target_time.hour/24*2*math.pi),
                        'humidity': 50 + int(20*math.sin(target_time.hour/12*math.pi)),
                        'temp_max': 25.0 + 5*math.sin(target_time.hour/24*2*math.pi)
                    }
                # Get feature names - fallback to default if not in metadata
                if 'features' in models_metadata:
                        feature_columns = models_metadata['features']
                else:
                    # Default feature columns based on your training data
                    feature_columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                                        'hour', 'week', 'year', 'dew_point', 'wind_speed',
                                        'wind_deg', 'pressure', 'feels_like', 'humidity', 'temp_max']
                    
                
                # Generate predictions
                predictions = []
                current_lags = pm_history.copy()
                
                with st.spinner("Generating 24-hour forecast..."):
                    for hour in hours:
                        # Get weather
                        weather = fetch_nairobi_weather(lat, lon, hour)
                        
                        # Prepare features
                        features = {
                            'lag_1': current_lags[-1],
                            'lag_2': current_lags[-2],
                            'lag_3': current_lags[-3],
                            'lag_4': current_lags[-4],
                            'lag_5': current_lags[-5],
                            'hour': hour.hour,
                            'week': hour.isocalendar()[1],
                            'year': hour.year,
                            **weather
                        }
                        
                        # Select the appropriate model based on user choice
                        if model_choice == "LightGBM":
                            model = loaded_models['LightGBM']
                            model_upper = models_metadata['lgb_upper']
                            model_lower = models_metadata['lgb_lower']
                        elif model_choice == "XGBoost":
                            model = loaded_models['XGBoost']
                            # For XGBoost, you might not have upper/lower bounds
                            model_upper = model  # Or implement your own bounds
                            model_lower = model
                        else:  # Hybrid
                            model = loaded_models['Stacked_model']
                            # For Hybrid, you might not have upper/lower bounds
                            model_upper = model
                            model_lower = model
                        
                        # Make prediction
                        input_df = pd.DataFrame([features])[models_metadata['features']]
                        pred = model.predict(input_df)[0]
                        
                        # Only calculate bounds if models exist
                        pred_upper = model_upper.predict(input_df)[0] if model_upper else pred
                        pred_lower = model_lower.predict(input_df)[0] if model_lower else pred
                        
                        predictions.append({
                            'timestamp': hour.strftime('%Y-%m-%d %H:%M'),
                            'prediction': pred,
                            'upper_bound': pred_upper,
                            'lower_bound': pred_lower,
                            'aqi_category': get_aqi_category(pred)
                        })
                        
                        # Update lags
                        current_lags.pop(0)
                        current_lags.append(pred)
                
                # Display results
                display_forecast_results(pd.DataFrame(predictions))
                
            except Exception as e:
                st.error(f"Forecast generation failed: {str(e)}")


elif selected == "Interpretation":
    st.markdown("### :orange[Model Interpretation with LIME]")
    
    # Load data and models
    data = read_data()
    loaded_models, loaded_model_results, models_metadata  = load_models()
    X_train, _, X_test, _, _, y_test = train_test_split_data(data)
    
    if not loaded_models:
        st.error("No models available for interpretation")
        st.stop()
    
    # Get cached explainer
    explainer = get_explainer(X_train)
    if not explainer:
        st.stop()
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model to Interpret",
        options=list(loaded_models.keys())
    )
    model = loaded_models[model_name]
    
    # Instance selection
    instance_idx = st.sidebar.selectbox(
        "Select Data Instance",
        options=X_test.index.to_list(),
        format_func=lambda x: f"Instance {x} (Actual: {y_test.loc[x]:.1f} μg/m³)"
    )
    
    # Display selected instance
    st.dataframe(X_test.loc[[instance_idx]])
    st.markdown(f"**Actual PM2.5:** {y_test.loc[instance_idx]:.2f} μg/m³")
    st.markdown(f"**Predicted PM2.5:** {model.predict(X_test.loc[[instance_idx]])[0]:.2f} μg/m³")
    
    if st.button("Explain Prediction"):
        with st.spinner("Generating explanation..."):
            try:
                exp = explainer.explain_instance(
                    X_test.loc[instance_idx].values,
                    model.predict,
                    num_features=15
                )
                
                # Show explanation
                st.markdown("### Feature Impact on Prediction")
                cols = st.columns(2)
                with cols[0]:
                    st.pyplot(exp.as_pyplot_figure())
                with cols[1]:
                    components.html(exp.as_html(), height=500)
                
                # Detailed view
                with st.expander("Detailed Explanation"):
                    st.write("How features affected this specific prediction:")
                    for feature, weight in exp.as_list():
                        st.write(f"- {feature}: {weight:.4f}")
                        
            except Exception as e:
                st.error(f"Explanation failed: {str(e)}")