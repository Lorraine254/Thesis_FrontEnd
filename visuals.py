import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st



def plot_model_results(df, metric):
    """Create comparison plots for models"""
    if metric == 'rmse':
        title = "RMSE Comparison"
        metric_cols = ['train_rmse', 'test_rmse']
        y_title = "RMSE"
    elif metric == 'r2':
        title = "R² Comparison"
        metric_cols = ['train_r2', 'test_r2']
        y_title = "R² Score"
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Melt the DataFrame for plotting
    plot_df = df.melt(
        id_vars=['model_name'],
        value_vars=metric_cols,
        var_name='dataset',
        value_name=metric
    )
    
    # Clean up dataset names
    plot_df['dataset'] = plot_df['dataset'].str.replace('train_', 'Training ').str.replace('test_', 'Test ')
    
    fig = px.bar(
        plot_df,
        x='model_name',
        y=metric,
        color='dataset',
        barmode='group',
        title=title,
        text_auto='.3f'
    )
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title=y_title,
        legend_title="Dataset"
    )
    return fig
    
def get_aqi_category(pm25):
        """Classify PM2.5 into AQI categories"""
        if pm25 < 12:
            return "Good"
        elif pm25 < 35:
            return "Moderate"
        elif pm25 < 55:
            return "Unhealthy for Sensitive Groups"
        elif pm25 < 150:
            return "Unhealthy"
        elif pm25 < 250:
            return "Very Unhealthy"
        else:
            return "Hazardous"

def get_aqi_color(pm25):
    """Get color for AQI category"""
    if pm25 < 12:
        return '#00E400'  # Green
    elif pm25 < 35:
        return '#FFFF00'  # Yellow
    elif pm25 < 55:
        return '#FF7E00'  # Orange
    elif pm25 < 150:
        return '#FF0000'  # Red
    elif pm25 < 250:
        return '#8F3F97'  # Purple
    else:
        return '#7E0023'  # Maroon

def display_forecast_results(results_df):
    """Display forecast results with visualization"""
    st.success("Forecast generated successfully!")
    st.markdown("### 24-Hour PM2.5 Forecast")
    
    # Create a copy to avoid modifying the original
    results_display = results_df.copy()
    
    # Add AQI color column if it doesn't exist
    if 'aqi_color' not in results_display.columns:
        results_display['aqi_color'] = results_display['prediction'].apply(get_aqi_color)
    
    # Plot with AQI coloring
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_display['timestamp'],
        y=results_display['prediction'],
        name='Predicted PM2.5',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=results_display['timestamp'],
        y=results_display['upper_bound'],
        name='Upper Bound',
        line=dict(color='gray', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=results_display['timestamp'],
        y=results_display['lower_bound'],
        name='Lower Bound',
        line=dict(color='gray', dash='dash')
    ))
    
    # Add AQI color bands
    aqi_thresholds = [0, 12, 35, 55, 150, 250, 500]
    aqi_colors = ['green', 'blue', 'orange', 'red', 'purple', 'maroon']
    
    for i in range(len(aqi_thresholds)-1):
        fig.add_hrect(
            y0=aqi_thresholds[i],
            y1=aqi_thresholds[i+1],
            fillcolor=aqi_colors[i],
            opacity=0.2,
            layer="below",
            line_width=0
        )
    
    fig.update_layout(
        title='PM2.5 Forecast with Confidence Intervals',
        xaxis_title='Time',
        yaxis_title='PM2.5 (µg/m³)',
        hovermode="x unified"
    )
    st.plotly_chart(fig)
    
    # Data table with formatting
    st.markdown("### Detailed Forecast Data")
    st.dataframe(
        results_display[['timestamp', 'prediction', 'aqi_category']].style.apply(
            lambda row: [f'background-color: {get_aqi_color(row["prediction"])}'] * len(row), 
            axis=1
        )
    )
    
    # Add download button
    csv = results_display.to_csv(index=False)
    st.download_button(
        label="Download Forecast Data",
        data=csv,
        file_name=f"pm25_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
