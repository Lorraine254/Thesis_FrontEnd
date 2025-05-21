import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st
import pandas as pd



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
    st.success("Forecast generated successfully!")

    # Show data source information
    if 'data_source' in results_df.columns:
        source = results_df['data_source'].iloc[0]
        if source == 'Simulated Data':
            st.warning("""
            **Note:** Using simulated weather data because:
            - Historical API only covers last 5 days
            - Try recent dates for real weather data
            """)
        else:
            st.info(f"Using real weather data from {source}")

    # Create two tabs for different views
    tab1, tab2 = st.tabs(["📈 Interactive Chart", "📋 Data Table"])
    
    with tab1:
        fig = go.Figure()
        
        # Confidence interval band (shown first so it's behind)
        fig.add_trace(go.Scatter(
            x=results_df['timestamp'],
            y=results_df['upper_95'],
            name='95th Percentile',
            line=dict(width=0),
            fillcolor='rgba(0, 100, 255, 0.2)',
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=results_df['timestamp'],
            y=results_df['lower_05'],
            name='5th Percentile',
            line=dict(width=0),
            showlegend=False
        ))
        
        # Main prediction line
        fig.add_trace(go.Scatter(
            x=results_df['timestamp'],
            y=results_df['prediction'],
            name='Predicted PM2.5',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='PM2.5 Forecast with 90% Confidence Interval',
            xaxis_title='Time',
            yaxis_title='PM2.5 (µg/m³)',
            hovermode="x unified",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Forecast Data with Confidence Intervals")
        
        # Format the dataframe for display
        display_df = results_df.copy()
        display_df['Prediction'] = display_df['prediction'].round(2)
        display_df['95% Upper'] = display_df['upper_95'].round(2)
        display_df['5% Lower'] = display_df['lower_05'].round(2)
        display_df['AQI Category'] = display_df['aqi_category']
        
        # Create a style function with more subtle highlighting
        def highlight_aqi(row):
            color = get_aqi_color(row['Prediction'])
            # Lighten the color by adding opacity
            light_color = color + '33'  # Adds 20% opacity (33 in hex)
            return [f'background-color: {light_color}' if col == 'AQI Category' else '' for col in row.index]
        
        # Apply styling only to the AQI Category column
        st.dataframe(
            display_df[['timestamp', 'Prediction', '5% Lower', '95% Upper', 'AQI Category']].style.apply(
                highlight_aqi, 
                axis=1
            ),
            height=600
        )