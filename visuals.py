import plotly.express as px
import pandas as pd
import numpy as np



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
