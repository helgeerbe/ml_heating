"""
Shared plotting utilities for consistent visualization across notebooks.
"""

import plotly.graph_objects as go
import pandas as pd
from typing import Optional, List, Dict, Any


def plot_prediction_vs_actual(
    df: pd.DataFrame,
    actual_col: str = "actual_outlet_temp",
    predicted_col: str = "predicted_outlet_temp",
    title: str = "Model Prediction vs Actual",
    height: int = 600
) -> go.Figure:
    """
    Create a standard interactive plot for model evaluation.
    
    Args:
        df: DataFrame containing the data
        actual_col: Name of the actual value column
        predicted_col: Name of the predicted value column
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add Actual
    if actual_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[actual_col],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
    # Add Predicted
    if predicted_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df[predicted_col],
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=height,
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig
