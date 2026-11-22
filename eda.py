"""
eda.py
------
Plotly-based chart builders for the Interactive EDA Dashboard tab.
Each function returns plotly Figure object(s) that Streamlit renders with
st.plotly_chart(); no plotting logic lives in app.py.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_histograms(df: pd.DataFrame, numeric_cols: list) -> dict:
    """
    Build a histogram (with marginal box plot) for every numeric column.

    Returns:
        dict of {column_name: plotly Figure}
    """
    figs = {}
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, marginal="box", title=f"Distribution of {col}")
        fig.update_layout(bargap=0.05, height=350, margin=dict(t=50, b=30))
        figs[col] = fig
    return figs


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    """
    Build a correlation heatmap across numeric columns.

    Returns:
        plotly Figure, or None if fewer than 2 numeric columns exist.
    """
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(title="Correlation Heatmap", height=500, margin=dict(t=50, b=30))
    return fig


def plot_boxplots(df: pd.DataFrame, numeric_cols: list) -> dict:
    """
    Build a box plot per numeric column, useful for spotting outliers.

    Returns:
        dict of {column_name: plotly Figure}
    """
    figs = {}
    for col in numeric_cols:
        fig = px.box(df, y=col, points="outliers", title=f"Box Plot: {col}")
        fig.update_layout(height=350, margin=dict(t=50, b=30))
        figs[col] = fig
    return figs


def plot_target_distribution(df: pd.DataFrame, target_col: str, task_type: str):
    """
    Build a distribution chart for the target column: a bar chart of class
    counts for classification, or a histogram for regression.

    Returns:
        plotly Figure
    """
    if task_type == "classification":
        counts = df[target_col].value_counts().reset_index()
        counts.columns = [target_col, "Count"]
        fig = px.bar(counts, x=target_col, y="Count", title=f"Target Distribution: {target_col}",
                     color=target_col)
    else:
        fig = px.histogram(df, x=target_col, nbins=30, marginal="box",
                            title=f"Target Distribution: {target_col}")
    fig.update_layout(height=400, margin=dict(t=50, b=30))
    return fig
