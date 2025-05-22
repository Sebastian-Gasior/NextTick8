"""
ta.py
Technische Analyse: MA, EMA, RSI, Plotly-Visualisierung
"""
import pandas as pd
import numpy as np
import plotly.graph_objs as go

def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt dem DataFrame die Spalten SMA_20 und SMA_50 hinzu."""
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    return df

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Berechnet den RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_ta_signals(df: pd.DataFrame) -> go.Figure:
    """Plotly-Visualisierung: Close, MA, EMA, RSI.
    Args:
        df (pd.DataFrame): DataFrame mit TA-Indikatoren
    Returns:
        go.Figure: Plotly-Figur für die TA-Signale
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='#a020f0')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA_20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA_50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA_200'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_12'], name='EMA_12'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_26'], name='EMA_26'))
    fig.update_layout(title='Technische Analyse', height=500)
    return fig

def plot_rsi(df: pd.DataFrame) -> go.Figure:
    """Plotly-Visualisierung: RSI mit Schwellen.
    Args:
        df (pd.DataFrame): DataFrame mit RSI-Spalte
    Returns:
        go.Figure: Plotly-Figur für den RSI
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='magenta')))
    fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=70, y1=70, line=dict(color='red', dash='dash'))
    fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=30, y1=30, line=dict(color='green', dash='dash'))
    fig.update_layout(title='RSI', height=400, yaxis=dict(range=[20, 80]))
    return fig 