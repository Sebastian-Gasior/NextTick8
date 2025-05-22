"""
feature_engineering.py
Feature Engineering für Zeitreihen (z.B. technische Indikatoren).
"""
from typing import Any
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt technische Indikatoren wie gleitende Durchschnitte, RSI, MACD, Bollinger Bands etc. hinzu."""
    df = df.copy()
    df['sma_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['sma_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['rsi_14'] = compute_rsi(df['Close'], window=14)
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['bollinger_upper'] = rolling_mean + (rolling_std * 2)
    df['bollinger_lower'] = rolling_mean - (rolling_std * 2)
    return df

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Berechnet den Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Exportfunktion für Features
def export_features(df: pd.DataFrame, ticker: str, export_dir: str = "data/ta_features") -> None:
    """Speichert den Feature-DataFrame als CSV für den gegebenen Ticker."""
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f"{ticker}_features.csv")
    df.to_csv(path)

# Beispielplots für Features (Matplotlib & Plotly)
def plot_features_matplotlib(df: pd.DataFrame, ticker: str):
    """Erstellt einen Matplotlib-Plot der wichtigsten Features und gibt das Figure-Objekt zurück."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Close"], label="Close")
    ax.plot(df["sma_10"], label="SMA 10")
    ax.plot(df["sma_50"], label="SMA 50")
    ax.plot(df["ema_10"], label="EMA 10")
    ax.plot(df["ema_50"], label="EMA 50")
    ax.fill_between(df.index, df["bollinger_upper"], df["bollinger_lower"], color="gray", alpha=0.2, label="Bollinger Bands")
    ax.set_title(f"Features für {ticker}")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_features_plotly(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Erstellt einen Plotly-Chart für die wichtigsten Features."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_10'], name='SMA 10'))
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_10'], name='EMA 10'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], name='EMA 50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['bollinger_upper'], name='Bollinger Upper', line=dict(dash='dot', color='gray')))
    fig.add_trace(go.Scatter(x=df.index, y=df['bollinger_lower'], name='Bollinger Lower', line=dict(dash='dot', color='gray')))
    fig.update_layout(title=f"Features für {ticker}", height=600)
    return fig 