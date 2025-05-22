"""
plots.py
Visualisierung von Equity-Kurven und Prognosen.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
import plotly.graph_objs as go

def plot_equity_curve(equity: np.ndarray, title: str = "Equity Curve", save_path: Optional[str] = None) -> plt.Figure:
    """Plottet die Equity-Kurve und speichert sie optional als PNG. Gibt die Figure zurück."""
    plt.figure(figsize=(10, 4))
    plt.plot(equity, label="Equity")
    plt.title(title)
    plt.xlabel("Zeit")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return plt.gcf()

def plot_predictions(true: np.ndarray, preds: np.ndarray, title: str = "Prognose", save_path: Optional[str] = None) -> plt.Figure:
    """Plottet echte Werte und Prognosen. Gibt die Figure zurück."""
    plt.figure(figsize=(10, 4))
    plt.plot(true, label="Echt")
    plt.plot(preds, label="Prognose")
    plt.title(title)
    plt.xlabel("Zeit")
    plt.ylabel("Wert")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return plt.gcf()

def plot_ml_prediction(true_vals, preds, index=None):
    """Plotly-Chart für ML-Vorhersage vs. tatsächliche Werte."""
    fig = go.Figure()
    if index is not None:
        fig.add_trace(go.Scatter(x=index, y=true_vals, mode="lines", name="Actual", line=dict(color="#a020f0")))
        fig.add_trace(go.Scatter(x=index, y=preds, mode="lines", name="Prediction", line=dict(color="red", dash="dash")))
    else:
        fig.add_trace(go.Scatter(y=true_vals, mode="lines", name="Actual", line=dict(color="#a020f0")))
        fig.add_trace(go.Scatter(y=preds, mode="lines", name="Prediction", line=dict(color="red", dash="dash")))
    fig.update_layout(title="ML Vorhersagen", height=400)
    return fig

def plot_strategy_comparison(equity_ta: np.ndarray, equity_ml: np.ndarray, drawdown_ta: np.ndarray, drawdown_ml: np.ndarray, index=None) -> go.Figure:
    """Plotly-Strategie-Vergleich: Equity und Drawdown für TA und ML."""
    if index is None:
        index = list(range(len(equity_ta)))
    fig = go.Figure()
    # Equity-Kurven
    fig.add_trace(go.Scatter(x=index, y=equity_ta, name="TA Strategy", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=index, y=equity_ml, name="ML Strategy", line=dict(color="red")))
    # Drawdown-Kurven (zweites Subplot)
    fig.add_trace(go.Scatter(x=index, y=drawdown_ta, name="TA Drawdown", line=dict(color="blue", dash="dot"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=index, y=drawdown_ml, name="ML Drawdown", line=dict(color="red", dash="dot"), yaxis="y2"))
    fig.update_layout(
        title="Strategie-Vergleich",
        height=600,
        yaxis=dict(title="Equity", side="left"),
        yaxis2=dict(title="Drawdown", overlaying="y", side="right", anchor="x", showgrid=False),
        legend=dict(orientation="h")
    )
    return fig

def plot_candlestick_volume(df: pd.DataFrame) -> go.Figure:
    """Plotly-Candlestick- und Volumen-Plot wie in den Beispielbildern."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', marker_color='rgba(255,100,100,0.3)'))
    fig.update_layout(
        title='Preis & Volumen',
        yaxis=dict(title='Preis'),
        yaxis2=dict(title='Volumen', overlaying='y', side='right', showgrid=False),
        height=600,
        legend=dict(orientation='h')
    )
    return fig 