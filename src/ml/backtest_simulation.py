"""
backtest_simulation.py
Backtesting und Simulation von ML-Signalen auf Zeitreihen.
"""
from typing import Tuple
import pandas as pd
import numpy as np

def simple_backtest(df: pd.DataFrame, signals: np.ndarray) -> pd.DataFrame:
    """Simuliert eine einfache Strategie basierend auf ML-Signalen (long/flat)."""
    df = df.copy()
    df['signal'] = signals
    df['returns'] = df['Close'].pct_change().shift(-1)
    df['strategy'] = df['returns'] * df['signal']
    df['equity'] = (1 + df['strategy'].fillna(0)).cumprod()
    return df 