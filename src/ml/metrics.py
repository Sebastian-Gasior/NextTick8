"""
metrics.py
Berechnung von Backtest-Metriken (Sharpe, Drawdown, Hit Ratio).
"""
import numpy as np
import pandas as pd

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Berechnet die annualisierte Sharpe Ratio."""
    excess = returns - risk_free_rate
    return np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252)

def max_drawdown(equity: np.ndarray) -> float:
    """Berechnet den maximalen Drawdown."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return np.min(drawdown)

def hit_ratio(preds: np.ndarray, targets: np.ndarray) -> float:
    """Berechnet die Trefferquote (Hit Ratio) von Vorhersagen."""
    return np.mean(np.sign(preds) == np.sign(targets)) 