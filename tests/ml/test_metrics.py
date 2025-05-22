import numpy as np
from src.ml.metrics import sharpe_ratio, max_drawdown, hit_ratio

def test_sharpe_ratio() -> None:
    """Testet die Sharpe Ratio Berechnung."""
    returns = np.random.normal(0.001, 0.01, 100)
    sr = sharpe_ratio(returns)
    assert isinstance(sr, float)

def test_max_drawdown() -> None:
    """Testet die Drawdown-Berechnung."""
    equity = np.linspace(1, 2, 100)
    dd = max_drawdown(equity)
    assert dd <= 0

def test_hit_ratio() -> None:
    """Testet die Hit Ratio Berechnung."""
    preds = np.array([1, -1, 1, -1])
    targets = np.array([1, 1, -1, -1])
    hr = hit_ratio(preds, targets)
    assert 0 <= hr <= 1 