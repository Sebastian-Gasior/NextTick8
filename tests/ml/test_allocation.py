from src.ml.allocation import equal_weight_allocation, risk_parity_allocation
import pandas as pd
import numpy as np

def test_equal_weight_allocation() -> None:
    """Testet die Gleichgewichtung."""
    tickers = ["A", "B", "C"]
    weights = equal_weight_allocation(tickers)
    assert all(abs(w - 1/3) < 1e-6 for w in weights.values())

def test_risk_parity_allocation() -> None:
    """Testet die risikobasierte Gewichtung."""
    returns = pd.DataFrame({"A": np.random.normal(0, 0.01, 100), "B": np.random.normal(0, 0.02, 100)})
    weights = risk_parity_allocation(returns)
    assert set(weights.keys()) == set(["A", "B"])
    assert abs(sum(weights.values()) - 1) < 1e-6 