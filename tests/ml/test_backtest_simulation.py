import pandas as pd
import numpy as np
from src.ml.backtest_simulation import simple_backtest

def test_simple_backtest() -> None:
    """Testet das Backtesting mit zufälligen Signalen."""
    df = pd.DataFrame({"Close": np.linspace(100, 120, 21)})
    signals = np.random.choice([0, 1], size=len(df))
    result = simple_backtest(df, signals)
    assert "equity" in result.columns
    assert result["equity"].iloc[0] == 1.0 