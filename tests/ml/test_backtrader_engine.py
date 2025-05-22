import pandas as pd
import numpy as np
from src.ml.backtrader_engine import run_backtest

def test_run_backtest() -> None:
    """Testet das Backtesting mit Dummy-Daten und Signalen."""
    df = pd.DataFrame({"Close": np.linspace(100, 120, 30)}, index=pd.date_range("2022-01-01", periods=30))
    signals = np.random.choice([0, 1], size=len(df))
    report = run_backtest(df, signals, strategy_name="test", commission=0.001, slippage=0.001, size=1, report_dir="backtest/reports/test")
    assert isinstance(report, pd.DataFrame)
    assert "equity" in report.columns
    # Prüfe, ob CSV existiert
    import os
    csv_path = "backtest/reports/test/test_backtest.csv"
    assert os.path.exists(csv_path) 