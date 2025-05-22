import pandas as pd
import numpy as np
from src.ml.feature_engineering import add_technical_indicators
from src.ml.model_training import train_lstm_with_val, predict_lstm
from src.ml.backtest_simulation import simple_backtest
from src.ml.metrics import sharpe_ratio, max_drawdown, hit_ratio
from src.ml.plots import plot_equity_curve

def test_e2e_ml_pipeline() -> None:
    """End-to-End-Test für die ML-Pipeline (Feature Engineering, LSTM, Backtest, Metriken, Plot)."""
    # Dummy-Daten
    df = pd.DataFrame({"Close": np.sin(np.linspace(0, 10, 120)) + 10})
    df = add_technical_indicators(df)
    model, scaler, log = train_lstm_with_val(df, window=10, epochs=2, dropout=0.1)
    assert model is not None
    assert scaler is not None
    assert 'train_loss' in log and 'val_loss' in log
    close_scaled = scaler.transform(df["Close"].values.reshape(-1, 1)).flatten()
    preds = predict_lstm(model, scaler, close_scaled, window=10, steps=10)
    # Backtest: Signal = 1, wenn Prognose > letzter Wert, sonst 0
    last_val = close_scaled[-1]
    signals = np.array([1 if p > last_val else 0 for p in preds])
    result = simple_backtest(df.iloc[-10:], signals)
    # Metriken
    sr = sharpe_ratio(result["strategy"].fillna(0).values)
    dd = max_drawdown(result["equity"].values)
    hr = hit_ratio(preds, df["Close"].iloc[-10:].values)
    # Plot
    plot_equity_curve(result["equity"].values, title="E2E Equity Curve")
    assert isinstance(sr, float)
    assert isinstance(dd, float)
    assert 0 <= hr <= 1 