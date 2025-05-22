import pandas as pd
import numpy as np
from src.ml.feature_engineering import add_technical_indicators

def test_add_technical_indicators() -> None:
    """Testet das Hinzufügen technischer Indikatoren (SMA, EMA, MACD, Bollinger, RSI)."""
    df = pd.DataFrame({"Close": np.arange(1, 21, dtype=float)})
    df = add_technical_indicators(df)
    # Prüfe auf alle neuen Features
    assert "sma_10" in df.columns
    assert "sma_50" in df.columns
    assert "ema_10" in df.columns
    assert "ema_50" in df.columns
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    assert "bollinger_upper" in df.columns
    assert "bollinger_lower" in df.columns
    assert "rsi_14" in df.columns
    # Werte sollten nicht komplett NaN sein
    assert df["sma_10"].notna().sum() > 0
    assert df["ema_10"].notna().sum() > 0
    assert df["macd"].notna().sum() > 0
    assert df["bollinger_upper"].notna().sum() > 0
    assert df["rsi_14"].notna().sum() > 0 