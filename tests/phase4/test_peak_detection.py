import numpy as np
import pandas as pd
from src import peak_detection

def test_smooth_series_ma() -> None:
    """Testet Moving Average Glättung."""
    s = pd.Series([1, 2, 3, 4, 5, 6, 7])
    smooth = peak_detection.smooth_series(s, method="ma", window=3)
    assert np.allclose(smooth.values[3], 4)

def test_smooth_series_savgol() -> None:
    """Testet Savitzky-Golay Glättung."""
    s = pd.Series(np.arange(10) + np.random.normal(0, 0.1, 10))
    smooth = peak_detection.smooth_series(s, method="savgol", window=5, polyorder=2)
    assert len(smooth) == len(s)

def test_detect_peaks_and_troughs() -> None:
    """Testet die Erkennung von Peaks und Troughs."""
    s = pd.Series([0, 1, 0, -1, 0, 1, 0, -1, 0])
    result = peak_detection.detect_peaks_and_troughs(s, prominence=0.5, distance=1)
    # Es sollten 2 Peaks und 2 Troughs erkannt werden
    assert len(result["peaks"]) == 2
    assert len(result["troughs"]) == 2

def test_process_cleaned_data(tmp_path) -> None:
    """Testet die End-to-End-Verarbeitung von bereinigten Daten."""
    # Testdaten mit zwei klaren Peaks und zwei klaren Troughs
    df = pd.DataFrame({"Close": [0, 2, 0, -2, 0, 2, 0, -2, 0]}, index=pd.date_range("2020-01-01", periods=9))
    cleaned_dir = tmp_path / "cleaned"
    out_dir = tmp_path / "peaks"
    cleaned_dir.mkdir()
    df.to_csv(cleaned_dir / "TEST.csv")
    peak_detection.process_cleaned_data(str(cleaned_dir), str(out_dir), column="Close", smoothing="ma", window=3, prominence=1.0, distance=1)
    out = pd.read_csv(out_dir / "TEST.csv", index_col=0, parse_dates=True)
    assert "Peak" in out.columns and "Trough" in out.columns
    assert out["Peak"].sum() == 2
    assert out["Trough"].sum() == 2 