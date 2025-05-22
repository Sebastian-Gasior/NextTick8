import os
import pandas as pd
import numpy as np
import pytest
from src import data_validation

def test_remove_duplicates() -> None:
    """Testet das Entfernen von Duplikaten im Index."""
    idx = pd.date_range("2020-01-01", periods=3).tolist() * 2
    df = pd.DataFrame({"Close": range(6)}, index=idx)
    cleaned, log = data_validation.validate_and_clean(df)
    assert cleaned.index.is_unique
    assert any("Duplikate entfernt" in l for l in log)

def test_fill_nans() -> None:
    """Testet das Entfernen von Zeilen mit NaN-Werten."""
    idx = pd.date_range("2020-01-01", periods=5)
    df = pd.DataFrame({"Close": [1, np.nan, np.nan, 4, 5]}, index=idx)
    cleaned, log = data_validation.validate_and_clean(df)
    # Es dürfen keine NaN mehr enthalten sein
    assert not cleaned.isna().any().any()
    # Es sollten nur die Zeilen ohne NaN übrig bleiben
    assert len(cleaned) == 3
    assert all(~cleaned["Close"].isna())

def test_process_all_raw(tmp_path) -> None:
    """Testet die Gesamtverarbeitung von Rohdaten zu bereinigten Daten."""
    raw_dir = tmp_path / "raw"
    cleaned_dir = tmp_path / "cleaned"
    os.makedirs(raw_dir)
    df = pd.DataFrame({"Close": [1, np.nan, 3]}, index=pd.date_range("2020-01-01", periods=3))
    df.to_csv(raw_dir / "TEST.csv")
    data_validation.process_all_raw(str(raw_dir), str(cleaned_dir))
    out = pd.read_csv(cleaned_dir / "TEST.csv", index_col=0, parse_dates=True)
    assert not out.isna().any().any()

def test_remove_repeated_tail() -> None:
    """Testet das Entfernen identischer Endzeilen durch remove_repeated_tail."""
    idx = pd.date_range("2020-01-01", periods=10)
    # Die letzten 4 Zeilen sind identisch
    data = [1,2,3,4,5,6,7,8,8,8,8][:10]
    df = pd.DataFrame({"Close": data}, index=idx)
    cleaned, _ = data_validation.validate_and_clean(df)
    # Es sollten 4 Zeilen entfernt werden, da die letzten 4 identisch sind
    assert len(cleaned) == 6
    # Die letzten Werte dürfen nicht mehr identisch sein
    assert not (cleaned["Close"].tail(4) == 8).all() 