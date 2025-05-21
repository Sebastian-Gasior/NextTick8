import os
import pandas as pd
import numpy as np
import pytest
from src import data_validation

def test_remove_duplicates() -> None:
    """Testet das Entfernen von Duplikaten im Index."""
    idx = pd.date_range("2020-01-01", periods=3).tolist() * 2
    df = pd.DataFrame({"A": range(6)}, index=idx)
    cleaned, log = data_validation.validate_and_clean(df)
    assert cleaned.index.is_unique
    assert any("Duplikate entfernt" in l for l in log)

def test_fill_nans() -> None:
    """Testet das Interpolieren und Füllen von NaN-Werten."""
    idx = pd.date_range("2020-01-01", periods=5)
    df = pd.DataFrame({"A": [1, np.nan, np.nan, 4, 5]}, index=idx)
    cleaned, log = data_validation.validate_and_clean(df)
    assert not cleaned.isna().any().any()
    assert any("NaN-Werte interpoliert" in l for l in log)

def test_zscore_outlier_removal() -> None:
    """Testet das Entfernen von Ausreißern mit z-Score an realistischeren Daten."""
    idx = pd.date_range("2020-01-01", periods=100)
    vals = [1]*49 + [100] + [1]*50
    df = pd.DataFrame({"A": vals}, index=idx)
    cleaned, log = data_validation.validate_and_clean(df, method_outlier="zscore", z_thresh=3.0)
    assert cleaned.loc[idx[49], "A"] != 100
    assert any("z-Score" in l for l in log)

def test_iqr_outlier_removal() -> None:
    """Testet das Entfernen von Ausreißern mit IQR."""
    idx = pd.date_range("2020-01-01", periods=10)
    vals = [1, 1, 1, 1, 100, 1, 1, 1, 1, 1]
    df = pd.DataFrame({"A": vals}, index=idx)
    cleaned, log = data_validation.validate_and_clean(df, method_outlier="iqr", iqr_factor=1.5)
    assert cleaned.loc[idx[4], "A"] != 100
    assert any("IQR" in l for l in log)

def test_hampel_outlier_removal() -> None:
    """Testet das Entfernen von Ausreißern mit Hampel-Filter."""
    idx = pd.date_range("2020-01-01", periods=10)
    vals = [1, 1, 1, 1, 100, 1, 1, 1, 1, 1]
    df = pd.DataFrame({"A": vals}, index=idx)
    cleaned, log = data_validation.validate_and_clean(df, method_outlier="hampel", hampel_window=3)
    assert cleaned.loc[idx[4], "A"] != 100
    assert any("Hampel" in l for l in log)

def test_process_all_raw(tmp_path) -> None:
    """Testet die Gesamtverarbeitung von Rohdaten zu bereinigten Daten."""
    raw_dir = tmp_path / "raw"
    cleaned_dir = tmp_path / "cleaned"
    os.makedirs(raw_dir)
    df = pd.DataFrame({"A": [1, np.nan, 3]}, index=pd.date_range("2020-01-01", periods=3))
    df.to_csv(raw_dir / "TEST.csv")
    data_validation.process_all_raw(str(raw_dir), str(cleaned_dir))
    out = pd.read_csv(cleaned_dir / "TEST.csv", index_col=0, parse_dates=True)
    assert not out.isna().any().any() 