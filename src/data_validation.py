import os
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import logging


def load_raw_data(raw_dir: str) -> List[Tuple[str, pd.DataFrame]]:
    """Lädt alle CSV-Dateien aus dem Rohdatenordner und gibt eine Liste von (Ticker, DataFrame) zurück.
    Entfernt automatisch fehlerhafte Header-/Tickerzeilen und alle Zeilen, in denen numerische Spalten nicht numerisch sind."""
    data = []
    print(f"[DEBUG] Lade Daten aus Verzeichnis: {raw_dir}")
    for file in os.listdir(raw_dir):
        if file.endswith('.csv'):
            print(f"[DEBUG] Verarbeite Datei: {file}")
            ticker = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(raw_dir, file), index_col=0)
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
            print(f"[DEBUG] Ursprüngliche Zeilen: {len(df)}")
            # Entferne alle Zeilen, deren Index kein Datum ist
            df = df[pd.to_datetime(df.index, errors='coerce').notna()]
            print(f"[DEBUG] Nach Index-Filter: {len(df)}")
            # Versuche, alle numerischen Spalten zu erzwingen
            for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Price']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Entferne alle Zeilen, in denen eine der numerischen Spalten nicht numerisch ist
            num_cols = [col for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Price'] if col in df.columns]
            df = df.dropna(subset=num_cols, how='any')
            print(f"[DEBUG] Nach numerischem Filter: {len(df)}")
            data.append((ticker, df))
    print(f"[DEBUG] Fertig mit Laden aller Dateien.")
    return data


def validate_and_clean(
    df: pd.DataFrame,
    method_outlier: str = 'zscore',
    z_thresh: float = 3.0,
    iqr_factor: float = 1.5,
    hampel_window: int = 7
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prüft und bereinigt einen DataFrame:
    - Entfernt Duplikate
    - Interpoliert/füllt NaN
    - Filtert Ausreißer (z-Score, IQR, Hampel)
    Gibt den bereinigten DataFrame und eine Liste der angewandten Korrekturen zurück.
    """
    log = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Duplikate
    before = len(df)
    df = df[~df.index.duplicated(keep='first')]
    after = len(df)
    if after < before:
        log.append(f"{before-after} Duplikate entfernt.")
    # Fehlende Werte
    nans = df.isna().sum().sum()
    if nans > 0:
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
        log.append(f"{nans} NaN-Werte interpoliert/gefüllt.")
    # Ausreißer
    if method_outlier == 'zscore':
        for col in df.select_dtypes(include=[np.number]).columns:
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z > z_thresh
            n_out = outliers.sum()
            if n_out > 0:
                df.loc[outliers, col] = np.nan
                log.append(f"{n_out} Ausreißer in {col} (z-Score) entfernt.")
        if df.isna().sum().sum() > 0:
            if len(num_cols) > 0:
                df[num_cols] = df[num_cols].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    elif method_outlier == 'iqr':
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            outliers = (df[col] < lower) | (df[col] > upper)
            n_out = outliers.sum()
            if n_out > 0:
                df.loc[outliers, col] = np.nan
                log.append(f"{n_out} Ausreißer in {col} (IQR) entfernt.")
        if df.isna().sum().sum() > 0:
            if len(num_cols) > 0:
                df[num_cols] = df[num_cols].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    elif method_outlier == 'hampel':
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col]
            rolling_median = series.rolling(window=hampel_window, center=True).median()
            diff = np.abs(series - rolling_median)
            mad = series.rolling(window=hampel_window, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
            threshold = 3 * mad
            outliers = diff > threshold
            n_out = outliers.sum()
            if n_out > 0:
                df.loc[outliers, col] = np.nan
                log.append(f"{n_out} Ausreißer in {col} (Hampel) entfernt.")
        if df.isna().sum().sum() > 0:
            if len(num_cols) > 0:
                df[num_cols] = df[num_cols].interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    return df, log


def process_all_raw(
    raw_dir: str,
    cleaned_dir: str,
    log_path: Optional[str] = None,
    method_outlier: str = 'zscore'
) -> None:
    """
    Lädt alle Rohdaten, bereinigt sie und speichert sie in cleaned_dir. Korrekturen werden geloggt.
    """
    os.makedirs(cleaned_dir, exist_ok=True)
    logger = logging.getLogger("data_validation")
    logger.setLevel(logging.INFO)
    if log_path:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    for ticker, df in load_raw_data(raw_dir):
        cleaned, corrections = validate_and_clean(df, method_outlier=method_outlier)
        cleaned.to_csv(os.path.join(cleaned_dir, f"{ticker}.csv"))
        logger.info(f"{ticker}: {', '.join(corrections) if corrections else 'keine Korrekturen'}")


if __name__ == "__main__":
    process_all_raw("data/raw", "data/cleaned", log_path="data/cleaning.log", method_outlier="zscore")
    print("Validierung & Bereinigung abgeschlossen.") 