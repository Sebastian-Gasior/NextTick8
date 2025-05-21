import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks


def smooth_series(
    series: pd.Series,
    method: str = "savgol",
    window: int = 7,
    polyorder: int = 3
) -> pd.Series:
    """
    Glättet eine Zeitreihe mit Moving Average oder Savitzky-Golay.
    """
    if method == "savgol":
        if window % 2 == 0:
            window += 1  # Savitzky-Golay benötigt ungerade Fenstergröße
        smooth = savgol_filter(series.values, window_length=window, polyorder=polyorder, mode="interp")
        return pd.Series(smooth, index=series.index)
    elif method == "ma":
        return series.rolling(window=window, min_periods=1, center=True).mean()
    else:
        raise ValueError("Unbekannte Glättungsmethode")


def detect_peaks_and_troughs(
    series: pd.Series,
    prominence: float = 0.01,
    distance: int = 5,
    include_edges: bool = True
) -> Dict[str, Any]:
    """
    Findet Peaks (Maxima) und Troughs (Minima) in einer Zeitreihe.
    Optional: Markiert auch Ränder als Peak/Trough, wenn sie Extremwerte sind.
    """
    arr = series.values
    peaks, peak_props = find_peaks(arr, prominence=prominence, distance=distance)
    troughs, trough_props = find_peaks(-arr, prominence=prominence, distance=distance)
    # Ränder prüfen
    if include_edges:
        if len(arr) > 2:
            if arr[0] > arr[1] and arr[0] > arr[2]:
                peaks = np.insert(peaks, 0, 0)
            if arr[0] < arr[1] and arr[0] < arr[2]:
                troughs = np.insert(troughs, 0, 0)
            if arr[-1] > arr[-2] and arr[-1] > arr[-3]:
                peaks = np.append(peaks, len(arr)-1)
            if arr[-1] < arr[-2] and arr[-1] < arr[-3]:
                troughs = np.append(troughs, len(arr)-1)
    return {
        "peaks": peaks,
        "troughs": troughs,
        "peak_props": peak_props,
        "trough_props": trough_props
    }


def process_cleaned_data(
    cleaned_dir: str,
    out_dir: str,
    column: str = "Close",
    smoothing: str = "savgol",
    window: int = 7,
    polyorder: int = 3,
    prominence: float = 0.01,
    distance: int = 5
) -> None:
    """
    Lädt alle bereinigten Daten, glättet sie und erkennt Peaks/Troughs. Ergebnisse werden als CSV gespeichert.
    """
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(cleaned_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(cleaned_dir, file), index_col=0, parse_dates=True)
            if column not in df.columns:
                continue
            smooth = smooth_series(df[column], method=smoothing, window=window, polyorder=polyorder)
            result = detect_peaks_and_troughs(smooth, prominence=prominence, distance=distance, include_edges=True)
            out = df.copy()
            out["Smooth"] = smooth
            out["Peak"] = 0
            out["Trough"] = 0
            out.iloc[result["peaks"], out.columns.get_loc("Peak")] = 1
            out.iloc[result["troughs"], out.columns.get_loc("Trough")] = 1
            out.to_csv(os.path.join(out_dir, file)) 