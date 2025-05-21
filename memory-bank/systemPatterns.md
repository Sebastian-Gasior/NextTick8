# systemPatterns.md

## Systemarchitektur
- Input: stocks.txt (editierbare Tickerliste)
- Datenabruf & Speicherung: Yahoo Finance → CSV in data/raw/
- Datenprüfung/-bereinigung: data/cleaned/
- Analyse: Peaks/Troughs, Wellenzählung nach Heuristik
- Visualisierung: Streamlit (ein File)
- Export: HTML, PNG Download
- Tests: In tests/phaseX/

## Wichtige Entscheidungen
- Daten werden immer zuerst gespeichert, dann geprüft und bereinigt
- Jede Funktionseinheit einzeln testbar und dokumentiert
- Datenintegrität hat höchste Priorität

## Design Patterns
- Single Responsibility für jede Funktion
- Modularisierung: Download → Clean → Analyse → Visualisierung → Export
- Jede Phase ist eigenständig testbar

## Komponentenbeziehungen
- stocks.txt ↔ Datenimport
- data/raw/ → Datenvalidierung → data/cleaned/ → Analyse/Visualisierung
- tests/phaseX/ ↔ jeweilige Hauptfunktion

### Ordnerstruktur
elliott-wave-analyzer/
├── app.py
├── README.md
├── memory-bank/
│ ├── projectbrief.md
│ ├── productContext.md
│ ├── activeContext.md
│ ├── systemPatterns.md
│ ├── techContext.md
│ └── progress.md
├── data/
│ ├── raw/
│ └── cleaned/
├── stocks.txt
├── tests/
│ ├── phase1/
│ ├── phase2/
│ └── ...

### Architekturprinzipien

- Jedes Modul hat Single Responsibility
- Daten-Import, -Prüfung, -Bereinigung, Peak-Detection und Wellenanalyse sind klar getrennt und testbar
- Tests werden nach Phasen in `tests/phaseX/` abgelegt
- Zentrale Code-Snippets und Patterns für alle Kernaufgaben

---

## Code Snippets & Patterns

### 1. Datenprüfung und Bereinigung

```python
import pandas as pd
import numpy as np

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bereinigt Finanz-Zeitreihendaten: Duplikate, Ausreißer, Lücken etc."""
    # Duplikate anhand des Zeitstempels entfernen und nach Datum sortieren
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').drop_duplicates(subset='Date')
        df.set_index('Date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df[df.index.notna()]
    # Ausreißerbehandlung – IQR-Methode
    q1 = df['Close'].quantile(0.25)
    q3 = df['Close'].quantile(0.75)
    IQR = q3 - q1
    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR
    outliers = (df['Close'] < lower_bound) | (df['Close'] > upper_bound)
    df.loc[outliers, 'Close'] = np.nan
    # Fehlende Werte interpolieren
    df['Close'] = df['Close'].interpolate(method='time')
    df['Close'].fillna(method='bfill', inplace=True)
    df['Close'].fillna(method='ffill', inplace=True)
    # Leichte Glättung (Moving Average)
    df['Close_smooth'] = df['Close'].rolling(window=3, min_periods=1, center=True).mean()
    return df
``` python Datenprüfung und Bereinigung Ende

### 2. Glättung und Peak-Erkennung

import numpy as np
from scipy.signal import savgol_filter, find_peaks

def smooth_and_find_peaks(series: np.ndarray, use_savgol: bool = True) -> dict:
    """
    Glättet eine Zeitreihe und findet Peaks (Maxima) sowie Valleys (Minima).
    Rückgabe: Dict mit Indizes der Peaks/Valleys und geglätteter Serie.
    """
    if use_savgol:
        smooth = savgol_filter(series, window_length=7, polyorder=3)
    else:
        smooth = np.convolve(series, np.ones(7)/7, mode='same')
    peak_indices, _ = find_peaks(smooth, distance=5, prominence=np.std(smooth)*0.5)
    valley_indices, _ = find_peaks(-smooth, distance=5, prominence=np.std(smooth)*0.5)
    return {
        "smooth": smooth,
        "peaks": peak_indices,
        "valleys": valley_indices
    }

### 3. Elliott-Wellen-Zählung (Heuristik)

def identify_impulse_wave(turning_points: list) -> list:
    """
    Identifiziert eine 5-teilige Impulswelle in den gegebenen Wendepunkten.
    turning_points: Liste von (index, price)-Tupeln alternierender Hoch-/Tiefpunkte.
    Rückgabe: Liste [P0, P1, P2, P3, P4, P5] der Punkte, die einen gültigen Impuls bilden,
             oder [] wenn kein gültiges Muster gefunden wurde.
    """
    if len(turning_points) < 6:
        return []
    P0, P1, P2, P3, P4, P5 = turning_points[:6]
    start_price = P0[1]; w1_price = P1[1]; w2_price = P2[1]
    w3_price = P3[1]; w4_price = P4[1]; w5_price = P5[1]
    uptrend = w1_price > start_price
    if uptrend:
        if w2_price <= start_price:
            return []
        if w4_price <= w1_price:
            return []
        len1 = w1_price - start_price
        len3 = w3_price - w2_price
        len5 = w5_price - w4_price
        if len3 < 0 or len5 < 0:
            return []
        if len3 < len1 and len3 < len5:
            return []
    else:
        if w2_price >= start_price:
            return []
        if w4_price >= w1_price:
            return []
        len1 = start_price - w1_price
        len3 = w2_price - w3_price
        len5 = w4_price - w5_price
        if len3 < 0 or len5 < 0:
            return []
        if len3 < len1 and len3 < len5:
            return []
    return [P0, P1, P2, P3, P4, P5]



