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



# systemPatterns.md

## ML & Zeitreihen-Integration

### Verzeichnisstruktur-Ergänzung (ML)
src/
 └── ml/
      ├── time_split.py
      ├── feature_engineering.py
      ├── model_training.py
      ├── backtest_simulation.py
      ├── metrics.py
      ├── plots.py
      ├── allocation.py
      └── long_term_simulation.py
tests/
 └── ml/
      ├── test_time_split.py
      ├── test_feature_engineering.py
      ├── test_model_training.py
      ├── test_backtest_simulation.py
      ├── test_metrics.py
      ├── test_plots.py
      ├── test_allocation.py
      ├── test_long_term_simulation.py
      └── test_e2e.py

### Architekturentscheidungen (ML)
- Echtdatenpflicht (data/cleaned/)
- Strikte Zeittrennung: kein Test-Leak in Training
- Modular: jeder Schritt eigener Test, eigene Datei
- Modelle, Scaler & Ergebnisse werden gespeichert

---

### Beispiel-Code: Zeitliche Splits (Walk-Forward)

```python
import pandas as pd
import numpy as np
from typing import Tuple, List

def split_fixed(df: pd.DataFrame, train_end: pd.Timestamp, sim_start: pd.Timestamp, sim_end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[df.index <= train_end]
    df_sim = df[(df.index >= sim_start) & (df.index <= sim_end)]
    return df_train, df_sim

def get_walk_forward_splits(df: pd.DataFrame, n_splits: int, test_size: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(df))
    test_len = int(len(df) * test_size)
    splits = []
    for i in range(n_splits):
        train_end = test_len + i * test_len
        train_idx = indices[:train_end]
        test_idx = indices[train_end:train_end + test_len]
        if len(test_idx) == 0: break
        splits.append((train_idx, test_idx))
    return splits


# Feature Engineering

import pandas as pd

def compute_momentum(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    for w in windows:
        df[f'momentum_{w}'] = df['Close'].pct_change(periods=w)
    return df

def compute_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df[f'volatility_{window}'] = df['Close'].rolling(window).std()
    return df

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df['Close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    df[f'rsi_{period}'] = 100 - 100 / (1 + rs)
    return df


# LSTM-Training (Minimalbeispiel)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

def train_lstm_model(close_prices, window=30, epochs=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices.reshape(-1,1))
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.expand_dims(X, axis=2)
    model = Sequential([
        LSTM(64, input_shape=(window, 1)),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=1)
    model.save("ml_model_lstm.h5")
    joblib.dump(scaler, "ml_model_scaler.gz")
    return model, scaler

# Backtest-Simulation

import pandas as pd

class SimpleBacktest:
    def __init__(self, df_sim: pd.DataFrame, model, scaler, window, start_capital: float):
        self.df = df_sim
        self.model = model
        self.scaler = scaler
        self.window = window
        self.capital = start_capital

    def run(self):
        navs = []
        capital = self.capital
        close = self.df['Close'].values
        for i in range(self.window, len(close)):
            X = self.scaler.transform(close[i-self.window:i].reshape(-1,1)).reshape(1, self.window, 1)
            pred = self.model.predict(X)[0,0]
            pred_price = self.scaler.inverse_transform([[pred]])[0,0]
            signal = 1 if pred_price > close[i-1] else -1
            ret = (close[i] - close[i-1]) / close[i-1] * signal
            capital *= (1 + ret)
            navs.append(capital)
        nav_df = pd.DataFrame({'date': self.df.index[self.window:], 'nav': navs})
        return nav_df


# Risikoprofile
def allocate_position(signal: float, capital: float, current_price: float, risk_profile: str) -> float:
    weights = {"aggressive": 1.0, "normal": 0.5, "cautious": 0.2}
    factor = weights.get(risk_profile, 0.5)
    return factor * signal * capital / current_price

# Langfristige Simulation
class LongTermSimulation:
    def __init__(self, df_test, ta_strategy, ml_strategy, start_capital=10000, risk_profile="normal"):
        self.df_test = df_test
        self.ta_strategy = ta_strategy
        self.ml_strategy = ml_strategy
        self.start_capital = start_capital
        self.risk_profile = risk_profile

    def run(self):
        nav_ta, nav_ml = [self.start_capital], [self.start_capital]
        for i in range(1, len(self.df_test)):
            signal_ta = self.ta_strategy(self.df_test.iloc[:i])
            signal_ml = self.ml_strategy(self.df_test.iloc[:i])
            price = self.df_test['Close'].iloc[i]
            pos_ta = allocate_position(signal_ta, nav_ta[-1], price, self.risk_profile)
            pos_ml = allocate_position(signal_ml, nav_ml[-1], price, self.risk_profile)
            nav_ta.append(nav_ta[-1] + pos_ta * (self.df_test['Close'].iloc[i] - self.df_test['Close'].iloc[i-1]))
            nav_ml.append(nav_ml[-1] + pos_ml * (self.df_test['Close'].iloc[i] - self.df_test['Close'].iloc[i-1]))
        result = pd.DataFrame({
            "date": self.df_test.index,
            "nav_ta": nav_ta,
            "nav_ml": nav_ml,
            "best_strategy": ["ta" if a > b else "ml" for a, b in zip(nav_ta, nav_ml)]
        })
        return result

## ML & Zeitreihen-Integration (final)

- Die ML-Pipeline ist vollständig modular in src/ml/ implementiert:
    - Zeitliche Splits (time_split.py)
    - Feature Engineering (feature_engineering.py)
    - LSTM-Training & Prognose (model_training.py)
    - Backtest & Simulation (backtest_simulation.py)
    - Metriken (metrics.py)
    - Visualisierung (plots.py)
    - Portfolio-Allokation (allocation.py)
    - Langfrist-Simulation (long_term_simulation.py)
- Alle Module sind unabhängig und greifen nicht in die bestehende Pipeline ein.
- Die gesamte ML-Pipeline ist mit Unittests und End-to-End-Tests in tests/ml/ abgedeckt.
- Beispielcode und Teststrategie siehe README und tests/ml/test_e2e.py.
- Erweiterungen (weitere Modelle, Features) sind jederzeit möglich.


