"""
model_training.py
LSTM-Modell für Zeitreihenprognose (Keras/TensorFlow).
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json
import plotly.graph_objs as go
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_log_return(series: np.ndarray) -> np.ndarray:
    """Berechnet den log-Return einer Zeitreihe. Schützt vor Division durch 0 und NaN/Inf."""
    # Ersetze Nullen durch einen kleinen Wert, um Division durch 0 zu vermeiden
    safe_series = np.where(series <= 0, 1e-8, series)
    logret = np.log(safe_series[1:] / safe_series[:-1])
    # Ersetze NaN/Inf durch 0
    logret = np.where(~np.isfinite(logret), 0, logret)
    return logret

def prepare_lstm_data_return(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bereitet Sequenzen für LSTM vor: X = Preisfenster, y = nächster log-Return. Schützt vor Division durch 0 und ungültigen Werten."""
    X, y = [], []
    for i in range(len(series) - window - 1):
        X.append(series[i:i+window])
        denom = series[i+window]
        numer = series[i+window+1]
        # Schutz vor Division durch 0 und ungültigen Werten
        if denom <= 0 or numer <= 0 or not np.isfinite(denom) or not np.isfinite(numer):
            y.append(0.0)
        else:
            y.append(np.log(numer / denom))
    return np.array(X), np.array(y)

def build_lstm_model(input_shape: Tuple[int, int], dropout: float = 0.2) -> Sequential:
    """Erstellt ein LSTM-Modell mit Dropout für Zeitreihenprognose."""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def time_series_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Teilt DataFrame in train/val/test (zeitbasiert, keine Leaks)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test

def train_lstm_with_val(df: pd.DataFrame, window: int = 20, epochs: int = 20, dropout: float = 0.2, export_dir: str = "data/exported", ticker: str = None) -> Tuple[Sequential, MinMaxScaler, dict]:
    """Trainiert LSTM mit Val-Split, Checkpointing, Logging. Ziel: log-Return. Prüft die Daten nach jedem Split."""
    train, val, _ = time_series_split(df)
    scaler = MinMaxScaler()
    close_train = train['Close'].values
    close_val = val['Close'].values
    # Zusätzliche Datenchecks nach dem Split
    for arr, name in zip([close_train, close_val], ["Train", "Val"]):
        if np.any(arr <= 0) or np.any(~np.isfinite(arr)):
            raise ValueError(f"Fehler: {name}-Daten enthalten Nullen, negative Werte oder NaN/Inf. Bitte Daten prüfen!")
    close_train_scaled = scaler.fit_transform(close_train.reshape(-1, 1)).flatten()
    close_val_scaled = scaler.transform(close_val.reshape(-1, 1)).flatten()
    X_train, y_train = prepare_lstm_data_return(close_train_scaled, window)
    X_train = X_train[..., np.newaxis]
    X_val, y_val = prepare_lstm_data_return(close_val_scaled, window)
    X_val = X_val[..., np.newaxis]
    model = build_lstm_model((window, 1), dropout=dropout)
    # Checkpointing
    if ticker:
        os.makedirs(export_dir, exist_ok=True)
        checkpoint_path = os.path.join(export_dir, f"{ticker}_lstm_best.keras")
    else:
        checkpoint_path = "best_lstm_tmp.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=[checkpoint, early],
        verbose=0
    )
    # Logging
    log = {
        'params': {'window': window, 'epochs': epochs, 'dropout': dropout},
        'train_loss': list(history.history['loss']),
        'val_loss': list(history.history['val_loss'])
    }
    if ticker:
        log_path = os.path.join(export_dir, f"{ticker}_lstm_trainlog.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
    # Lade bestes Modell
    best_model = load_model(checkpoint_path)
    return best_model, scaler, log

def reconstruct_prices_recursive(start_price: float, log_returns: np.ndarray) -> np.ndarray:
    """Rekonstruiert eine Preisreihe aus Startpreis und log-Returns (rekursiv, Standard)."""
    prices = [start_price]
    for r in log_returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:])

def reconstruct_prices_cumsum(start_price: float, log_returns: np.ndarray) -> np.ndarray:
    """Rekonstruiert eine Preisreihe aus Startpreis und log-Returns (per cumsum/exp, vektorisiert)."""
    return start_price * np.exp(np.cumsum(log_returns))

def compare_reconstruction_methods(true_prices: np.ndarray, start_price: float, log_returns: np.ndarray) -> dict:
    """Vergleicht beide Preisrekonstruktionsmethoden und gibt Fehlermaße zurück."""
    rec1 = reconstruct_prices_recursive(start_price, log_returns)
    rec2 = reconstruct_prices_cumsum(start_price, log_returns)
    mse1 = mean_squared_error(true_prices, rec1)
    mse2 = mean_squared_error(true_prices, rec2)
    mae1 = mean_absolute_error(true_prices, rec1)
    mae2 = mean_absolute_error(true_prices, rec2)
    return {
        "recursive": {"mse": mse1, "mae": mae1},
        "cumsum": {"mse": mse2, "mae": mae2},
        "rec1": rec1,
        "rec2": rec2
    }

def reconstruct_prices_best(start_price: float, log_returns: np.ndarray, true_prices: np.ndarray = None) -> np.ndarray:
    """Wählt automatisch die beste Methode zur Preisrekonstruktion anhand echter Preise (falls vorhanden)."""
    if true_prices is not None:
        cmp = compare_reconstruction_methods(true_prices, start_price, log_returns)
        # Wähle Methode mit geringstem MSE
        if cmp["cumsum"]["mse"] < cmp["recursive"]["mse"]:
            return reconstruct_prices_cumsum(start_price, log_returns)
        else:
            return reconstruct_prices_recursive(start_price, log_returns)
    # Fallback: Standardmethode
    return reconstruct_prices_recursive(start_price, log_returns)

def reconstruct_prices(start_price: float, log_returns: np.ndarray, true_prices: np.ndarray = None) -> np.ndarray:
    """Wrapper: Rekonstruiert eine Preisreihe aus Startpreis und log-Returns, wählt beste Methode."""
    return reconstruct_prices_best(start_price, log_returns, true_prices)

def predict_lstm(
    model: Sequential,
    scaler: MinMaxScaler,
    series: np.ndarray,
    window: int,
    steps: int = 1,
    start_price: float = None,
    return_prices: bool = True,
    true_prices: np.ndarray = None,
    log_compare: bool = False
) -> np.ndarray:
    """
    Prognostiziert die nächsten log-Returns und gibt wahlweise die rekonstruierten Preise zurück.
    Wenn true_prices übergeben wird, wird die beste Rekonstruktionsmethode gewählt und der Vergleich geloggt.
    """
    input_seq = series[-window:].copy()
    preds = []
    price = start_price if start_price is not None else scaler.inverse_transform(input_seq[-1].reshape(-1, 1))[0, 0]
    for _ in range(steps):
        x = input_seq.reshape(1, window, 1)
        pred_return = model.predict(x, verbose=0)[0, 0]
        # Prüfe auf NaN/Inf/Overflow
        if not np.isfinite(pred_return) or not np.isfinite(price):
            preds.append(0)
            break
        preds.append(pred_return)
        # Preis für nächsten Schritt berechnen
        price = price * np.exp(pred_return)
        if not np.isfinite(price) or price > 1e8 or price < 1e-8:
            break
        # Skalierten Preis für nächsten Schritt berechnen
        price_scaled = scaler.transform(np.array([[price]]))[0, 0]
        input_seq = np.append(input_seq[1:], price_scaled)
    preds = np.array(preds)
    if return_prices:
        price0 = start_price if start_price is not None else scaler.inverse_transform(input_seq[-1].reshape(-1, 1))[0, 0]
        if true_prices is not None:
            # Vergleiche beide Methoden und logge die Fehlermaße
            cmp = compare_reconstruction_methods(true_prices, price0, preds)
            if log_compare:
                print(f"Rekonstruktionsvergleich: recursive MSE={cmp['recursive']['mse']:.6f}, cumsum MSE={cmp['cumsum']['mse']:.6f}")
            # Wähle beste Methode
            return reconstruct_prices_best(price0, preds, true_prices)
        else:
            return reconstruct_prices(price0, preds)
    else:
        return preds

def save_lstm_model(model: Sequential, scaler: MinMaxScaler, ticker: str, export_dir: str = "data/exported") -> None:
    """Speichert das LSTM-Modell und den Scaler für einen bestimmten Ticker."""
    os.makedirs(export_dir, exist_ok=True)
    model_path = os.path.join(export_dir, f"{ticker}_lstm_model.keras")
    scaler_path = os.path.join(export_dir, f"{ticker}_scaler.joblib")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

def load_lstm_model(ticker: str, export_dir: str = "data/exported") -> Optional[Tuple[Sequential, MinMaxScaler, dict]]:
    """Lädt das LSTM-Modell, den Scaler und das Trainingslog für einen bestimmten Ticker, falls vorhanden."""
    model_path = os.path.join(export_dir, f"{ticker}_lstm_model.keras")
    scaler_path = os.path.join(export_dir, f"{ticker}_scaler.joblib")
    log_path = os.path.join(export_dir, f"{ticker}_lstm_trainlog.json")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        log = None
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log = json.load(f)
        return model, scaler, log
    return None

# Visualisierung Trainingsverlauf
def plot_training_history(log: dict) -> go.Figure:
    """Plotly-Chart für Trainings- und Validierungsverlust."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=log['train_loss'], name='Train Loss'))
    fig.add_trace(go.Scatter(y=log['val_loss'], name='Val Loss'))
    fig.update_layout(title='LSTM Training/Validation Loss', xaxis_title='Epoch', yaxis_title='Loss', height=400)
    return fig

# Hyperparameter-Tuning (Randomized Search)
def random_search_lstm(df: pd.DataFrame, param_dist: dict, n_iter: int = 10, window: int = 20, export_dir: str = "data/exported", ticker: str = None) -> Tuple[Sequential, MinMaxScaler, dict]:
    """Führt Randomized Search für LSTM-Hyperparameter durch. Gibt bestes Modell, Scaler und Log zurück. Fehler werden abgefangen."""
    best_loss = float('inf')
    best_model = None
    best_scaler = None
    best_log = None
    for params in ParameterSampler(param_dist, n_iter=n_iter, random_state=42):
        try:
            model, scaler, log = train_lstm_with_val(df, window=window, epochs=params['epochs'], dropout=params['dropout'], export_dir=export_dir, ticker=ticker)
            val_loss = min(log['val_loss'])
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
                best_scaler = scaler
                best_log = log
        except Exception as e:
            print(f"Fehler bei Training mit Parametern {params}: {e}")
            continue
    if best_model is None:
        raise RuntimeError("Randomized Search: Kein valides Modell gefunden. Prüfe Daten und Parameter.")
    return best_model, best_scaler, best_log

def generate_ml_signals(df: pd.DataFrame, ticker: str, window: int = 20) -> pd.Series:
    """Erzeugt ML-Signale (1=Long, 0=Flat) auf Basis der Preisprognose (rekonstruierte Preise).
    Die Signale werden nur für den Testzeitraum (letzte 15% der Daten) generiert, Rest = 0.
    Gibt eine Serie mit gleicher Länge wie df zurück.
    """
    loaded = load_lstm_model(ticker)
    if loaded is None:
        print("[ML-Signale] Modell konnte nicht geladen werden!")
        return pd.Series([0]*len(df), index=df.index)
    model, scaler, _ = loaded  # log wird nicht benötigt
    n = len(df)
    val_end = int(n * 0.85)
    test_len = n - val_end
    close_scaled = scaler.transform(df["Close"].values.reshape(-1, 1)).flatten()
    start_price = df["Close"].values[val_end-1] if val_end > 0 else df["Close"].values[0]
    preds = predict_lstm(model, scaler, close_scaled, window=window, steps=test_len, start_price=start_price, return_prices=True)
    # Signal: 1 wenn Preisprognose > aktuellem Kurs, sonst 0 (nur für Testzeitraum)
    test_close = df["Close"].values[val_end:]
    signals_test = (preds > test_close).astype(int)
    # Rest mit 0 auffüllen
    signals = np.zeros(n, dtype=int)
    signals[-test_len:] = signals_test
    print(f"[ML-Signale] preds: {preds[:10]}")
    print(f"[ML-Signale] signals_test: {signals_test[:10]}")
    print(f"[ML-Signale] signals (gesamt): {signals[:20]}")
    return pd.Series(signals, index=df.index)

def test_reconstruction(prices: np.ndarray) -> dict:
    """
    Testet die Preisrekonstruktion aus log-Returns für einen echten Preisausschnitt.
    Gibt die Fehlermaße (MSE, MAE) für beide Methoden und die beste Methode zurück.
    """
    # Berechne log-Returns
    log_returns = compute_log_return(prices)
    start_price = prices[0]
    true_prices = prices[1:]
    cmp = compare_reconstruction_methods(true_prices, start_price, log_returns)
    print(f"Rekonstruktion: recursive MSE={cmp['recursive']['mse']:.6f}, cumsum MSE={cmp['cumsum']['mse']:.6f}")
    best = 'cumsum' if cmp['cumsum']['mse'] < cmp['recursive']['mse'] else 'recursive'
    print(f"Beste Methode: {best}")
    return {
        'recursive': cmp['recursive'],
        'cumsum': cmp['cumsum'],
        'best': best,
        'rec1': cmp['rec1'],
        'rec2': cmp['rec2'],
        'true': true_prices
    } 