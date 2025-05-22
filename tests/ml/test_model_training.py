import pandas as pd
import numpy as np
from src.ml.model_training import train_lstm_with_val, predict_lstm, plot_training_history

def test_train_and_predict_lstm() -> None:
    """Testet das Training und die Vorhersage eines LSTM-Modells mit Val-Split und Logging."""
    df = pd.DataFrame({"Close": np.linspace(100, 120, 50)})
    model, scaler, log = train_lstm_with_val(df, window=5, epochs=2, dropout=0.1)
    assert model is not None
    assert scaler is not None
    assert 'train_loss' in log and 'val_loss' in log
    # Teste Vorhersage
    close_scaled = scaler.transform(df["Close"].values.reshape(-1, 1)).flatten()
    preds = predict_lstm(model, scaler, close_scaled, window=5, steps=3)
    assert len(preds) == 3
    # Teste Visualisierung
    fig = plot_training_history(log)
    assert fig is not None 