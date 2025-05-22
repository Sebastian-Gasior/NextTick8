import numpy as np
from src.ml.plots import plot_equity_curve, plot_predictions

def test_plot_equity_curve() -> None:
    """Testet das Plotten der Equity-Kurve (ohne Datei)."""
    equity = np.linspace(1, 2, 100)
    plot_equity_curve(equity, title="Test Equity")

def test_plot_predictions() -> None:
    """Testet das Plotten von Prognosen (ohne Datei)."""
    true = np.linspace(1, 2, 100)
    preds = true + np.random.normal(0, 0.1, 100)
    plot_predictions(true, preds, title="Test Prognose") 