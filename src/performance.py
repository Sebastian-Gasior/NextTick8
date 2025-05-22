"""
performance.py
Berechnung von Performance-Kennzahlen für Strategien.
"""
import numpy as np
import pandas as pd

def compute_performance_metrics(equity: np.ndarray, returns: np.ndarray, trades: np.ndarray) -> dict:
    """Berechnet Performance-Kennzahlen für eine Strategie.

    Args:
        equity (np.ndarray): Equity-Kurve (Kapitalverlauf)
        returns (np.ndarray): Einzelne Renditen (z.B. Tagesrenditen)
        trades (np.ndarray): Einzelne Trades (Gewinn/Verlust pro Trade)

    Returns:
        dict: Dictionary mit Performance-Kennzahlen
    """
    total_return = float(equity[-1] / equity[0] - 1) if len(equity) > 1 else 0.0
    win_trades = trades[trades > 0]
    loss_trades = trades[trades < 0]
    num_trades = int(len(trades))
    win_rate = float(len(win_trades) / num_trades * 100) if num_trades > 0 else 0.0
    avg_profit = float(win_trades.mean()) if len(win_trades) > 0 else 0.0
    avg_loss = float(abs(loss_trades.mean())) if len(loss_trades) > 0 else 0.0
    # Profit Factor: Summe Gewinne / Summe Verluste (nur wenn Verluste vorhanden)
    profit_factor = float(win_trades.sum() / abs(loss_trades.sum())) if abs(loss_trades.sum()) > 1e-9 else float('nan')
    # Max Drawdown
    if len(equity) > 1:
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_drawdown = float(np.min(drawdowns))
    else:
        max_drawdown = 0.0
    # Volatilität (annualisiert)
    volatility = float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0
    # Sharpe Ratio (annualisiert, robust gegen 0)
    std_returns = np.std(returns)
    sharpe = float((np.mean(returns) / (std_returns + 1e-9)) * np.sqrt(252)) if len(returns) > 1 else 0.0
    # Plausibilitäts-Check: unrealistische Werte abfangen
    if not np.isfinite(profit_factor) or profit_factor > 100:
        profit_factor = 0.0
    if not np.isfinite(sharpe) or abs(sharpe) > 10:
        sharpe = 0.0
    if not np.isfinite(max_drawdown):
        max_drawdown = 0.0
    return {
        "Total Return": round(total_return, 3),
        "Number of Trades": num_trades,
        "Win Rate": round(win_rate, 2),
        "Avg Profit": round(avg_profit, 2),
        "Avg Loss": round(avg_loss, 2),
        "Profit Factor": round(profit_factor, 2),
        "Max Drawdown": round(max_drawdown, 2),
        "Volatility": round(volatility, 2),
        "Sharpe Ratio": round(sharpe, 2)
    } 