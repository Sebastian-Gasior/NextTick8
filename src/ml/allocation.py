"""
allocation.py
Einfache Portfolio-Allokation (Gleichgewichtung, risikobasiert).
"""
import numpy as np
import pandas as pd
from typing import Dict

def equal_weight_allocation(tickers: list) -> Dict[str, float]:
    """Gibt für alle Ticker die gleiche Gewichtung zurück."""
    n = len(tickers)
    return {ticker: 1.0 / n for ticker in tickers}

def risk_parity_allocation(returns: pd.DataFrame) -> Dict[str, float]:
    """Berechnet risikobasierte Gewichtung (inverse Volatilität)."""
    vol = returns.std()
    inv_vol = 1 / (vol + 1e-9)
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict() 