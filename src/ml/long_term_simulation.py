"""
long_term_simulation.py
Langfristige Simulationen und Risikoprofile für ML/TA-Strategien.
"""
import numpy as np
import pandas as pd
from typing import Dict

def simulate_long_term(equity_curves: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Simuliert die Entwicklung mehrerer Strategien über lange Zeiträume."""
    df = pd.DataFrame(equity_curves)
    df['mean'] = df.mean(axis=1)
    df['min'] = df.min(axis=1)
    df['max'] = df.max(axis=1)
    return df 