"""
time_split.py
Modul für Walk-Forward- und Out-of-Sample-Splits von Zeitreihen.
"""
from typing import Tuple, List
import pandas as pd

def split_fixed(df: pd.DataFrame, train_end: pd.Timestamp, sim_start: pd.Timestamp, sim_end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splittet DataFrame in Trainings- und Simulationsdaten anhand fixer Zeitpunkte."""
    train = df[df.index <= train_end].copy()
    sim = df[(df.index >= sim_start) & (df.index <= sim_end)].copy()
    return train, sim

def walk_forward_splits(df: pd.DataFrame, window: int, step: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Erzeugt Walk-Forward-Splits für Zeitreihenprognosen."""
    splits = []
    for start in range(0, len(df) - window, step):
        train = df.iloc[start:start+window]
        test = df.iloc[start+window:start+window+step]
        if len(test) == 0:
            break
        splits.append((train.copy(), test.copy()))
    return splits 