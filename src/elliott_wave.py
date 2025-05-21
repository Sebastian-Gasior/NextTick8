import os
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np


def find_impulse_waves(turning_points: List[Tuple[int, float]]) -> List[List[Tuple[int, float]]]:
    """
    Sucht alle gültigen 5-teiligen Impulswellen (Elliott) in einer Liste von Wendepunkten.
    Regeln:
    - Welle 3 nie die kürzeste
    - Welle 4 überschneidet Welle 1 nicht
    Rückgabe: Liste aller gültigen Impulswellen (je 6 Punkte)
    """
    results = []
    n = len(turning_points)
    for i in range(n - 5):
        P = turning_points[i:i+6]
        # Wellenlängen
        w1 = abs(P[1][1] - P[0][1])
        w3 = abs(P[3][1] - P[2][1])
        w5 = abs(P[5][1] - P[4][1])
        if w3 < w1 or w3 < w5:
            continue  # Welle 3 nie kürzeste
        # Überschneidung prüfen
        if (P[4][1] > P[2][1] and P[4][1] < P[0][1]) or (P[4][1] < P[2][1] and P[4][1] > P[0][1]):
            continue  # Welle 4 überschneidet Welle 1
        results.append(P)
    return results


def find_correction_waves(turning_points: List[Tuple[int, float]]) -> List[List[Tuple[int, float]]]:
    """
    Sucht alle gültigen 3-teiligen Korrekturwellen (ABC) in einer Liste von Wendepunkten.
    Rückgabe: Liste aller gültigen Korrekturwellen (je 4 Punkte)
    """
    results = []
    n = len(turning_points)
    for i in range(n - 3):
        P = turning_points[i:i+4]
        # Einfache ABC-Logik: A-B-C alternierend
        if (P[1][1] > P[0][1] and P[2][1] < P[1][1] and P[3][1] > P[2][1]) or \
           (P[1][1] < P[0][1] and P[2][1] > P[1][1] and P[3][1] < P[2][1]):
            results.append(P)
    return results


def extract_turning_points(df: pd.DataFrame) -> List[Tuple[int, float]]:
    """
    Extrahiert die Indizes und Werte der Peaks und Troughs aus einem DataFrame.
    """
    points = []
    for idx, row in df.iterrows():
        if row.get("Peak", 0) == 1 or row.get("Trough", 0) == 1:
            points.append((df.index.get_loc(idx), row["Close"]))
    return points


def process_peaks_to_waves(
    peaks_dir: str,
    out_dir: str
) -> None:
    """
    Lädt alle Peak/Trough-annotierten Daten, sucht Impuls- und Korrekturwellen und speichert die Ergebnisse als CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(peaks_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(peaks_dir, file), index_col=0, parse_dates=True)
            points = extract_turning_points(df)
            impulses = find_impulse_waves(points)
            corrections = find_correction_waves(points)
            # Ergebnisse als DataFrame
            result = pd.DataFrame({
                "Impulswellen": [str(impulses)],
                "Korrekturwellen": [str(corrections)]
            })
            result.to_csv(os.path.join(out_dir, file.replace(".csv", "_waves.csv"))) 