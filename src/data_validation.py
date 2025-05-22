import os
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import logging


def load_raw_data(raw_dir: str) -> List[Tuple[str, pd.DataFrame]]:
    """Lädt alle CSV-Dateien aus dem Rohdatenordner und gibt eine Liste von (Ticker, DataFrame) zurück.
    Entfernt automatisch fehlerhafte Header-/Tickerzeilen und alle Zeilen, in denen numerische Spalten nicht numerisch sind."""
    data = []
    print(f"[DEBUG] Lade Daten aus Verzeichnis: {raw_dir}")
    for file in os.listdir(raw_dir):
        if file.endswith('.csv'):
            print(f"[DEBUG] Verarbeite Datei: {file}")
            ticker = file.replace('.csv', '')
            df = pd.read_csv(os.path.join(raw_dir, file), index_col=0)
            print(f"[DEBUG] Ursprüngliche Zeilen: {len(df)}")
            # Entferne alle Zeilen, deren Index kein Datum ist (z.B. 'Ticker', 'Date', leere Zeilen)
            before_rows = len(df)
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
            df = df[df.index.notna()]
            print(f"[DEBUG] Nach Index-Filter: {len(df)} (entfernt: {before_rows - len(df)})")
            # Versuche, alle numerischen Spalten zu erzwingen
            for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Price']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Entferne alle Zeilen, in denen eine der numerischen Spalten nicht numerisch ist
            num_cols = [col for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Price'] if col in df.columns]
            before_num = len(df)
            df = df.dropna(subset=num_cols, how='any')
            print(f"[DEBUG] Nach numerischem Filter: {len(df)} (entfernt: {before_num - len(df)})")
            if before_num - len(df) > 0:
                print(f"[DEBUG] Beispiel entfernte Zeilen (nicht-numerisch):")
                print(df.head(3))
            data.append((ticker, df))
    print(f"[DEBUG] Fertig mit Laden aller Dateien.")
    return data


def remove_repeated_tail(df: pd.DataFrame, num_cols: list[str], min_repeats: int = 4) -> pd.DataFrame:
    """
    Entfernt am Ende der Zeitreihe identische Zeilen, die durch Interpolation/Fill entstanden sind.
    Entfernt alle aufeinanderfolgenden identischen Endzeilen, wenn mindestens min_repeats vorhanden sind.
    """
    if len(df) < min_repeats:
        return df
    tail = df[num_cols].iloc[-min_repeats:]
    # Prüfe, ob die letzten min_repeats Zeilen identisch sind
    if tail.drop_duplicates().shape[0] == 1:
        # Suche, wie viele identische Endzeilen es wirklich gibt
        ref = df[num_cols].iloc[-1]
        n_ident = 1
        for i in range(2, len(df)+1):
            if (df[num_cols].iloc[-i] == ref).all():
                n_ident += 1
            else:
                break
        print(f"[DEBUG] Entferne {n_ident} identische Zeilen am Ende.")
        return df.iloc[:-n_ident]
    return df


def validate_and_clean(
    df: pd.DataFrame,
    method_outlier: str = 'zscore',
    z_thresh: float = 3.0,
    iqr_factor: float = 1.5,
    hampel_window: int = 7
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prüft und bereinigt einen DataFrame:
    - Entfernt Zeilen mit ungültigem Datum
    - Entfernt Zeilen, in denen numerische Spalten nicht numerisch sind oder fehlen
    - Entfernt doppelte Datumswerte (Index)
    Gibt den bereinigten DataFrame und eine Liste der angewandten Korrekturen zurück.
    """
    log = []
    print(f"[DEBUG] validate_and_clean: Start mit {len(df)} Zeilen")
    # Stelle sicher, dass Index ein Datum ist
    before = len(df)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    df = df[df.index.notna()]
    after = len(df)
    if after < before:
        log.append(f"{before-after} Zeilen mit ungültigem Datum entfernt.")
        print(f"[DEBUG] Entfernte Zeilen mit ungültigem Datum: {before-after}")
    # Entferne doppelte Datumswerte
    before_dups = len(df)
    df = df[~df.index.duplicated(keep='first')]
    after_dups = len(df)
    if after_dups < before_dups:
        log.append("Duplikate entfernt.")
        print(f"[DEBUG] Entfernte Duplikate: {before_dups-after_dups}")
    # Stelle sicher, dass numerische Spalten numerisch sind
    num_cols = [col for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Price'] if col in df.columns]
    for col in num_cols:
        before_num = len(df)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=[col], how='any')
        after_num = len(df)
        if after_num < before_num:
            log.append(f"{before_num-after_num} Zeilen mit nicht-numerischem Wert in {col} entfernt.")
            print(f"[DEBUG] Entfernte Zeilen mit nicht-numerischem Wert in {col}: {before_num-after_num}")
    # Nach Umwandlung: Entferne alle Zeilen mit NaN in numerischen Spalten (nochmals, für alle Spalten gemeinsam)
    before_nan_final = len(df)
    if num_cols:
        df = df.dropna(subset=num_cols, how='any')
    after_nan_final = len(df)
    if after_nan_final < before_nan_final:
        log.append(f"{before_nan_final-after_nan_final} Zeilen mit fehlenden Werten entfernt (final).")
        print(f"[DEBUG] Entfernte Zeilen mit fehlenden Werten (final): {before_nan_final-after_nan_final}")
    # Entferne identische Endzeilen (Test-Logik)
    df = remove_repeated_tail(df, list(num_cols), min_repeats=4)
    print(f"[DEBUG] validate_and_clean: Ende mit {len(df)} Zeilen")
    print("[DEBUG] Letzte 10 Zeilen NACH Bereinigung:")
    print(df.tail(10))
    return df, log


def process_all_raw(
    raw_dir: str,
    cleaned_dir: str,
    log_path: Optional[str] = None,
    method_outlier: str = 'zscore'
) -> None:
    """
    Lädt alle Rohdaten, bereinigt sie und speichert sie in cleaned_dir. Korrekturen werden geloggt.
    Vor dem Schreiben wird die Zieldatei gelöscht, falls sie existiert. PermissionError wird abgefangen.
    """
    os.makedirs(cleaned_dir, exist_ok=True)
    logger = logging.getLogger("data_validation")
    logger.setLevel(logging.INFO)
    if log_path:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    for ticker, df in load_raw_data(raw_dir):
        cleaned, corrections = validate_and_clean(df, method_outlier=method_outlier)
        path = os.path.join(cleaned_dir, f"{ticker}.csv")
        try:
            if os.path.exists(path):
                os.remove(path)
            with open(path, "w", encoding="utf-8") as f:
                cleaned.to_csv(f)
        except PermissionError:
            msg = f"[ERROR] Datei {path} ist gesperrt und kann nicht überschrieben werden!"
            print(msg)
            logger.error(msg)
            continue
        logger.info(f"{ticker}: {', '.join(corrections) if corrections else 'keine Korrekturen'}")


if __name__ == "__main__":
    process_all_raw("data/raw", "data/cleaned", log_path="data/cleaning.log", method_outlier="zscore")
    print("Validierung & Bereinigung abgeschlossen.") 