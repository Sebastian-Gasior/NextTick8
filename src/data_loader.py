import os
from typing import List, Optional
import pandas as pd
import yfinance as yf
import logging


def read_tickers(file_path: str) -> List[str]:
    """Liest die Ticker-Symbole aus einer Datei ein."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def download_and_save_data(
    tickers: List[str],
    dest_dir: str,
    period: str = 'max',
    interval: str = '1d',
    log_path: Optional[str] = None
) -> None:
    """
    Lädt historische Kursdaten für die angegebenen Ticker von Yahoo Finance und speichert sie als CSV.
    Fehler werden geloggt.
    """
    os.makedirs(dest_dir, exist_ok=True)
    logger = logging.getLogger("data_loader")
    logger.setLevel(logging.INFO)
    if log_path:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty:
                logger.warning(f"Keine Daten für {ticker} geladen.")
                continue
            csv_path = os.path.join(dest_dir, f"{ticker}.csv")
            df.to_csv(csv_path)
            logger.info(f"{ticker}: {len(df)} Zeilen gespeichert unter {csv_path}.")
        except Exception as e:
            logger.error(f"Fehler beim Laden von {ticker}: {e}")


if __name__ == "__main__":
    tickers = read_tickers("stocks.txt")
    download_and_save_data(tickers, "data/raw", period="max", interval="1d", log_path="data/download.log")
    print("Download abgeschlossen.") 