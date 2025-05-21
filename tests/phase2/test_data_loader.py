import os
import shutil
import tempfile
from typing import List
import pytest
from src import data_loader


def test_read_tickers(tmp_path) -> None:
    """Testet das Einlesen von Ticker-Symbolen aus einer Datei."""
    content = "AAPL\nMSFT\nGOOGL\n"
    file_path = tmp_path / "tickers.txt"
    file_path.write_text(content, encoding="utf-8")
    tickers = data_loader.read_tickers(str(file_path))
    assert tickers == ["AAPL", "MSFT", "GOOGL"]


def test_download_and_save_data(monkeypatch, tmp_path) -> None:
    """Testet das Herunterladen und Speichern von Daten (Mock)."""
    tickers = ["AAPL"]
    dest_dir = tmp_path / "raw"
    os.makedirs(dest_dir, exist_ok=True)
    # Mock yf.download
    class DummyDF:
        empty = False
        def to_csv(self, path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("dummy")
        def __len__(self):
            return 42
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **kw: DummyDF())
    data_loader.download_and_save_data(tickers, str(dest_dir), period="1d", interval="1d")
    assert os.path.exists(dest_dir / "AAPL.csv")


def test_download_and_save_data_empty(monkeypatch, tmp_path) -> None:
    """Testet das Verhalten, wenn keine Daten geladen werden (Mock)."""
    tickers = ["AAPL"]
    dest_dir = tmp_path / "raw"
    os.makedirs(dest_dir, exist_ok=True)
    class DummyDF:
        empty = True
        def to_csv(self, path):
            pass
        def __len__(self):
            return 0
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **kw: DummyDF())
    data_loader.download_and_save_data(tickers, str(dest_dir), period="1d", interval="1d")
    assert not os.path.exists(dest_dir / "AAPL.csv") 