import pandas as pd
from src import elliott_wave

def test_find_impulse_waves() -> None:
    """Testet die Erkennung von Impulswellen nach Elliott-Regeln."""
    # Künstliche Wendepunkte: 6 Punkte, Welle 3 am längsten
    points = [(0, 0), (1, 2), (2, 1), (3, 5), (4, 3), (5, 7)]
    result = elliott_wave.find_impulse_waves(points)
    assert len(result) == 1
    assert result[0][3][1] - result[0][2][1] > result[0][1][1] - result[0][0][1]

def test_find_correction_waves() -> None:
    """Testet die Erkennung von Korrekturwellen (ABC)."""
    points = [(0, 0), (1, 2), (2, 1), (3, 3)]
    result = elliott_wave.find_correction_waves(points)
    assert len(result) == 1

def test_extract_turning_points() -> None:
    """Testet das Extrahieren von Peaks/Troughs aus DataFrame."""
    df = pd.DataFrame({"Close": [1, 2, 1], "Peak": [0, 1, 0], "Trough": [1, 0, 1]}, index=[0, 1, 2])
    points = elliott_wave.extract_turning_points(df)
    assert len(points) == 3

def test_process_peaks_to_waves(tmp_path) -> None:
    """Testet die End-to-End-Verarbeitung von Peak/Trough-Daten zu Wellenzählung."""
    df = pd.DataFrame({"Close": [0, 2, 1, 5, 3, 7], "Peak": [0, 1, 0, 1, 0, 1], "Trough": [1, 0, 1, 0, 1, 0]}, index=pd.date_range("2020-01-01", periods=6))
    peaks_dir = tmp_path / "peaks"
    out_dir = tmp_path / "waves"
    peaks_dir.mkdir()
    df.to_csv(peaks_dir / "TEST.csv")
    elliott_wave.process_peaks_to_waves(str(peaks_dir), str(out_dir))
    out = pd.read_csv(out_dir / "TEST_waves.csv")
    assert "Impulswellen" in out.columns and "Korrekturwellen" in out.columns 