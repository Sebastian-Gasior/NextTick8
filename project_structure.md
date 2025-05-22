# Projektstruktur

Die folgende Struktur bildet die Basis für das Elliott-Wave-Stock-Analyzer-Projekt. Sie wird nach jeder Änderung automatisch aktualisiert.

```
/
├── README.md
├── .gitignore
├── LICENSE
├── memory-bank/
│   ├── projectbrief.md
│   ├── productContext.md
│   ├── techContext.md
│   ├── systemPatterns.md
│   ├── activeContext.md
│   └── progress.md
├── .cursor/
│   └── rules/
│       ├── memory-bank.mdc
│       └── python-main-rules.mdc
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── exported/         # Exportierte Visualisierungen (PNG, HTML)
│   ├── download.log      # Logfile für Daten-Downloads
│   ├── cleaning.log      # Logfile für Datenbereinigung
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── peak_detection.py
│   ├── elliott_wave.py
│   └── ml/
│       ├── __init__.py
│       ├── time_split.py
│       ├── feature_engineering.py
│       ├── model_training.py
│       ├── backtest_simulation.py
│       ├── metrics.py
│       ├── plots.py
│       ├── allocation.py
│       └── long_term_simulation.py
├── tests/
│   ├── __init__.py
│   ├── phase2/
│   │   ├── __init__.py
│   │   └── test_data_loader.py
│   ├── phase3/
│   │   ├── __init__.py
│   │   └── test_data_validation.py
│   ├── phase4/
│   │   ├── __init__.py
│   │   └── test_peak_detection.py
│   └── phase5/
│       ├── __init__.py
│       └── test_elliott_wave.py
│   └── ml/
│       ├── __init__.py
│       ├── test_time_split.py
│       ├── test_feature_engineering.py
│       ├── test_model_training.py
│       ├── test_backtest_simulation.py
│       ├── test_metrics.py
│       ├── test_plots.py
│       ├── test_allocation.py
│       ├── test_long_term_simulation.py
│       └── test_e2e.py
├── stocks.txt
├── requirements.txt
├── app.py
``` 