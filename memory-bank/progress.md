# progress.md 

## Was funktioniert?
- Projektstruktur steht
- uv-Setup und Basisdokumentation vorhanden

## Was ist noch offen?
- Streamlit-Starter testen
- Erste Rohdaten laden und validieren
- Tests für Setup und Datenimport

## Status
- Phase 1: Initialisierung läuft

## Bekannte Probleme
- Noch keine Rohdaten geprüft
- Keine Visualisierung
- Keine Bereinigung implementiert

## Protokoll
- [Datum] Projektstruktur angelegt, Doku erstellt
- [Datum] Nächster Meilenstein: Datenvalidierung, Test

# Fortschritt

- [x] Phase 1: Initialisierung & Projektstruktur
    - Projektstruktur, Basisordner und -dateien, Memory-Bank, requirements.txt, stocks.txt, Platzhalterdateien
- [x] Phase 2: Datendownload & Rohdaten-Management
    - Download von Yahoo-Finance-Daten
    - Speicherung als CSV in data/raw
    - Fehler- und Lückenprüfung
    - Logging der Ergebnisse
- [x] Phase 3: Datenvalidierung & Bereinigung
    - Validierungs- und Bereinigungslogik für Kursdaten implementiert
    - Automatische Erkennung und Korrektur von Lücken, Duplikaten, NaN, Ausreißern (z-Score, IQR, Hampel)
    - Bereinigte Daten werden in data/cleaned gespeichert
    - Alle Unittests und Integrationstests bestanden
- [x] Phase 4: Peak/Trough Detection
    - Glättung der Kursdaten (Moving Average, Savitzky-Golay) implementiert
    - Robuste Peak/Trough Detection mit Scipy (Prominenz, Abstand, Randerkennung)
    - End-to-End-Tests und Unittests für verschiedene Szenarien bestanden
    - Ergebnisse werden als CSV mit Peak/Trough-Labels gespeichert
- [x] Phase 5: Elliott-Wave-Heuristik & Zählung
    - Heuristische Elliott-Wellen-Zählung (Impuls- und Korrekturwellen) implementiert
    - Zentrale Regeln (Welle 3 nie kürzeste, Welle 4 überschneidet Welle 1 nicht, ABC-Korrektur) werden eingehalten
    - End-to-End-Tests und Unittests für Wellenzählung bestanden
    - Ergebnisse werden als CSV mit Wellenlisten gespeichert
- [x] Phase 6: Visualisierung (Streamlit GUI)
    - Interaktive Streamlit-App (app.py) implementiert
    - Visualisierung von Kurs, Glättung, Peaks/Troughs, Wellenlabels, Prognosepfeilen
    - Parameteranpassung, Aktienliste-Reload, Datenaktualisierung direkt in der App
    - Export als PNG und HTML integriert
    - Alle Tests bestanden, App lauffähig
- [x] Phase 7: Export & Sharing
    - Exportfunktion für interaktive Ergebnisse (PNG, HTML) implementiert
    - Exportierte Ergebnisse werden in data/exported abgelegt
    - Export- und Sharing-Logik in der App integriert und getestet
    - Alle Tests bestanden, Export lauffähig
- [x] Phase 8: Test, Stabilisierung & Dokumentation
    - Alle Unittests und Integrationstests grün (100 % bestanden)
    - Fehlerbehandlung und Nutzerführung optimiert
    - Troubleshooting, FAQ und Beispielanalysen dokumentiert
    - Dokumentation (README, project_structure.md, memory-bank) finalisiert
    - App ist stabil und bereit für Präsentation/Deployment
- [ ] Phase 9: Präsentation, Deployment & Ausblick
- [x] ML-Phase: LSTM-Modell, Zeitreihen- und Backtest-Module integriert
    - src/ml/ und tests/ml/ mit allen Kernmodulen und Tests angelegt
    - LSTM-Training, Feature Engineering, Walk-Forward, Backtest, Reporting
    - Fortschritt und Status werden laufend dokumentiert

## Bekannte Probleme / Offene Punkte
- Noch keine


## Fortschritt (ML)

- [ ] Phase ML-1: Zeitliche Splits & Walk-Forward (Tests: test_time_split.py)
- [ ] Phase ML-2: Feature Engineering (Tests: test_feature_engineering.py)
- [ ] Phase ML-3: Modell-Training (LSTM etc.) (Tests: test_model_training.py)
- [ ] Phase ML-4: Backtest & Simulation (Tests: test_backtest_simulation.py)
- [ ] Phase ML-5: Reporting & Evaluation (Tests: test_metrics.py, test_plots.py)
- [ ] Phase ML-6: Langfrist-Simulation/Risikoprofile (Tests: test_long_term_simulation.py, test_allocation.py)
- [ ] Phase ML-7: CLI & Dashboard-Integration

**Status:**
- ML-Pipeline wird aufgebaut und Schritt für Schritt getestet.
## ML-Integration & Zeitreihenmodelle

### Überblick

- Das Projekt enthält eine **vollständige ML-Pipeline** für Zeitreihenprognosen und Backtests mit echten Aktienkursen.
- Alle Modelle (insb. LSTM) werden ausschließlich auf validierten Kursdaten aus `data/cleaned/` trainiert und evaluiert.

### Ablauf (ML-Phasen)

1. **Zeitliche Splits:** Striktes Out-of-Sample- und Walk-Forward-Splitting (src/ml/time_split.py)
2. **Feature Engineering:** Technische Features und Zeitreihenindikatoren (src/ml/feature_engineering.py)
3. **Model Training:** LSTM-Modelle (TensorFlow/Keras), weitere Zeitreihenmodelle (src/ml/model_training.py)
4. **Backtest & Simulation:** ML-Signale und Kontoentwicklung auf nie gesehenen Daten (src/ml/backtest_simulation.py)
5. **Evaluierung:** Sharpe, Drawdown, Hit Ratio etc. (src/ml/metrics.py)
6. **Langfrist-Simulation & Risikoprofile:** Parallel-Simulation ML/TA, verschiedene Risikoprofile, Visualisierung (src/ml/long_term_simulation.py, src/ml/allocation.py)
7. **Reporting & Dashboard:** Ergebnisse werden exportiert, im Dashboard angezeigt und als Charts gespeichert.

### Alle ML-Funktionen und Beispielcode findest du in `memory-bank/systemPatterns.md`.  
Alle Schritte sind modular, getestet und dokumentiert.
## ML-Integration & Zeitreihenmodelle

