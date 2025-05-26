# projectbrief.md 

## Projektname
Elliott-Wave-Stock-Analyzer

## Foundation Document

### Ziel
Entwicklung eines modularen Python-Tools zur automatisierten Analyse und Visualisierung von Elliott-Wellen in echten Aktienkursen (Yahoo Finance).

### Core Requirements
- Echte Yahoo-Finance-Daten, keine Demo-Daten
- Robuste Erkennung von Hochs und Tiefs
- Heuristische, interpretierbare Elliott-Wellen-Zählung
- Streamlit als einfache, interaktive GUI (ein File)
- Prognosepfeile in der Grafik
- Exportfunktion (HTML, PNG)
- Aktienliste als editierbare Datei, Neuladen möglich
- Steuerung und Dokumentation über diese Memory-Bank (nur Markdown)
- Robust, testgetrieben, ML-ready
- **Nur `uv` für Abhängigkeiten** (kein pip)

### Goals
- Präsentationsreifes, nachvollziehbares Endprodukt
- Jede Projektphase klar dokumentiert und testbar
- Bereit für künftige ML-Erweiterung


# projectbrief.md

## Ziel (Erweiterung)
Zusätzlich: ML-basierte Prognosen (insb. LSTM, weitere Zeitreihenmodelle) zur Simulation und Bewertung von Handelsstrategien mit echten Aktienkursen.
Alle ML-Komponenten sind modular, getestet, und arbeiten ausschließlich mit bereinigten Echtdaten.

## Erweiterte Core Requirements
- Modularisierte ML-Pipeline (Splits, Feature Engineering, LSTM-Training, Backtest, Reporting)
- Risikoprofil-Handling und langfristige Out-of-Sample-Simulationen
- CLI und Dashboard für ML-Modelle & Ergebnisse
