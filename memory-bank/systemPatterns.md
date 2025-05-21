# systemPatterns.md

## Systemarchitektur
- Input: stocks.txt (editierbare Tickerliste)
- Datenabruf & Speicherung: Yahoo Finance → CSV in data/raw/
- Datenprüfung/-bereinigung: data/cleaned/
- Analyse: Peaks/Troughs, Wellenzählung nach Heuristik
- Visualisierung: Streamlit (ein File)
- Export: HTML, PNG Download
- Tests: In tests/phaseX/

## Wichtige Entscheidungen
- Daten werden immer zuerst gespeichert, dann geprüft und bereinigt
- Jede Funktionseinheit einzeln testbar und dokumentiert
- Datenintegrität hat höchste Priorität

## Design Patterns
- Single Responsibility für jede Funktion
- Modularisierung: Download → Clean → Analyse → Visualisierung → Export
- Jede Phase ist eigenständig testbar

## Komponentenbeziehungen
- stocks.txt ↔ Datenimport
- data/raw/ → Datenvalidierung → data/cleaned/ → Analyse/Visualisierung
- tests/phaseX/ ↔ jeweilige Hauptfunktion
