# activeContext.md 

## Aktueller Fokus

**Abgeschlossen:**
- Phase 1: Initialisierung & Projektstruktur
  - Alle Basisordner und -dateien angelegt
  - Memory-Bank und Dokumentation aktualisiert
- Phase 2: Datendownload & Rohdaten-Management
  - Laden von Yahoo-Finance-Daten gemäß stocks.txt
  - Speicherung als CSV in data/raw
  - Fehler- und Lückenprüfung, Logging der Ergebnisse
- Phase 3: Datenvalidierung & Bereinigung
  - Validierungs- und Bereinigungslogik für Kursdaten implementiert
  - Automatische Erkennung und Korrektur von Lücken, Duplikaten, NaN, Ausreißern (z-Score, IQR, Hampel)
  - Alle Unittests und Integrationstests bestanden
- Phase 4: Peak/Trough Detection
  - Glättung der Kursdaten (Moving Average, Savitzky-Golay) implementiert
  - Robuste Peak/Trough Detection mit Scipy (Prominenz, Abstand, Randerkennung)
  - End-to-End-Tests und Unittests für verschiedene Szenarien bestanden
- Phase 5: Elliott-Wave-Heuristik & Zählung
  - Heuristische Elliott-Wellen-Zählung (Impuls- und Korrekturwellen) implementiert
  - Zentrale Regeln (Welle 3 nie kürzeste, Welle 4 überschneidet Welle 1 nicht, ABC-Korrektur) werden eingehalten
  - End-to-End-Tests und Unittests für Wellenzählung bestanden
- Phase 6: Visualisierung (Streamlit GUI)
  - Interaktive Streamlit-App (app.py) implementiert
  - Visualisierung von Kurs, Glättung, Peaks/Troughs, Wellenlabels, Prognosepfeilen
  - Parameteranpassung, Aktienliste-Reload, Datenaktualisierung direkt in der App
  - Export als PNG und HTML integriert
  - Alle Tests bestanden, App lauffähig
- Phase 7: Export & Sharing
  - Exportfunktion für interaktive Ergebnisse
  - Exportierte Ergebnisse in eigenen Ordner ablegen
- Phase 8: Test, Stabilisierung & Dokumentation abgeschlossen
  - Alle Unittests und Integrationstests bestanden
  - Fehlerbehandlung, Nutzerführung und Dokumentation finalisiert
  - Troubleshooting, FAQ und Beispiele ergänzt
  - App ist stabil und bereit für Präsentation/Deployment

**Aktueller Fokus:**
- Phase 9: Präsentation, Deployment & Ausblick

**Nächste Aufgaben:**
- Export- und Sharing-Logik prüfen und ggf. erweitern
- Dokumentation der Exportmöglichkeiten
- Test und Validierung der Exportfunktionen

## Letzte Änderungen
- Projektverzeichnis erstellt
- Ordnerstruktur angelegt
- uv-Setup und erste Dateien vorbereitet

## Nächste Schritte
- Streamlit-Starter (app.py) lauffähig machen
- Basis-Test für Setup
- stocks.txt Beispielaktien eintragen

## Wichtige Überlegungen
- Jede Änderung muss dokumentiert und getestet werden
- Datenvalidierung ist ein Kernfeature (später: eigene Phase!)
- Alle Tests nach Phasen geordnet


## Aktueller Fokus (ML/Deep Learning)
- ML-Phase abgeschlossen: LSTM-Modell, Zeitreihen- und Backtest-Module integriert, alle Tests bestanden
- src/ml/ und tests/ml/ mit allen Kernmodulen und Unittests
- Test, Persistenz, Simulation & Reporting erfolgreich umgesetzt

## Nächste Aufgaben (ML)
- Präsentation und Deployment der ML-Pipeline
- Erweiterung um weitere Modelle/Features bei Bedarf
