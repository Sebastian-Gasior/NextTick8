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
- [ ] Phase 7: Export & Sharing
    - Exportfunktion für interaktive Ergebnisse
    - Exportierte Ergebnisse in eigenen Ordner ablegen
- [ ] Phase 8: Test, Stabilisierung & Dokumentation
- [ ] Phase 9: Präsentation, Deployment & Ausblick

## Bekannte Probleme / Offene Punkte
- Noch keine
