# Elliott-Wave-Stock-Analyzer

## Projektbeschreibung

Dieses Projekt entwickelt ein robustes, automatisiertes System zur Analyse und Visualisierung von Elliott-Wellen in echten Aktienkursdaten (Yahoo Finance).  
Das Ziel: Analysten und Unternehmen erhalten verständliche, nachvollziehbare und reproduzierbare Wellenanalysen als Basis für Investment-Entscheidungen.  
Alle Abläufe und Dokumentationen werden ausschließlich mit Markdown geführt.  
Die Implementierung ist vollständig modular, testgetrieben und für spätere ML-Erweiterungen vorbereitet.

---

## Projektphasen (Übersicht)

1. **Initialisierung & Projektstruktur**  
    Grundstruktur, Setup, Ordner, uv, Memory-Bank, README.md

2. **Datendownload & Rohdaten-Management**  
    - Laden von Yahoo-Finance-Daten gemäß stocks.txt  
    - Speicherung als CSV im `data/raw/`-Ordner  
    - Fehler- und Lückenprüfung, Logging der Ergebnisse

3. **Datenvalidierung & Bereinigung**  
    - Prüfung auf Inkonsistenzen: fehlende/duplizierte Zeitstempel, Ausreißer, NaNs  
    - Automatische Bereinigung: Interpolation, Ausreißerfilter (z-Score, IQR, Hampel)  
    - Speicherung bereinigter Daten in `data/cleaned/`  
    - Protokollierung aller Korrekturen

4. **Peak/Trough Detection**  
    - Glättung (Moving Average, Savitzky-Golay)  
    - Robuste lokale Extremwerterkennung mit einstellbaren Parametern  
    - Tests auf verschiedenen Aktienreihen

5. **Elliott-Wave-Heuristik & Zählung**  
    - Zuordnung von Wendepunkten zu Impuls- und Korrekturwellen  
    - Einhaltung zentraler Elliott-Regeln (z.B. Welle 3 nie kürzeste)  
    - Nutzung ggf. externer Open-Source-Module

6. **Visualisierung (Streamlit GUI)**  
    - Ein File: Anzeige Kurs, Hochs/Tiefs, Wellenlabels, Prognosepfeile  
    - Parameteranpassung, Aktienliste neuladen, Export als HTML/PNG

7. **Export & Sharing**  
    - Exportfunktion für interaktive Ergebnisse  
    - Exportierte Ergebnisse in eigenen Ordner ablegen

8. **Test, Stabilisierung & Dokumentation**  
    - Unittests, Integrationstests  
    - Zuordnung aller Tests in `tests/phaseX/`  
    - Aktualisierung von README und Memory-Bank

9. **Präsentation, Deployment & Ausblick**  
    - Demo-Script, Projektvorstellung  
    - Hinweise zur ML-Erweiterung, Deployment (z.B. auf netcup)  
    - Abschlussprotokoll

---

## Projektstand (Phase 6 abgeschlossen)

- Interaktive Streamlit-App (app.py) implementiert
- Visualisierung von Kurs, Glättung, Peaks/Troughs, Wellenlabels, Prognosepfeilen
- Parameteranpassung, Aktienliste-Reload, Datenaktualisierung direkt in der App
- Export als PNG und HTML integriert
- Alle Tests bestanden, App lauffähig

**Nächste Phase:**
Export & Sharing (Phase 7)

## Setup & Nutzung

```bash
uv venv
uv pip install -r requirements.txt
streamlit run app.py
