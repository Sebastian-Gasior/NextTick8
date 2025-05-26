# techContext.md 

## Technologien
- Python 3.10+
- uv (Paketmanagement)
- yfinance (Datenabruf)
- numpy, pandas (Datenhandling)
- scipy (Peak/Trough Detection, Glättung)
- plotly/matplotlib (Visualisierung, Prognosepfeile)
- streamlit (Interaktive GUI)
- pytest/unittest (Tests)

## Setup
- Lokale Entwicklung (Cursor IDE)
- uv venv
- uv pip install -r requirements.txt
- Daten-/Testordnerstruktur wie vorgegeben
- App-Start: streamlit run app.py

## Einschränkungen
- Keine Demo-Daten, nur Yahoo Finance
- Tests und Doku für jede Phase obligatorisch
- Nur Markdown für Doku


## Erweiterte Technologien ML

- tensorflow/keras (LSTM-Modelle)
- scikit-learn (Scaler, Metriken, Pipeline)
- joblib (Speichern/Laden von Scaler, Modellen)
- pandas/numpy (Feature Engineering, Daten-Pipeline)
- pytest (Unittests für alle ML-Phasen)
- Echte, validierte Kursdaten (data/cleaned/)
