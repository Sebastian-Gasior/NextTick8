# Platzhalter für die Streamlit-App 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
from src import data_loader, data_validation, peak_detection, elliott_wave
from datetime import datetime, timedelta

st.set_page_config(page_title="Elliott-Wave-Stock-Analyzer", layout="wide")
st.title("Elliott-Wave-Stock-Analyzer")

# --- Sidebar: Parameter und Aktienauswahl ---
st.sidebar.header("Parameter & Steuerung")

stocks_file = "stocks.txt"
if st.sidebar.button("Aktienliste neu laden"):
    st.rerun()

with open(stocks_file, "r", encoding="utf-8") as f:
    tickers = [line.strip() for line in f if line.strip()]
selected_ticker = st.sidebar.selectbox("Wähle Aktie", tickers)

smoothing = st.sidebar.selectbox("Glättung", ["savgol", "ma"], index=0)
window = st.sidebar.slider("Glättungsfenster", 3, 21, 7, step=2)
polyorder = st.sidebar.slider("Savitzky-Golay Ordnung", 2, 5, 3) if smoothing == "savgol" else 2
prominence = st.sidebar.slider("Peak-Prominenz", 0.001, 0.1, 0.01, step=0.001)
distance = st.sidebar.slider("Peak-Mindestabstand", 1, 30, 5)

# --- Sidebar: Zeitraum-Auswahl ---
st.sidebar.subheader("Zeitraum-Auswahl")
min_date = None
max_date = None
try:
    df_tmp = pd.read_csv(f"data/cleaned/{tickers[0]}.csv", index_col=0)
    df_tmp.index = pd.to_datetime(df_tmp.index, format="%Y-%m-%d", errors="coerce")
    min_date = df_tmp.index.min().date()
    max_date = df_tmp.index.max().date()
except Exception:
    min_date = datetime(2000,1,1).date()
    max_date = datetime.today().date()
def_date_start = min_date

def_date_end = max_date
start_date = st.sidebar.date_input("Startdatum", value=def_date_start, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Enddatum", value=def_date_end, min_value=min_date, max_value=max_date)
prognose_tage = st.sidebar.slider("Prognosezeitraum (Tage)", 10, 120, 30, step=5)

if st.sidebar.button("Daten aktualisieren (Download & Clean)"):
    data_loader.download_and_save_data(tickers, "data/raw", period="max", interval="1d", log_path="data/download.log")
    data_validation.process_all_raw("data/raw", "data/cleaned", log_path="data/cleaning.log", method_outlier="zscore")
    st.success("Daten aktualisiert!")

try:
    # --- Daten laden ---
    raw_path = f"data/raw/{selected_ticker}.csv"
    cleaned_path = f"data/cleaned/{selected_ticker}.csv"
    if not os.path.exists(cleaned_path):
        st.warning("Bereinigte Daten nicht gefunden. Bitte Daten aktualisieren.")
        st.stop()
    df_all = pd.read_csv(cleaned_path, index_col=0)
    df_all.index = pd.to_datetime(df_all.index, format="%Y-%m-%d", errors="coerce")
    # Zeitraum für Hauptanzeige filtern
    df = df_all[(df_all.index.date >= start_date) & (df_all.index.date <= end_date)]
    # Zeitraum für Prognose (Echt-Daten nach Enddatum)
    prognose_ende = end_date + timedelta(days=prognose_tage)
    df_future = df_all[(df_all.index.date > end_date) & (df_all.index.date <= prognose_ende)]
    # Entferne alle Zeilen, in denen eine der numerischen Spalten nicht numerisch ist
    for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    num_cols = [col for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Price'] if col in df.columns]
    df = df.dropna(subset=num_cols, how='any')
    # --- Peak/Trough Detection ---
    smooth = peak_detection.smooth_series(df["Close"], method=smoothing, window=window, polyorder=polyorder)
    peaks_troughs = peak_detection.detect_peaks_and_troughs(smooth, prominence=prominence, distance=distance)
    df["Smooth"] = smooth
    df["Peak"] = 0
    df["Trough"] = 0
    df.iloc[peaks_troughs["peaks"], df.columns.get_loc("Peak")] = 1
    df.iloc[peaks_troughs["troughs"], df.columns.get_loc("Trough")] = 1
    # --- Elliott-Wellen-Zählung ---
    turning_points = elliott_wave.extract_turning_points(df)
    impulses = elliott_wave.find_impulse_waves(turning_points)
    corrections = elliott_wave.find_correction_waves(turning_points)
    # --- Plotly-Visualisierung ---
    try:
        fig = go.Figure()
        close_valid = df["Close"].replace([np.inf, -np.inf], np.nan).dropna()
        smooth_valid = df["Smooth"].replace([np.inf, -np.inf], np.nan).dropna()
        fig.add_trace(go.Scatter(x=close_valid.index, y=close_valid, mode="lines", name="Kurs", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=smooth_valid.index, y=smooth_valid, mode="lines", name="Glättung", line=dict(color="#ff7f0e", dash="dot")))
        fig.add_trace(go.Scatter(x=df.index[df["Peak"] == 1], y=df["Close"][df["Peak"] == 1], mode="markers", name="Peaks", marker=dict(color="green", size=10, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=df.index[df["Trough"] == 1], y=df["Close"][df["Trough"] == 1], mode="markers", name="Troughs", marker=dict(color="red", size=10, symbol="triangle-down")))
        # --- Kursverlauf nach Enddatum als graue Linie ---
        if not df_future.empty:
            fig.add_trace(go.Scatter(x=df_future.index, y=df_future["Close"], mode="lines", name="Zukunft (echt)", line=dict(color="gray", width=2, dash="dot"), opacity=0.5))
        # --- Verbesserte Elliott-Wellen-Prognose auf Basis echter Wellen ---
        if not df.empty:
            # 1. Letztes Hoch (Peak) im gewählten Zeitraum
            peak_indices = df.index[df["Peak"] == 1].tolist()
            if peak_indices:
                last_peak_idx = peak_indices[-1]
                last_peak_val = df.loc[last_peak_idx, "Close"]
                # 2. Letzte abgeschlossene Korrekturwellen vor Enddatum analysieren
                # Wir nehmen die letzten 2 Korrekturwellen aus der echten Analyse
                last_corrections = [wave for wave in corrections if wave[-1][0] < len(df)]
                if len(last_corrections) >= 2:
                    # Berechne durchschnittliche Tiefe und Dauer
                    depths = []
                    durations = []
                    for wave in last_corrections[-2:]:
                        idx0, price0 = wave[0]
                        idx1, price1 = wave[-1]
                        depth = abs(price1 - price0) / price0
                        duration = abs((df.index[idx1] - df.index[idx0]).days)
                        depths.append(depth)
                        durations.append(duration)
                    avg_depth = np.mean(depths)
                    avg_duration = int(np.mean(durations))
                else:
                    avg_depth = 0.3
                    avg_duration = prognose_tage // 2
                # 3. Korrekturtief berechnen (analog zu echten Wellen)
                corr_idx = df.index[-1] + timedelta(days=avg_duration)
                corr_val = last_peak_val * (1 - avg_depth)
                # 4. Letzte abgeschlossene Erholungswellen analysieren
                last_impulses = [wave for wave in impulses if wave[-1][0] < len(df)]
                if len(last_impulses) >= 2:
                    recov_gains = []
                    recov_durations = []
                    for wave in last_impulses[-2:]:
                        idx0, price0 = wave[0]
                        idx1, price1 = wave[-1]
                        gain = abs(price1 - price0) / price0
                        duration = abs((df.index[idx1] - df.index[idx0]).days)
                        recov_gains.append(gain)
                        recov_durations.append(duration)
                    avg_gain = np.mean(recov_gains)
                    avg_recov_duration = int(np.mean(recov_durations))
                else:
                    avg_gain = 0.5
                    avg_recov_duration = prognose_tage - avg_duration
                # 5. Erholungsziel berechnen
                recov_idx = corr_idx + timedelta(days=avg_recov_duration)
                recov_val = corr_val * (1 + avg_gain)
                # 6. Prognosepfeile und Linien zeichnen
                # Roter Pfeil: Hoch -> Korrekturtief
                fig.add_trace(go.Scatter(
                    x=[last_peak_idx, corr_idx],
                    y=[last_peak_val, corr_val],
                    mode="lines+markers",
                    line=dict(color="red", width=3, dash="dot"),
                    marker=dict(symbol="arrow-bar-down", color="red", size=12),
                    name="Prognose Abwärts (Elliott)"
                ))
                # Horizontale rote Linie am Korrekturtief
                fig.add_shape(type="line",
                              x0=last_peak_idx, x1=recov_idx,
                              y0=corr_val, y1=corr_val,
                              line=dict(color="red", width=1, dash="dash"))
                fig.add_annotation(
                    x=corr_idx,
                    y=corr_val,
                    text=f"{corr_val:.2f}",
                    showarrow=False,
                    font=dict(color="red", size=14),
                    bgcolor="black",
                    bordercolor="red"
                )
                # Grüner Pfeil: Korrekturtief -> Erholungsziel
                fig.add_trace(go.Scatter(
                    x=[corr_idx, recov_idx],
                    y=[corr_val, recov_val],
                    mode="lines+markers",
                    line=dict(color="green", width=3, dash="dot"),
                    marker=dict(symbol="arrow-bar-up", color="green", size=12),
                    name="Prognose Aufwärts (Elliott)"
                ))
                # Horizontale grüne Linie am Erholungsziel
                fig.add_shape(type="line",
                              x0=corr_idx, x1=recov_idx,
                              y0=recov_val, y1=recov_val,
                              line=dict(color="green", width=1, dash="dash"))
                fig.add_annotation(
                    x=recov_idx,
                    y=recov_val,
                    text=f"{recov_val:.2f}",
                    showarrow=False,
                    font=dict(color="green", size=14),
                    bgcolor="black",
                    bordercolor="green"
                )
                # X-Achse erweitern
                fig.update_xaxes(range=[df.index[0], recov_idx + timedelta(days=10)])
        fig.update_layout(height=700, width=1200, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as plot_e:
        st.error(f"Fehler bei der Plotly-Visualisierung: {plot_e}")

    # --- Export ---
    export_dir = "data/exported"
    os.makedirs(export_dir, exist_ok=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Plot als PNG speichern"):
            png_path = os.path.join(export_dir, f"{selected_ticker}_plot.png")
            fig.write_image(png_path)
            with open(png_path, "rb") as f:
                st.download_button("Download PNG", f, file_name=f"{selected_ticker}_plot.png")
    with col2:
        if st.button("Plot als HTML speichern"):
            html_path = os.path.join(export_dir, f"{selected_ticker}_plot.html")
            fig.write_html(html_path)
            with open(html_path, "rb") as f:
                st.download_button("Download HTML", f, file_name=f"{selected_ticker}_plot.html")

    # --- Wellen-Tabellen ---
    st.subheader("Gefundene Impulswellen (5-teilig)")
    for wave in impulses:
        st.write([f"{df.index[idx].date()} ({price:.2f})" for idx, price in wave])
    st.subheader("Gefundene Korrekturwellen (ABC)")
    for wave in corrections:
        st.write([f"{df.index[idx].date()} ({price:.2f})" for idx, price in wave])
    # --- Help-Bereich / Legende ---
    with st.expander("ℹ️ Legende & Parameter-Erklärung anzeigen", expanded=False):
        st.markdown("""
        ### Legende & Parameter-Erklärung
        
        **Kurs:** Originaler Schlusskurs der Aktie (blaue Linie)
        
        **Glättung:**
        - **ma**: Moving Average (gleitender Durchschnitt)
        - **savgol**: Savitzky-Golay-Filter (polynomiale Glättung)
        
        **Peaks:** Lokale Hochpunkte (grüne Dreiecke)
        
        **Troughs:** Lokale Tiefpunkte (rote Dreiecke)
        
        **Zukunft (echt):** Tatsächlicher Kursverlauf nach dem gewählten Enddatum (graue Linie)
        
        **Prognose Abwärts (Elliott):** Roter Pfeil – prognostizierte Korrekturwelle gemäß Elliott-Logik
        
        **Prognose Aufwärts (Elliott):** Grüner Pfeil – prognostizierte Erholungswelle gemäß Elliott-Logik
        
        **Parameter:**
        - **Glättungsfenster:** Größe des Fensters für die Glättung (Anzahl Tage)
        - **Savitzky-Golay Ordnung:** Grad des Polynoms für Savitzky-Golay
        - **Peak-Prominenz:** Mindesthöhe, damit ein Peak erkannt wird
        - **Peak-Mindestabstand:** Mindestabstand zwischen Peaks
        - **Start-/Enddatum:** Zeitraum für die Analyse
        - **Prognosezeitraum:** Länge der Prognose in Tagen
        
        **Funktionsweise:**
        1. Echte Yahoo-Finance-Daten werden geladen und bereinigt.
        2. Die Zeitreihe wird geglättet ("ma" oder "savgol").
        3. Peaks und Troughs werden erkannt.
        4. Die Elliott-Wellen werden heuristisch gezählt.
        5. Prognosepfeile werden auf Basis der letzten echten Wellen berechnet.
        6. Die echte Kursentwicklung nach dem Enddatum wird als Vergleich angezeigt.
        
        **Hinweis:**
        - Die Prognose ist keine Finanzberatung, sondern eine algorithmische Projektion nach Elliott-Heuristik.
        - Alle Daten sind original von Yahoo Finance.
        """)
except Exception as e:
    st.error(f"Fehler im Haupt-Workflow: {e}")

# --- Help-Bereich / Legende ---
with st.expander("ℹ️ Hilfe & Legende anzeigen", expanded=False):
    st.markdown("""
    ### Legende & Parameter-Erklärung
    
    **Kurs:** Originaler Schlusskurs der Aktie (blaue Linie)
    
    **Glättung:**
    - **ma**: Moving Average (gleitender Durchschnitt)
    - **savgol**: Savitzky-Golay-Filter (polynomiale Glättung)
    
    **Peaks:** Lokale Hochpunkte (grüne Dreiecke)
    
    **Troughs:** Lokale Tiefpunkte (rote Dreiecke)
    
    **Zukunft (echt):** Tatsächlicher Kursverlauf nach dem gewählten Enddatum (graue Linie)
    
    **Prognose Abwärts (Elliott):** Roter Pfeil – prognostizierte Korrekturwelle gemäß Elliott-Logik
    
    **Prognose Aufwärts (Elliott):** Grüner Pfeil – prognostizierte Erholungswelle gemäß Elliott-Logik
    
    **Parameter:**
    - **Glättungsfenster:** Größe des Fensters für die Glättung (Anzahl Tage)
    - **Savitzky-Golay Ordnung:** Grad des Polynoms für Savitzky-Golay
    - **Peak-Prominenz:** Mindesthöhe, damit ein Peak erkannt wird
    - **Peak-Mindestabstand:** Mindestabstand zwischen Peaks
    - **Start-/Enddatum:** Zeitraum für die Analyse
    - **Prognosezeitraum:** Länge der Prognose in Tagen
    
    **Funktionsweise:**
    1. Echte Yahoo-Finance-Daten werden geladen und bereinigt.
    2. Die Zeitreihe wird geglättet ("ma" oder "savgol").
    3. Peaks und Troughs werden erkannt.
    4. Die Elliott-Wellen werden heuristisch gezählt.
    5. Prognosepfeile werden auf Basis der letzten echten Wellen berechnet.
    6. Die echte Kursentwicklung nach dem Enddatum wird als Vergleich angezeigt.
    
    **Hinweis:**
    - Die Prognose ist keine Finanzberatung, sondern eine algorithmische Projektion nach Elliott-Heuristik.
    - Alle Daten sind original von Yahoo Finance.
    """) 