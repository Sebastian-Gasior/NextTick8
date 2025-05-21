# Platzhalter für die Streamlit-App 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
from src import data_loader, data_validation, peak_detection, elliott_wave

st.set_page_config(page_title="Elliott-Wave-Stock-Analyzer", layout="wide")
st.title("Elliott-Wave-Stock-Analyzer")

# --- Sidebar: Parameter und Aktienauswahl ---
st.sidebar.header("Parameter & Steuerung")

stocks_file = "stocks.txt"
if st.sidebar.button("Aktienliste neu laden"):
    st.experimental_rerun()

with open(stocks_file, "r", encoding="utf-8") as f:
    tickers = [line.strip() for line in f if line.strip()]
selected_ticker = st.sidebar.selectbox("Wähle Aktie", tickers)

smoothing = st.sidebar.selectbox("Glättung", ["savgol", "ma"], index=0)
window = st.sidebar.slider("Glättungsfenster", 3, 21, 7, step=2)
polyorder = st.sidebar.slider("Savitzky-Golay Ordnung", 2, 5, 3) if smoothing == "savgol" else 2
prominence = st.sidebar.slider("Peak-Prominenz", 0.001, 0.1, 0.01, step=0.001)
distance = st.sidebar.slider("Peak-Mindestabstand", 1, 30, 5)

if st.sidebar.button("Daten aktualisieren (Download & Clean)"):
    data_loader.download_and_save_data(tickers, "data/raw", period="max", interval="1d", log_path="data/download.log")
    data_validation.process_all_raw("data/raw", "data/cleaned", log_path="data/cleaning.log", method_outlier="zscore")
    st.success("Daten aktualisiert!")

# --- Daten laden ---
raw_path = f"data/raw/{selected_ticker}.csv"
cleaned_path = f"data/cleaned/{selected_ticker}.csv"
if not os.path.exists(cleaned_path):
    st.warning("Bereinigte Daten nicht gefunden. Bitte Daten aktualisieren.")
    st.stop()
df = pd.read_csv(cleaned_path, index_col=0, parse_dates=True, dayfirst=False, infer_datetime_format=True)
# Entferne alle Zeilen, deren Index kein Datum ist
_df_index_dt = pd.to_datetime(df.index, errors='coerce')
df = df[_df_index_dt.notna()]
# Entferne alle Zeilen, in denen 'Close' nicht numerisch ist
_df_close_num = pd.to_numeric(df['Close'], errors='coerce')
df = df[_df_close_num.notna()]

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
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Kurs", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=df.index, y=df["Smooth"], mode="lines", name="Glättung", line=dict(color="#ff7f0e", dash="dot")))
fig.add_trace(go.Scatter(x=df.index[df["Peak"] == 1], y=df["Close"][df["Peak"] == 1], mode="markers", name="Peaks", marker=dict(color="green", size=10, symbol="triangle-up")))
fig.add_trace(go.Scatter(x=df.index[df["Trough"] == 1], y=df["Close"][df["Trough"] == 1], mode="markers", name="Troughs", marker=dict(color="red", size=10, symbol="triangle-down")))

# Wellenlabel (Impulswellen)
for wave in impulses:
    for i, (idx, price) in enumerate(wave):
        fig.add_annotation(x=df.index[idx], y=price, text=f"{i}", showarrow=True, arrowhead=1, ax=0, ay=-30, font=dict(color="blue"))
# Wellenlabel (Korrekturwellen)
for wave in corrections:
    for i, (idx, price) in enumerate(wave):
        fig.add_annotation(x=df.index[idx], y=price, text=chr(65+i), showarrow=True, arrowhead=1, ax=0, ay=30, font=dict(color="orange"))

# Prognosepfeil (letzter Impuls)
if impulses:
    last = impulses[-1][-1]
    fig.add_annotation(x=df.index[last[0]], y=last[1], ax=df.index[min(last[0]+5, len(df)-1)], ay=last[1]*1.05, text="Prognose", showarrow=True, arrowhead=3, font=dict(color="purple"))

fig.update_layout(height=700, width=1200, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

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