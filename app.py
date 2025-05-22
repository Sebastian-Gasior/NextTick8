# Platzhalter für die Streamlit-App 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
from src import data_loader, data_validation, peak_detection, elliott_wave
from datetime import datetime, timedelta
import src.ml.feature_engineering as ml_feat
import src.ml.model_training as ml_train
import src.ml.metrics as ml_metrics
import src.ml.plots as ml_plots
import src.ta as ta
import src.performance as perf
import src.ml.backtrader_engine as bt_engine
import hashlib
import json
import joblib

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
    st.rerun()

try:
    # --- Daten laden (zentral, für alle Tabs) ---
    raw_path = f"data/raw/{selected_ticker}.csv"
    cleaned_path = f"data/cleaned/{selected_ticker}.csv"
    df_all = pd.read_csv(cleaned_path, index_col=0)
    df_all.index = pd.to_datetime(df_all.index, format="%Y-%m-%d", errors="coerce")

    # --- Tabs wie in den Beispielbildern, plus Elliott-Wave-Stock-Analyzer ---
    tabs = st.tabs(["Elliott-Wave-Stock-Analyzer", "Preise & Indikatoren", "TA Signale", "ML Vorhersagen", "Strategie-Vergleich"])

    # --- Tab 1: Elliott-Wave-Stock-Analyzer ---
    with tabs[0]:
        st.header("Elliott-Wave-Stock-Analyzer")
        # Zeitraum filtern (Kopie erzeugen, um SettingWithCopyWarning zu vermeiden)
        df = df_all[(df_all.index.date >= start_date) & (df_all.index.date <= end_date)].copy()
        # Glättung & Peak/Trough
        smooth = peak_detection.smooth_series(df["Close"], method=smoothing, window=window, polyorder=polyorder)
        peaks_troughs = peak_detection.detect_peaks_and_troughs(smooth, prominence=prominence, distance=distance)
        df.loc[:, "Smooth"] = smooth
        df.loc[:, "Peak"] = 0
        df.loc[:, "Trough"] = 0
        df.iloc[peaks_troughs["peaks"], df.columns.get_loc("Peak")] = 1
        df.iloc[peaks_troughs["troughs"], df.columns.get_loc("Trough")] = 1
        # Elliott-Wellen
        turning_points = elliott_wave.extract_turning_points(df)
        impulses = elliott_wave.find_impulse_waves(turning_points)
        corrections = elliott_wave.find_correction_waves(turning_points)
        # Plotly-Visualisierung (Kurs, Glättung, Peaks, Troughs, Prognose, Zukunft)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Kurs", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Smooth"], mode="lines", name="Glättung", line=dict(color="#ff7f0e", dash="dot")))
        fig.add_trace(go.Scatter(x=df.index[df["Peak"] == 1], y=df["Close"][df["Peak"] == 1], mode="markers", name="Peaks", marker=dict(color="green", size=10, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=df.index[df["Trough"] == 1], y=df["Close"][df["Trough"] == 1], mode="markers", name="Troughs", marker=dict(color="red", size=10, symbol="triangle-down")))
        # --- Graue Linie: Zukunft (echt) ---
        prognose_ende = end_date + timedelta(days=prognose_tage)
        df_future = df_all[(df_all.index.date > end_date) & (df_all.index.date <= prognose_ende)]
        if not df_future.empty:
            fig.add_trace(go.Scatter(x=df_future.index, y=df_future["Close"], mode="lines", name="Zukunft (echt)", line=dict(color="gray", width=2, dash="dot"), opacity=0.5))
        # --- Prognosepfeile (wie im alten Stand) ---
        if not df.empty:
            peak_indices = df.index[df["Peak"] == 1].tolist()
            if peak_indices:
                last_peak_idx = peak_indices[-1]
                last_peak_val = df.loc[last_peak_idx, "Close"]
                # Korrekturwellen-Analyse
                last_corrections = [wave for wave in corrections if wave[-1][0] < len(df)]
                if len(last_corrections) >= 2:
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
                corr_idx = df.index[-1] + timedelta(days=avg_duration)
                corr_val = last_peak_val * (1 - avg_depth)
                # Erholungswellen-Analyse
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
                recov_idx = corr_idx + timedelta(days=avg_recov_duration)
                recov_val = corr_val * (1 + avg_gain)
                # --- Prognosepfeile und Linien zeichnen ---
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
        fig.update_layout(height=600, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        # Export (optional)
        export_dir = "data/exported"
        os.makedirs(export_dir, exist_ok=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Plot als PNG speichern", key="ewave_png"):
                png_path = os.path.join(export_dir, f"{selected_ticker}_ewave_plot.png")
                fig.write_image(png_path)
                with open(png_path, "rb") as f:
                    st.download_button("Download PNG", f, file_name=f"{selected_ticker}_ewave_plot.png")
        with col2:
            if st.button("Plot als HTML speichern", key="ewave_html"):
                html_path = os.path.join(export_dir, f"{selected_ticker}_ewave_plot.html")
                fig.write_html(html_path)
                with open(html_path, "rb") as f:
                    st.download_button("Download HTML", f, file_name=f"{selected_ticker}_ewave_plot.html")
        # Legende/Hilfe
        with st.expander("ℹ️ Hilfe & Legende anzeigen", expanded=True):
            st.markdown("""
            **Kurs:** Originaler Schlusskurs der Aktie (blaue Linie)
            **Glättung:**
            - **ma**: Moving Average (gleitender Durchschnitt)
            - **savgol**: Savitzky-Golay-Filter (polynomiale Glättung)
            **Peaks:** Lokale Hochpunkte (grüne Dreiecke)
            **Troughs:** Lokale Tiefpunkte (rote Dreiecke)
            **Parameter:**
            - **Glättungsfenster:** Größe des Fensters für die Glättung (Anzahl Tage)
            - **Savitzky-Golay Ordnung:** Grad des Polynoms für Savitzky-Golay
            - **Peak-Prominenz:** Mindesthöhe, damit ein Peak erkannt wird
            - **Peak-Mindestabstand:** Mindestabstand zwischen Peaks
            - **Start-/Enddatum:** Zeitraum für die Analyse
            - **Prognosezeitraum:** Länge der Prognose in Tagen
            """)
        # Wellen-Tabellen
        st.subheader("Gefundene Impulswellen (5-teilig)")
        for wave in impulses:
            st.write([f"{df.index[idx].date()} ({price:.2f})" for idx, price in wave])
        st.subheader("Gefundene Korrekturwellen (ABC)")
        for wave in corrections:
            st.write([f"{df.index[idx].date()} ({price:.2f})" for idx, price in wave])

    # --- Tab 2: Preise & Indikatoren ---
    with tabs[1]:
        st.header("Preis & Volumen")
        # --- Feature Engineering & Export ---
        df_features = ml_feat.add_technical_indicators(df_all)
        ml_feat.export_features(df_features, selected_ticker)
        st.success(f"Features für {selected_ticker} wurden berechnet und exportiert.")
        # --- Bisheriger Preis-Plot ---
        fig = ml_plots.plot_candlestick_volume(df_all)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Performance-Metriken")
        returns = np.diff(df_all["Close"].values, prepend=df_all["Close"].values[0])
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252)
        total_return = df_all["Close"].values[-1] / df_all["Close"].values[0] - 1 if len(df_all) > 1 else 0
        st.write({"Total Return": round(total_return, 3), "Volatility": round(volatility, 2), "Sharpe Ratio": round(sharpe, 2)})
        # Hilfe & Legende für Preise & Volumen
        with st.expander("ℹ️ Hilfe & Legende", expanded=True):
            st.markdown("""
            **Werte und Bereiche:**
            - **Preis & Volumen:** Candlestick-Chart zeigt Tageskurse (Open, High, Low, Close) und Handelsvolumen.
            - **Zeitraum:** Entspricht dem gewählten Analysezeitraum.

            **Performance-Metriken:**
            - **Total Return:** Gesamtrendite über den Zeitraum (in Prozent)
            - **Volatility:** Schwankungsbreite der Tagesrenditen (annualisiert)
            - **Sharpe Ratio:** Verhältnis von Rendite zu Risiko (je höher, desto besser; >1 gilt als gut)
            """)

    # --- Tab 3: TA Signale ---
    with tabs[2]:
        st.header("Technische Analyse")
        # --- Feature-Visualisierung (Matplotlib & Plotly) ---
        st.subheader("Feature-Visualisierung (Matplotlib)")
        st.pyplot(ml_feat.plot_features_matplotlib(df_features, selected_ticker))
        st.subheader("Feature-Visualisierung (Plotly)")
        fig_feat = ml_feat.plot_features_plotly(df_features, selected_ticker)
        st.plotly_chart(fig_feat, use_container_width=True)
        # --- TA-Plot ---
        df_ta = ta.add_ta_indicators(df_all)
        fig_ta = ta.plot_ta_signals(df_ta)
        st.plotly_chart(fig_ta, use_container_width=True)
        # Export-Buttons für TA-Plot
        col_ta1, col_ta2 = st.columns(2)
        with col_ta1:
            if st.button("TA-Plot als PNG speichern", key="ta_png"):
                png_path = os.path.join(export_dir, f"{selected_ticker}_ta_plot.png")
                fig_ta.write_image(png_path)
                with open(png_path, "rb") as f:
                    st.download_button("Download TA-PNG", f, file_name=f"{selected_ticker}_ta_plot.png")
        with col_ta2:
            if st.button("TA-Plot als HTML speichern", key="ta_html"):
                html_path = os.path.join(export_dir, f"{selected_ticker}_ta_plot.html")
                fig_ta.write_html(html_path)
                with open(html_path, "rb") as f:
                    st.download_button("Download TA-HTML", f, file_name=f"{selected_ticker}_ta_plot.html")
        fig_rsi = ta.plot_rsi(df_ta)
        st.plotly_chart(fig_rsi, use_container_width=True)
        # Export-Buttons für RSI-Plot
        col_rsi1, col_rsi2 = st.columns(2)
        with col_rsi1:
            if st.button("RSI-Plot als PNG speichern", key="rsi_png"):
                png_path = os.path.join(export_dir, f"{selected_ticker}_rsi_plot.png")
                fig_rsi.write_image(png_path)
                with open(png_path, "rb") as f:
                    st.download_button("Download RSI-PNG", f, file_name=f"{selected_ticker}_rsi_plot.png")
        with col_rsi2:
            if st.button("RSI-Plot als HTML speichern", key="rsi_html"):
                html_path = os.path.join(export_dir, f"{selected_ticker}_rsi_plot.html")
                fig_rsi.write_html(html_path)
                with open(html_path, "rb") as f:
                    st.download_button("Download RSI-HTML", f, file_name=f"{selected_ticker}_rsi_plot.html")
        st.subheader("Performance-Metriken")
        returns = np.diff(df_all["Close"].values, prepend=df_all["Close"].values[0])
        trades = returns  # Annahme: jede Änderung = Trade
        st.write(perf.compute_performance_metrics(df_all["Close"].values, returns, trades))
        # Hilfe & Legende für TA-Plot
        with st.expander("ℹ️ Hilfe & Legende", expanded=True):
            st.markdown("""
            **Linien und Farben:**
            - **Close (lila):** Schlusskurs der Aktie
            - **SMA_20, SMA_50, SMA_200:** Gleitende Durchschnitte (20, 50, 200 Tage)
            - **EMA_12, EMA_26:** Exponentiell gewichtete Durchschnitte
            - **RSI (magenta):** Relative Strength Index (Oszillator, 0-100)
            - **Bereiche:** Der sichtbare Zeitraum entspricht dem gewählten Analysezeitraum
            - **Interpretation:**
                - SMA/EMA zeigen Trendrichtungen
                - RSI > 70: Überkauft, RSI < 30: Überverkauft

            **Performance-Metriken:**
            - **Total Return:** Gesamtrendite über den betrachteten Zeitraum (in Prozent)
            - **Number of Trades:** Anzahl der Trades (Handelsentscheidungen) im Zeitraum
            - **Win Rate:** Anteil der Gewinn-Trades in Prozent
            - **Avg Profit:** Durchschnittlicher Gewinn pro Gewinn-Trade
            - **Avg Loss:** Durchschnittlicher Verlust pro Verlust-Trade
            - **Profit Factor:** Verhältnis aller Gewinne zu allen Verlusten (>1 ist profitabel)
            - **Max Drawdown:** Maximaler Kapitalrückgang vom Höchststand (negativ, z.B. -0.82 = -82%)
            - **Volatility:** Schwankungsbreite der Renditen (annualisiert)
            - **Sharpe Ratio:** Rendite-Risiko-Verhältnis (je höher, desto besser; >1 gilt als gut)
            """)

    # --- Tab 4: ML Vorhersagen ---
    with tabs[3]:
        st.header("ML Vorhersagen")
        ml_window = st.number_input("LSTM-Fenstergröße (Tage)", min_value=5, max_value=60, value=20, key="ml_window")
        ml_epochs = st.number_input("Trainings-Epochen", min_value=1, max_value=50, value=10, key="ml_epochs")
        ml_dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05, key="ml_dropout")
        ml_pred_steps = st.number_input("Prognoselänge (Tage)", min_value=1, max_value=60, value=30, key="ml_pred_steps")
        use_tuning = st.checkbox("Hyperparameter-Tuning (Randomized Search)")
        retrain = st.checkbox("Modell neu trainieren (überschreibt vorhandenes)")
        model_loaded = False
        model, scaler, log = None, None, None
        # --- Eindeutigen Hash für Modellkonfiguration erzeugen ---
        config_str = f"{selected_ticker}_{ml_window}_{ml_epochs}_{ml_dropout:.3f}_{use_tuning}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        export_dir = "data/exported"
        forecast_path = os.path.join(export_dir, f"{selected_ticker}_ml_forecast_{config_hash}.csv")
        log_path = os.path.join(export_dir, f"{selected_ticker}_lstm_trainlog_{config_hash}.json")
        model_path = os.path.join(export_dir, f"{selected_ticker}_lstm_model_{config_hash}.keras")
        scaler_path = os.path.join(export_dir, f"{selected_ticker}_scaler_{config_hash}.joblib")
        # --- Modell und Prognose laden, falls vorhanden und nicht retrain ---
        if not retrain and os.path.exists(forecast_path) and os.path.exists(log_path) and os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                forecast_df = pd.read_csv(forecast_path)
                with open(log_path, "r") as f:
                    log = json.load(f)
                model = ml_train.load_lstm_model(selected_ticker, export_dir=export_dir)
                scaler = joblib.load(scaler_path)
                model_loaded = True
                st.info(f"Vorhandene Prognose und Modell für {selected_ticker} (Konfiguration: {config_hash}) geladen.")
                # Visualisierung Prognose
                st.plotly_chart(ml_plots.plot_ml_prediction(df_all["Close"].values[-len(forecast_df):], forecast_df["Forecast"].values, index=forecast_df["Date"]), use_container_width=True)
                # Loss-Kurve
                if log is not None and 'train_loss' in log and 'val_loss' in log:
                    st.subheader("Trainingsverlauf (Loss)")
                    fig_loss = ml_train.plot_training_history(log)
                    st.plotly_chart(fig_loss, use_container_width=True)
                # Download-Button
                with open(forecast_path, "rb") as f:
                    st.download_button("Download ML-Prognose (CSV)", f, file_name=os.path.basename(forecast_path))
                # Performance-Metriken
                st.subheader("Performance-Metriken")
                preds = forecast_df["Forecast"].values
                returns = np.diff(preds, prepend=preds[0])
                trades = returns
                st.write(perf.compute_performance_metrics(preds, returns, trades))
            except Exception as e:
                st.error(f"Fehler beim Laden der gespeicherten Prognose: {e}")
                model_loaded = False
        # --- Button: Prognose berechnen ---
        if st.button("LSTM-Prognose berechnen", key="ml_predict") or (not model_loaded and not retrain):
            try:
                with st.spinner("Trainiere oder lade LSTM-Modell..."):
                    # Prognose-Zeitraum: Test-Set
                    n = len(df_all)
                    min_days = 500
                    if n > min_days:
                        df_ml = df_all.copy().iloc[-min_days:]
                    else:
                        df_ml = df_all.copy()
                    train_end = int(len(df_ml) * 0.7)
                    val_end = int(len(df_ml) * 0.85)
                    test_start = df_ml.index[val_end].date()
                    test_end = df_ml.index[-1].date()
                    st.info(f"Prognosezeitraum: {test_start} bis {test_end} (Test-Set)")
                    if not model_loaded or retrain:
                        if use_tuning:
                            param_dist = {"epochs": [5, 10, 20, 30], "dropout": [0.1, 0.2, 0.3, 0.4]}
                            model, scaler, log = ml_train.random_search_lstm(df_ml, param_dist, n_iter=6, window=ml_window, export_dir=export_dir, ticker=selected_ticker)
                        else:
                            model, scaler, log = ml_train.train_lstm_with_val(df_ml, window=ml_window, epochs=ml_epochs, dropout=ml_dropout, export_dir=export_dir, ticker=selected_ticker)
                        # Nach dem Training (direkt vor model.save(...))
                        if model is None or scaler is None or log is None:
                            st.error("Fehler: Es konnte kein valides Modell gefunden werden. Bitte prüfe die Datenbasis und Trainingsparameter.")
                            st.stop()
                        # Speichere Modell, Scaler, Log
                        model.save(model_path)
                        joblib.dump(scaler, scaler_path)
                        with open(log_path, "w") as f:
                            json.dump(log, f, indent=2)
                    # Daten-Checks vor ML-Training
                    if (df_ml['Close'] <= 0).any():
                        st.error("Fehler: Die Preisdaten enthalten Nullen oder negative Werte. Bitte Daten prüfen und ggf. neu laden!")
                        st.stop()
                    if df_ml['Close'].isna().any():
                        st.error("Fehler: Die Preisdaten enthalten NaN-Werte. Bitte Daten prüfen und ggf. neu laden!")
                        st.stop()
                    if len(df_ml) < ml_window + 10:
                        st.error(f"Fehler: Zu wenig Daten für die gewählte Fenstergröße ({ml_window}). Bitte Fenster verkleinern oder mehr Daten verwenden.")
                        st.stop()
                    close_scaled = scaler.transform(df_ml["Close"].values.reshape(-1, 1)).flatten()
                    test_len = len(df_ml) - val_end
                    start_price = df_ml["Close"].values[val_end-1] if val_end > 0 else df_ml["Close"].values[0]
                    # Prognose mit true_prices-Vergleich (Debugging)
                    true_prices = df_ml["Close"].values[val_end:]
                    preds = ml_train.predict_lstm(model, scaler, close_scaled, window=ml_window, steps=test_len, start_price=start_price, return_prices=True, true_prices=true_prices, log_compare=True)
                    # Speichere Prognose
                    forecast_df = pd.DataFrame({"Date": df_ml.index[val_end:], "Forecast": preds})
                    forecast_df.to_csv(forecast_path, index=False)
                    st.success(f"Neue Prognose und Modell für {selected_ticker} (Konfiguration: {config_hash}) gespeichert.")
                    # Visualisierung Prognose
                    st.plotly_chart(ml_plots.plot_ml_prediction(true_prices, preds, index=df_ml.index[val_end:]), use_container_width=True)
                    # Loss-Kurve
                    if log is not None and 'train_loss' in log and 'val_loss' in log:
                        st.subheader("Trainingsverlauf (Loss)")
                        fig_loss = ml_train.plot_training_history(log)
                        st.plotly_chart(fig_loss, use_container_width=True)
                    # Download-Button
                    with open(forecast_path, "rb") as f:
                        st.download_button("Download ML-Prognose (CSV)", f, file_name=os.path.basename(forecast_path))
                    # Performance-Metriken
                    st.subheader("Performance-Metriken")
                    returns = np.diff(preds, prepend=preds[0])
                    trades = returns
                    st.write(perf.compute_performance_metrics(preds, returns, trades))
            except Exception as ml_e:
                st.error(f"Unerwarteter Fehler bei der LSTM-Prognose: {ml_e}")
                st.stop()

    # --- Tab 5: Strategie-Vergleich ---
    with tabs[4]:
        st.header("Strategie-Vergleich & Backtesting")
        # --- Backtest-Parameter ---
        st.subheader("Backtest-Parameter")
        col1, col2, col3 = st.columns(3)
        with col1:
            commission = st.number_input("Kommission (%)", min_value=0.0, max_value=0.05, value=0.001, step=0.0005, format="%.4f")
        with col2:
            slippage = st.number_input("Slippage (%)", min_value=0.0, max_value=0.05, value=0.001, step=0.0005, format="%.4f")
        with col3:
            size = st.number_input("Positionsgröße", min_value=1, max_value=1000, value=1, step=1)
        # --- TA-Strategie: SMA/EMA-Crossover ---
        df_ta = ta.add_ta_indicators(df_all)
        ta_signal = (df_ta['SMA_20'] > df_ta['SMA_50']).astype(int)
        # --- ML-Strategie: Rolling/Walk-Forward-Backtest ---
        ml_signal = ml_train.generate_ml_signals(df_all, selected_ticker)
        # --- Backtest-Buttons ---
        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            if st.button("TA-Strategie Backtest starten"):
                ta_report = bt_engine.run_backtest(df_all, ta_signal, strategy_name="ta", commission=commission, slippage=slippage, size=size)
                st.session_state["ta_report"] = ta_report
        with col_bt2:
            if st.button("ML-Strategie Backtest starten"):
                ml_report = bt_engine.run_backtest(df_all, ml_signal, strategy_name="ml", commission=commission, slippage=slippage, size=size)
                st.session_state["ml_report"] = ml_report
        # --- Ergebnisse anzeigen, wenn vorhanden ---
        if "ta_report" in st.session_state and "ml_report" in st.session_state:
            st.subheader("Equity-Kurven (Overlay)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=st.session_state["ta_report"]["date"], y=st.session_state["ta_report"]["equity"], mode="lines", name="TA-Strategie"))
            fig.add_trace(go.Scatter(x=st.session_state["ml_report"]["date"], y=st.session_state["ml_report"]["equity"], mode="lines", name="ML-Strategie"))
            st.plotly_chart(fig, use_container_width=True)
            # --- Performance-Kennzahlen ---
            st.subheader("Performance-Kennzahlen")
            def calc_stats(report):
                returns = pd.Series(report["equity"]).pct_change().fillna(0)
                sharpe = returns.mean() / (returns.std() + 1e-9) * (252 ** 0.5)
                drawdown = (report["equity"] / report["equity"].cummax() - 1).min()
                cagr = (report["equity"].iloc[-1] / report["equity"].iloc[0]) ** (252/len(report)) - 1
                return dict(Sharpe=sharpe, Drawdown=drawdown, CAGR=cagr)
            ta_stats = calc_stats(st.session_state["ta_report"])
            ml_stats = calc_stats(st.session_state["ml_report"])
            st.table(pd.DataFrame([ta_stats, ml_stats], index=["TA", "ML"]))
            # --- Download-Links ---
            st.subheader("Backtest-Reports Download")
            ta_csv = f"backtest/reports/ta_backtest.csv"
            ml_csv = f"backtest/reports/ml_backtest.csv"
            if os.path.exists(ta_csv):
                st.download_button("TA-Report (CSV)", open(ta_csv, "rb"), file_name="ta_backtest.csv")
            if os.path.exists(ml_csv):
                st.download_button("ML-Report (CSV)", open(ml_csv, "rb"), file_name="ml_backtest.csv")
except Exception as e:
    st.error(f"Fehler im Haupt-Workflow: {e}") 