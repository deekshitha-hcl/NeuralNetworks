from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf


DATA_PATH = Path(__file__).with_name("powerconsumption.csv")
POWER_COLUMNS = {
    "Zone 1": "PowerConsumption_Zone1",
    "Zone 2": "PowerConsumption_Zone2",
    "Zone 3": "PowerConsumption_Zone3",
}
LOOKBACK = 96
HORIZON = 4
FREQUENCY = "15min"

st.set_page_config(
    page_title="Power Forecast | Neural Networks",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #20251f; --muted: #69736b; --paper: #f4f5ef; --line: #dce1d7; --green: #24785b; --lime: #c9e66b; --coral: #cb634e; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: 0 !important; color: var(--ink); }
    h1 { font-size: 2.35rem !important; margin-bottom: .25rem !important; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; }
    [data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; font-size: 1.55rem; }
    [data-testid="stSidebar"] { background: #e9eee4; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h2 { font-size: 1rem !important; }
    .eyebrow { color: var(--green); font: 500 .75rem 'DM Mono', monospace; text-transform: uppercase; }
    .subtitle { color: var(--muted); margin: 0 0 1.5rem; }
    .note { color: var(--muted); font-size: .82rem; }
    .status { display: inline-block; border: 1px solid #a7c8ad; color: #266545; background: #e6f1df; padding: .32rem .6rem; border-radius: 3px; font: 500 .74rem 'DM Mono', monospace; }
    div.stButton > button[kind="primary"] { background: var(--green); border-color: var(--green); }
    div.stButton > button { border-radius: 4px; }
    [data-testid="stAlert"] { border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: str, modified: float) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["Datetime"], date_format="%m/%d/%Y %H:%M")
    raw = raw.sort_values("Datetime").set_index("Datetime")
    numeric = raw.select_dtypes(include="number")
    resampled = numeric.resample(FREQUENCY).mean().interpolate(limit=2)
    return resampled.dropna()


@st.cache_resource(show_spinner=False)
def train_model(values: tuple[float, ...], epochs: int) -> tuple[tf.keras.Model, float, float, float]:
    series = np.asarray(values, dtype=np.float32)
    split = int(len(series) * 0.8)
    data_min = float(series[:split].min())
    data_max = float(series[:split].max())
    scale = max(data_max - data_min, 1e-6)
    normalized = (series - data_min) / scale

    windows = np.lib.stride_tricks.sliding_window_view(normalized, LOOKBACK + HORIZON)
    window_starts = np.arange(len(windows))
    training_stride = 4
    train_mask = (window_starts + LOOKBACK + HORIZON <= split) & (window_starts % training_stride == 0)
    validation_mask = (window_starts + LOOKBACK >= split) & (window_starts % training_stride == 0)
    inputs = windows[:, :LOOKBACK, np.newaxis].astype(np.float32)
    targets = windows[:, LOOKBACK:].astype(np.float32)
    x_train, y_train = inputs[train_mask], targets[train_mask]
    x_valid, y_valid = inputs[validation_mask], targets[validation_mask]

    tf.keras.utils.set_random_seed(41)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOKBACK, 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(HORIZON),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mae")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=256,
        shuffle=False,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    )
    predictions = model.predict(x_valid, verbose=0)
    mae = float(np.mean(np.abs(predictions - y_valid)) * scale)
    return model, data_min, scale, mae


def format_power(value: float) -> str:
    return f"{value:,.0f}"


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as error:
    st.error(f"Could not load the power dataset: {error}")
    st.stop()

st.sidebar.markdown("## Forecast setup")
zone = st.sidebar.selectbox("Load series", ["All zones", *POWER_COLUMNS])
epochs = st.sidebar.slider("Training epochs", min_value=3, max_value=12, value=5)
st.sidebar.markdown("---")
st.sidebar.markdown("### Model input")
st.sidebar.markdown("**96 points · previous 24 hours**")
st.sidebar.markdown("**4 outputs · next 60 minutes**")
st.sidebar.caption("Input and forecast are aligned to 15-minute steps.")

if zone == "All zones":
    series = data[list(POWER_COLUMNS.values())].sum(axis=1)
else:
    series = data[POWER_COLUMNS[zone]]

st.markdown('<p class="eyebrow">Household energy · sequence forecasting</p>', unsafe_allow_html=True)
heading_col, status_col = st.columns([4, 1])
with heading_col:
    st.title("Power Forecast")
    st.markdown(
        f'<p class="subtitle">15-minute load outlook for {zone.lower()} · '
        f'{data.index.min():%b %d, %Y} — {data.index.max():%b %d, %Y}</p>',
        unsafe_allow_html=True,
    )
with status_col:
    st.markdown('<span class="status">HISTORICAL REPLAY</span>', unsafe_allow_html=True)

st.info(
    "This file contains three Istanbul zone loads sampled every 10 minutes, not the UCI "
    "single-household dataset. The model uses their 15-minute averages; source units are "
    "not specified, so values are shown in dataset units rather than assumed kW."
)

if len(series) <= LOOKBACK + HORIZON:
    st.error("There is not enough data to form a 24-hour input window and four forecast steps.")
    st.stop()

max_start = data.index[-HORIZON - 1].to_pydatetime()
min_start = data.index[LOOKBACK - 1].to_pydatetime()
start_at = st.slider(
    "Replay forecast from",
    min_value=min_start,
    max_value=max_start,
    value=max_start,
    step=timedelta(minutes=15),
    format="MMM D, YYYY · HH:mm",
)
start_position = data.index.get_indexer([pd.Timestamp(start_at)], method="nearest")[0]

if "forecast_model_key" not in st.session_state:
    st.session_state.forecast_model_key = None
if st.session_state.forecast_model_key != (zone, epochs):
    st.session_state.forecast_model_key = None

train_col, message_col = st.columns([1, 4])
with train_col:
    train_clicked = st.button("Train LSTM", type="primary", icon="⚡")
if train_clicked:
    st.session_state.forecast_model_key = (zone, epochs)
    with st.spinner("Training on the chronological training split…"):
        model, data_min, scale, validation_mae = train_model(tuple(series), epochs)
        st.session_state.model = model
        st.session_state.data_min = data_min
        st.session_state.scale = scale
        st.session_state.validation_mae = validation_mae

if st.session_state.forecast_model_key != (zone, epochs) or "model" not in st.session_state:
    with message_col:
        st.markdown('<p class="note">Train the model to generate a 60-minute forecast.</p>', unsafe_allow_html=True)
    st.stop()

model = st.session_state.model
data_min = st.session_state.data_min
scale = st.session_state.scale
window = series.iloc[start_position - LOOKBACK + 1 : start_position + 1].to_numpy(dtype=np.float32)
normalized_window = ((window - data_min) / scale).reshape(1, LOOKBACK, 1)
forecast = model.predict(normalized_window, verbose=0)[0] * scale + data_min
forecast = np.maximum(forecast, 0)
forecast_times = pd.date_range(pd.Timestamp(start_at) + pd.Timedelta(minutes=15), periods=HORIZON, freq=FREQUENCY)
current_value = float(series.iloc[start_position])
today = series.loc[series.index.date == pd.Timestamp(start_at).date()]
daily_consumption = float(today.sum() * 0.25)
daily_peak = float(today.max())
recent_week = series.loc[:pd.Timestamp(start_at)].tail(7 * 24 * 4)
warning_threshold = float(recent_week.quantile(0.9))
peak_forecast = float(forecast.max())

with message_col:
    st.markdown(
        f'<p class="note">Validation MAE: {format_power(st.session_state.validation_mae)} dataset units · '
        f'forecast anchored at {pd.Timestamp(start_at):%b %d, %Y %H:%M}</p>',
        unsafe_allow_html=True,
    )

metrics = st.columns(5)
metrics[0].metric("Current load", format_power(current_value), "dataset units")
metrics[1].metric("Next 15 min", format_power(float(forecast[0])), "dataset units")
metrics[2].metric("Next 60 min", format_power(float(forecast[-1])), "dataset units")
metrics[3].metric("Daily consumption", format_power(daily_consumption), "unit·h, estimated")
metrics[4].metric("Today's peak", format_power(daily_peak), "dataset units")

chart_col, detail_col = st.columns([2.2, 1])
with chart_col:
    st.subheader("Next-hour forecast")
    history = series.iloc[max(0, start_position - LOOKBACK + 1) : start_position + 1]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history.index,
            y=history,
            mode="lines",
            name="Observed",
            line={"color": "#24785b", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[pd.Timestamp(start_at), *forecast_times],
            y=[current_value, *forecast],
            mode="lines+markers",
            name="LSTM forecast",
            line={"color": "#cb634e", "width": 2.5, "dash": "dash"},
            marker={"size": 7},
        )
    )
    figure.add_vline(x=pd.Timestamp(start_at), line_dash="dot", line_color="#a3aaa0")
    figure.update_layout(
        height=370,
        margin={"l": 8, "r": 12, "t": 12, "b": 8},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "DM Sans, sans-serif", "color": "#20251f"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={"title": None, "showgrid": False},
        yaxis={"title": "Dataset power units", "gridcolor": "#edf0e9", "zeroline": False},
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch")

with detail_col:
    st.subheader("Forecast steps")
    forecast_table = pd.DataFrame(
        {"Time": forecast_times.strftime("%H:%M"), "Predicted load": [format_power(float(value)) for value in forecast]}
    )
    st.dataframe(forecast_table, hide_index=True, width="stretch")
    st.markdown("#### Peak check")
    if peak_forecast > warning_threshold:
        st.warning(
            f"Forecast peak ({format_power(peak_forecast)}) exceeds the recent 7-day 90th percentile "
            f"({format_power(warning_threshold)}). Consider shifting flexible loads away from this hour."
        )
    else:
        st.success("The next-hour forecast is below the recent 7-day high-use threshold.")
    if float(forecast.mean()) > current_value * 1.1:
        recommendation = "Load is trending upward. Delay flexible appliances until the forecast settles."
    elif float(forecast.mean()) < current_value * 0.9:
        recommendation = "Load is trending down. This is a lower-demand window for flexible tasks."
    else:
        recommendation = "Load is steady. Review overnight baseload for always-on devices."
    st.caption("Energy note")
    st.write(recommendation)

st.markdown('<p class="note">Daily consumption is the sum of 15-minute mean load × 0.25 hours. The dataset does not state its power unit.</p>', unsafe_allow_html=True)