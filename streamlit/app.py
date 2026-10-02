
import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = "/content/mydrive/MyDrive/cryptocast"
STREAMLIT_DIR = os.path.join(BASE_DIR, "streamlit")
MODEL_DIR = os.path.join(STREAMLIT_DIR, "models")
ASSET_DIR = os.path.join(STREAMLIT_DIR, "assets")

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bitcoin_cleaned.csv"
)

FEATURES = [
    "Price", "Open", "High", "Low", "Vol_Log", "Change %"
]
LOOKBACK = 60

ARCHITECTURES = ["CNN", "RNN", "LSTM", "Transformer"]
HORIZONS = [1, 3, 7]

st.set_page_config(
    page_title="CryptoCast | Bitcoin Forecasting",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. CUSTOM TRANSFORMER LAYER
# =====================================================

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]

        positions = tf.cast(
            tf.range(seq_len)[:, tf.newaxis], tf.float32
        )
        dimensions = tf.cast(
            tf.range(d_model)[tf.newaxis, :], tf.float32
        )

        angle_rates = 1.0 / tf.pow(
            10000.0,
            (2 * tf.floor(dimensions / 2))
            / tf.cast(d_model, tf.float32)
        )
        angle_rads = positions * angle_rates

        even_mask = tf.cast(
            tf.range(d_model) % 2 == 0, tf.float32
        )

        encoding = (
            tf.sin(angle_rads) * even_mask
            + tf.cos(angle_rads) * (1 - even_mask)
        )

        return inputs + encoding[tf.newaxis, :, :]

# =====================================================
# 3. PAGE DESIGN
# =====================================================

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 4px;
    }
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 25px;
    }
    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">₿ CryptoCast</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'Multi-Horizon Bitcoin Price Forecasting Using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

# =====================================================
# 4. BANNER IMAGE
# =====================================================

banner_candidates = [
    os.path.join(ASSET_DIR, "bitcoin_banner.png"),
    os.path.join(ASSET_DIR, "bitcoin.png"),
    os.path.join(STREAMLIT_DIR, "bitcoin_banner.png"),
    os.path.join(STREAMLIT_DIR, "banner.png")
]

banner_path = next(
    (p for p in banner_candidates if os.path.exists(p)),
    None
)

if banner_path:
    st.image(banner_path, use_container_width=True)
else:
    st.info(
        "Banner image not found. Place your image at "
        "streamlit/assets/bitcoin_banner.png"
    )

st.markdown(
    "Explore Bitcoin forecasts using four deep learning "
    "architectures across multiple prediction horizons."
)

# =====================================================
# 5. DATA LOADING
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    if "Vol_Log" not in df.columns:
        if "Vol." not in df.columns:
            raise ValueError("Dataset does not contain Vol. column.")
        df["Vol_Log"] = np.log1p(
            pd.to_numeric(df["Vol."], errors="coerce")
        )

    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing feature: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if len(df) < LOOKBACK:
        raise ValueError("Dataset has fewer than 60 valid rows.")

    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

latest_price = float(df["Price"].iloc[-1])

# =====================================================
# 6. MODEL AND SCALER FILE RESOLUTION
# =====================================================

def find_model(architecture, horizon):
    arch = architecture.lower()

    candidates = [
        f"{arch}_{horizon}d.keras",
        f"{arch}_{horizon}day.keras",
        f"{arch}_{horizon}.keras",
        f"{arch}_model_{horizon}d.keras",
        f"{arch}_model_{horizon}.keras"
    ]

    for filename in candidates:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.isfile(path):
            return path

    return None


def find_scaler(architecture, horizon):
    arch = architecture.lower()

    candidates = [
        f"{arch}_scaler_{horizon}d.pkl",
        f"{arch}_scaler_{horizon}.pkl",
        f"{arch}_scaler.pkl",
        f"scaler_{horizon}d.pkl",
        f"scaler_{horizon}.pkl",
        "scaler.pkl"
    ]

    for filename in candidates:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.isfile(path):
            return path

    return None

# =====================================================
# 7. MODEL AND SCALER LOADING
# =====================================================

@st.cache_resource
def load_prediction_model(model_path):
    return load_model(
        model_path,
        custom_objects={
            "PositionalEncoding": PositionalEncoding
        },
        compile=False
    )


@st.cache_resource
def load_scaler(scaler_path):
    return joblib.load(scaler_path)

# =====================================================
# 8. INPUT PREPARATION
# =====================================================

def prepare_input(data, scaler):
    latest = data[FEATURES].tail(LOOKBACK).copy()

    if len(latest) != LOOKBACK:
        raise ValueError("Need exactly 60 historical rows.")

    if hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
        if expected != FEATURES:
            raise ValueError(
                f"Scaler feature order mismatch. "
                f"Expected {expected}, got {FEATURES}"
            )

    scaled = scaler.transform(latest)

    return np.asarray(
        scaled, dtype=np.float32
    ).reshape(1, LOOKBACK, len(FEATURES))

# =====================================================
# 9. PREDICTION
# =====================================================

def generate_prediction(model, input_data, horizon):
    result = model.predict(input_data, verbose=0)
    result = np.asarray(result).reshape(-1)

    if len(result) < horizon:
        raise ValueError(
            f"Model returned {len(result)} values; "
            f"{horizon} required."
        )

    return result[:horizon]


def convert_prediction(raw_values, scaler, output_type, data):
    values = np.asarray(raw_values).reshape(-1)

    if output_type == "Log Return":
        current_price = float(data["Price"].iloc[-1])
        forecast_prices = []

        for value in values:
            current_price *= np.exp(float(value))
            forecast_prices.append(current_price)

        return np.array(forecast_prices)

    if output_type == "Scaled Price":
        if not hasattr(scaler, "inverse_transform"):
            raise ValueError("Scaler has no inverse_transform method.")

        dummy = np.zeros((len(values), len(FEATURES)))
        dummy[:, 0] = values

        inverse = scaler.inverse_transform(dummy)
        return inverse[:, 0]

    if output_type == "Price":
        return values

    raise ValueError("Unsupported output type.")

# =====================================================
# 10. SIDEBAR
# =====================================================

st.sidebar.title("Forecast Settings")
st.sidebar.markdown("Choose a model and forecast horizon.")

architecture = st.sidebar.selectbox(
    "Deep Learning Architecture",
    ARCHITECTURES
)

horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    HORIZONS,
    format_func=lambda x: f"{x} Day{'s' if x > 1 else ''}"
)

output_type = st.sidebar.selectbox(
    "Prediction Output Format",
    ["Log Return", "Scaled Price", "Price"],
    help=(
        "Select the output type used when training this model. "
        "Check the training notebook before choosing."
    )
)

st.sidebar.divider()
st.sidebar.caption("CryptoCast | Research Project")

# =====================================================
# 11. OVERVIEW METRICS
# =====================================================

st.markdown(
    '<div class="section-title">Market Overview</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Latest Bitcoin Price",
        f"${latest_price:,.2f}"
    )

with m2:
    st.metric(
        "Historical Data Points",
        f"{len(df):,}"
    )

with m3:
    st.metric(
        "Input Window",
        f"{LOOKBACK} Days"
    )

# =====================================================
# 12. HISTORICAL PRICE CHART
# =====================================================

st.markdown(
    '<div class="section-title">Historical Bitcoin Price</div>',
    unsafe_allow_html=True
)

history = df.tail(60).copy()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    range(len(history)),
    history["Price"].values,
    label="Historical Price"
)
ax.set_xlabel("Historical Time Steps")
ax.set_ylabel("Bitcoin Price")
ax.set_title("Last 60 Observations")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)
plt.close(fig)

# =====================================================
# 13. FORECAST SECTION
# =====================================================

st.markdown(
    '<div class="section-title">Generate Forecast</div>',
    unsafe_allow_html=True
)

model_path = find_model(architecture, horizon)
scaler_path = find_scaler(architecture, horizon)

if model_path is None:
    st.error(
        f"No model file found for {architecture}, "
        f"{horizon}-day forecast."
    )
    st.caption(
        "Check the actual model filename in the models folder."
    )
    st.stop()

if scaler_path is None:
    st.error(
        f"No scaler found for {architecture}, "
        f"{horizon}-day forecast."
    )
    st.stop()

with st.expander("Selected Files"):
    st.write("Model:", os.path.basename(model_path))
    st.write("Scaler:", os.path.basename(scaler_path))

if st.button("Generate Forecast", type="primary",
             use_container_width=True):

    try:
        with st.spinner("Loading model and generating forecast..."):
            model = load_prediction_model(model_path)
            scaler = load_scaler(scaler_path)

            input_data = prepare_input(df, scaler)

            raw_values = generate_prediction(
                model, input_data, horizon
            )

            forecast_prices = convert_prediction(
                raw_values,
                scaler,
                output_type,
                df
            )

        if not np.all(np.isfinite(forecast_prices)):
            raise ValueError("Forecast contains invalid values.")

        st.success("Forecast generated successfully!")

        forecast_df = pd.DataFrame({
            "Forecast Day": [
                f"Day {i}" for i in range(1, horizon + 1)
            ],
            "Predicted Price": forecast_prices
        })

        # -----------------------------
        # RESULTS TABLE
        # -----------------------------

        st.markdown("### Forecast Values")

        st.dataframe(
            forecast_df.style.format({
                "Predicted Price": "${:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------
        # PRICE CHART
        # -----------------------------

        st.markdown("### Forecast Visualization")

        historical_prices = df["Price"].tail(60).values

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            range(len(historical_prices)),
            historical_prices,
            label="Historical Price"
        )

        future_x = range(
            len(historical_prices),
            len(historical_prices) + horizon
        )

        ax.plot(
            future_x,
            forecast_prices,
            marker="o",
            linestyle="--",
            label="Predicted Price"
        )

        ax.axvline(
            x=len(historical_prices) - 1,
            linestyle=":",
            label="Forecast Start"
        )

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Bitcoin Price")
        ax.set_title(
            f"{architecture} - {horizon}-Day Forecast"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        # -----------------------------
        # DOWNLOAD
        # -----------------------------

        csv_data = forecast_df.to_csv(index=False)

        st.download_button(
            label="Download Forecast CSV",
            data=csv_data,
            file_name=(
                f"{architecture.lower()}_"
                f"{horizon}day_forecast.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.markdown("#### Troubleshooting")
        st.write(
            "- Verify model and scaler compatibility."
        )
        st.write(
            "- Confirm the model input shape."
        )
        st.write(
            "- Confirm the model output format."
        )
        st.write(
            "- Check whether the Transformer uses "
            "the same custom layer implementation."
        )

# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.caption(
    "CryptoCast | Multi-Horizon Bitcoin Price Forecasting "
    "Using Deep Learning"
)
st.caption(
    "This application is for academic and research purposes. "
    "Forecasts are not financial advice."
)
