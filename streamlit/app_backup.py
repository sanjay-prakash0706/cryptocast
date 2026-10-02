
import streamlit as st
import base64
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CryptoCast",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/content/mydrive/MyDrive/cryptocast/streamlit"

BACKGROUND_IMAGE = os.path.join(
    BASE_DIR,
    "assets",
    "bitcoin_banner.png"
)

# ============================================================
# BACKGROUND
# ============================================================

def set_background(image_path):

    if os.path.exists(image_path):

        with open(image_path, "rb") as f:
            encoded = base64.b64encode(
                f.read()
            ).decode()

        st.markdown(
            f"""
            <style>

            .stApp {{
                background-image:
                    linear-gradient(
                        rgba(255,255,255,0.82),
                        rgba(245,248,252,0.94)
                    ),
                    url("data:image/png;base64,{encoded}");

                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )


set_background(BACKGROUND_IMAGE)

# ============================================================
# LIGHT UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #4b5563;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.94);
        border-radius: 18px;
        padding: 28px;
        margin: 18px 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }

    .card h2 {
        color: #111827;
    }

    .card p {
        color: #4b5563;
        font-size: 16px;
    }

    /* Metric */
    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 12px;
        font-size: 17px;
        font-weight: 700;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: white;
        border-radius: 10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">₿ CryptoCast</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multi-Horizon Bitcoin Price Forecasting Using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Forecast Settings")

    horizon = st.selectbox(
        "Select Forecast Horizon",
        [
            "1-Day",
            "3-Day",
            "7-Day"
        ]
    )

    st.markdown("---")

    st.write("### 🧠 Model")

    st.info(
        "Transformer-based Deep Learning"
    )

    st.markdown("---")

    st.write("### 📥 Input Data")

    st.caption(
        "The model automatically uses the "
        "latest 60 days of historical Bitcoin data."
    )

# ============================================================
# WELCOME CARD
# ============================================================

st.markdown(
    """
    <div class="card">

    <h2>🚀 Welcome to CryptoCast</h2>

    <p>
    CryptoCast uses deep learning to forecast Bitcoin prices
    across multiple time horizons.
    </p>

    <p>
    <b>Available Forecasts:</b>
    &nbsp; 1-Day &nbsp; | &nbsp; 3-Day &nbsp; | &nbsp; 7-Day
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# INPUT / FORECAST CONTROL
# ============================================================

st.markdown(
    """
    <div class="card">

    <h2>🔮 Generate Bitcoin Forecast</h2>

    <p>
    Choose a forecast horizon and generate a prediction
    using the Transformer model.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

with col1:

    selected_horizon = st.selectbox(
        "📅 Forecast Horizon",
        [
            "1-Day",
            "3-Day",
            "7-Day"
        ],
        index=["1-Day", "3-Day", "7-Day"].index(horizon)
    )

with col2:

    st.write("")
    st.write("")

    generate = st.button(
        "🚀 Generate Forecast",
        use_container_width=True
    )

# ============================================================
# INPUT INFORMATION
# ============================================================

st.markdown(
    """
    <div class="card">

    <h3>📊 Model Input</h3>

    <p>
    <b>Historical Window:</b> Previous 60 days
    </p>

    <p>
    <b>Features:</b>
    Price, Open, High, Low, Volume, Change %
    </p>

    <p>
    The application automatically prepares the required
    input sequence from the historical dataset.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# FORECAST RESULT PLACEHOLDER
# ============================================================

if generate:

    st.markdown(
        """
        <div class="card">
        <h2>📈 Forecast Result</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "₿ Current Price",
            "$--"
        )

    with col2:
        st.metric(
            "🔮 Predicted Price",
            "$--"
        )

    with col3:
        st.metric(
            "📊 Expected Change",
            "--"
        )

    st.info(
        f"Transformer forecast selected: {selected_horizon}"
    )

else:

    st.info(
        "Select a forecast horizon and click "
        "**Generate Forecast** to view the prediction."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        padding:30px;
        color:#6b7280;
    ">
        ₿ CryptoCast • Multi-Horizon Bitcoin Price Forecasting
    </div>
    """,
    unsafe_allow_html=True
)
