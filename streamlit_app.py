import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
from pathlib import Path

from src.features import run_feature_pipeline, TARGET_COL

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Obsidian-Style Dark Theme CSS
st.markdown("""
<style>
    /* Main body background and font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Background */
    .stApp {
        background-color: #0E1117;
        color: #E2E8F0;
    }
    
    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(25, 29, 38, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(0, 242, 254, 0.3);
    }
    
    /* Header Gradient Text */
    .gradient-text {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    .subheader-text {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Prediction Large Text */
    .predicted-fare {
        font-size: 3.5rem;
        font-weight: 800;
        color: #00f2fe;
        text-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Coordinate Dictionary for NYC Taxi Zones ──────────────────────────
# Approximate Latitude/Longitude mapping for popular NYC taxi zones
COORD_LOOKUP = {
    1: (40.6895, -74.1745),   # Newark Airport (EWR)
    132: (40.6413, -73.7781), # JFK Airport
    138: (40.7769, -73.8740), # LGA Airport
    43: (40.7829, -73.9654),  # Central Park
    230: (40.7580, -73.9855), # Times Square / Theatre District
    186: (40.7506, -73.9935), # Penn Station / Union Sq West
    162: (40.7527, -73.9772), # Grand Central / Murray Hill
    263: (40.7061, -74.0092), # Wall Street / Financial District
    100: (40.7538, -73.9904), # Garment District
    140: (40.7736, -73.9566), # Lenox Hill East
    141: (40.7781, -73.9515), # Lenox Hill West
    236: (40.7801, -73.9538), # Upper East Side North
    237: (40.7736, -73.9641), # Upper East Side South
    238: (40.8033, -73.9674), # Upper West Side North
    239: (40.7879, -73.9799), # Upper West Side South
    142: (40.7750, -73.9818), # Lincoln Square
    48: (40.7618, -73.9984),  # Clinton East
    246: (40.7513, -74.0044), # Chelsea West
    68: (40.7431, -73.9973),  # Chelsea East
    79: (40.7289, -73.9892),  # East Village
    107: (40.7324, -73.9873), # Gramercy
    170: (40.7454, -73.9782), # Murray Hill
    90: (40.7405, -74.0071),  # Flatiron / Union Sq
    234: (40.7402, -73.9896), # Union Sq
    113: (40.7325, -73.9973), # Greenwich Village North
    114: (40.7291, -74.0012), # Greenwich Village South
    125: (40.7247, -74.0084), # Hudson Sq
    249: (40.7346, -74.0062), # West Village
    231: (40.7202, -74.0102), # TriBeCa
    148: (40.7161, -73.9912), # Lower East Side
    224: (40.7214, -73.9798), # Stuyvesant Town/Peter Cooper Village
    262: (40.7757, -73.9431), # Yorkville East
    261: (40.7075, -74.0113), # World Trade Center
}

# Borough Centroid Fallbacks
BOROUGH_CENTROIDS = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
    "EWR": (40.6895, -74.1745),
    "Unknown": (40.7831, -73.9712)
}

def get_coordinates(location_id, borough_name):
    """Retrieve Latitude/Longitude with deterministic jitter for fallback centroids."""
    if location_id in COORD_LOOKUP:
        return COORD_LOOKUP[location_id]
    
    # Fallback to Borough centroid with jitter
    centroid = BOROUGH_CENTROIDS.get(borough_name, BOROUGH_CENTROIDS["Unknown"])
    np.random.seed(int(location_id))
    jitter_lat = np.random.uniform(-0.015, 0.015)
    jitter_lon = np.random.uniform(-0.015, 0.015)
    return (centroid[0] + jitter_lat, centroid[1] + jitter_lon)

# ── Load Metadata & Models ───────────────────────────────────────────────────

@st.cache_resource
def load_assets():
    """Load metadata lookup, fitted scaler, and trained champion model."""
    lookup_path = Path("notebooks/taxi_zone_lookup.csv")
    scaler_path = Path("models/engineered/scaler.pkl")
    model_path  = Path("models/engineered/gradient_boosting.pkl")
    
    if not lookup_path.exists():
        st.error(f"Metadata file lookup missing at `{lookup_path}`.")
        st.stop()
    
    if not scaler_path.exists() or not model_path.exists():
        st.error("Engineered scaler or model not found. Please run `pipeline.py` first.")
        st.stop()
        
    lookup_df = pd.read_csv(lookup_path)
    scaler    = joblib.load(scaler_path)
    model     = joblib.load(model_path)
    
    return lookup_df, scaler, model

lookup_df, scaler, model = load_assets()

# Format zone names for UI selection dropdowns
lookup_df["DropdownName"] = lookup_df["Borough"] + " - " + lookup_df["Zone"]
zone_options = lookup_df.sort_values("DropdownName").to_dict("records")

# ── Main Layout ───────────────────────────────────────────────────────────────

st.markdown('<div class="gradient-text">NYC Taxi Fare Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Instantly estimate yellow cab fares across New York City using GBDT machine learning</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📍 Journey Details")
    
    # Pickup selection
    pickup_selection = st.selectbox(
        "Pickup Location (From)",
        options=zone_options,
        format_func=lambda x: x["DropdownName"],
        index=next((i for i, x in enumerate(zone_options) if x["LocationID"] == 230), 0) # Times Square
    )
    
    # Dropoff selection
    dropoff_selection = st.selectbox(
        "Dropoff Location (To)",
        options=zone_options,
        format_func=lambda x: x["DropdownName"],
        index=next((i for i, x in enumerate(zone_options) if x["LocationID"] == 132), 0) # JFK Airport
    )
    
    # Trip distance
    trip_distance = st.slider(
        "Estimated Trip Distance (miles)",
        min_value=0.1,
        max_value=40.0,
        value=13.5,
        step=0.1,
        help="Metering distance. Airport trips are typically longer."
    )
    
    st.write("---")
    st.subheader("📅 Time & Schedule")
    
    # Pickup hour
    pickup_hour = st.slider(
        "Pickup Hour of Day",
        min_value=0,
        max_value=23,
        value=17,
        help="0 represents midnight, 12 represents noon, 17 represents 5 PM."
    )
    
    # Day of week
    day_name = st.selectbox(
        "Pickup Day of Week",
        options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=4 # Friday
    )
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    day_of_week = day_map[day_name]
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("💵 Estimated Fare")
    
    # Construct raw dataframe for feature extraction
    ref_date = pd.to_datetime("2026-06-01") + pd.to_timedelta(day_of_week, unit="D")
    pickup_datetime = ref_date + pd.to_timedelta(pickup_hour, unit="h")
    
    infer_data = {
        "tpep_pickup_datetime": [pickup_datetime],
        "trip_distance": [trip_distance],
        "PULocationID": [pickup_selection["LocationID"]],
        "DOLocationID": [dropoff_selection["LocationID"]],
        "RatecodeID": [1.0],
        "payment_type": [1],
        "fare_amount": [0.0],
        "total_amount": [0.0],
        "tip_amount": [0.0]
    }
    
    infer_df = pd.DataFrame(infer_data)
    
    with st.spinner("Calculating optimal path and predicting fare..."):
        # Execute prediction
        features_df, _ = run_feature_pipeline(infer_df, scaler=scaler, is_training=False)
        X_infer = features_df.drop(columns=[TARGET_COL])
        
        predicted_fare = model.predict(X_infer)[0]
        
        # Guard against illogical predictions
        predicted_fare = max(3.0, predicted_fare)
        
    st.markdown(f'<div class="predicted-fare">${predicted_fare:.2f}</div>', unsafe_allow_html=True)
    st.caption("Estimated total fare (excluding tolls & voluntary tips). Base rate starts at $3.00.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")
    
    # ── Pydeck 3D Map Overlay ─────────────────────────────────────────────────
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("🗺️ Journey Visualisation")
    
    pu_lat, pu_lon = get_coordinates(pickup_selection["LocationID"], pickup_selection["Borough"])
    do_lat, do_lon = get_coordinates(dropoff_selection["LocationID"], dropoff_selection["Borough"])
    
    map_data = [{
        "pickup_lat": pu_lat,
        "pickup_lon": pu_lon,
        "dropoff_lat": do_lat,
        "dropoff_lon": do_lon,
        "pickup_name": pickup_selection["Zone"],
        "dropoff_name": dropoff_selection["Zone"]
    }]
    
    # Center map view between source and target
    view_state = pdk.ViewState(
        latitude=(pu_lat + do_lat) / 2,
        longitude=(pu_lon + do_lon) / 2,
        zoom=10.5,
        pitch=45,
        bearing=15
    )
    
    # 3D Arc linking Pickup -> Dropoff
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=map_data,
        get_source_position="[pickup_lon, pickup_lat]",
        get_target_position="[dropoff_lon, dropoff_lat]",
        get_source_color=[0, 242, 254, 200], # Teal
        get_target_color=[79, 172, 254, 200], # Blue
        get_width=6,
        pickable=True,
        auto_highlight=True
    )
    
    # Highlight points
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[
            {"lat": pu_lat, "lon": pu_lon, "name": pickup_selection["Zone"], "type": "Pickup"},
            {"lat": do_lat, "lon": do_lon, "name": dropoff_selection["Zone"], "type": "Dropoff"}
        ],
        get_position="[lon, lat]",
        get_color="type == 'Pickup' ? [0, 242, 254, 255] : [79, 172, 254, 255]",
        get_radius=120,
        pickable=True
    )
    
    r = pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip={"html": "<b>{type}:</b> {name}"},
        map_style="mapbox://styles/mapbox/dark-v10"
    )
    
    st.pydeck_chart(r)
    st.markdown('</div>', unsafe_allow_html=True)
