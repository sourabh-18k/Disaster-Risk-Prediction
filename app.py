# Disaster Risk Prediction — Web Dashboard
# Multi-hazard early warning: Heatwave, Flood, Landslide (India)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Disaster Risk Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

RISK_LABELS = {0: "No Risk", 1: "Low", 2: "Moderate", 3: "High", 4: "Extreme"}
RISK_COLORS = {
    "No Risk": "#2ecc71", "Low": "#f1c40f", "Moderate": "#e67e22",
    "High": "#e74c3c", "Extreme": "#8e44ad",
}
RISK_COLORS_LIST = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

MODEL_NAMES = ["random_forest", "xgboost", "lightgbm", "gradient_boosting"]

# Static geographic defaults for landslide regions (representative values)
LANDSLIDE_STATIC_DEFAULTS = {
    "Shimla": {"slope_deg": 28, "elevation_m": 2200, "soil_depth_m": 2.5,
               "geology_susceptibility": 0.6, "ndvi": 0.45, "land_use": 0,
               "seismic_pga (g)": 0.15, "aspect_deg": 180, "curvature": 0.1,
               "distance_to_road (km)": 0.8, "distance_to_stream (km)": 0.5},
    "Munnar": {"slope_deg": 32, "elevation_m": 1600, "soil_depth_m": 2.0,
               "geology_susceptibility": 0.55, "ndvi": 0.6, "land_use": 0,
               "seismic_pga (g)": 0.05, "aspect_deg": 210, "curvature": 0.15,
               "distance_to_road (km)": 1.0, "distance_to_stream (km)": 0.4},
    "Darjeeling": {"slope_deg": 30, "elevation_m": 2050, "soil_depth_m": 1.8,
                   "geology_susceptibility": 0.65, "ndvi": 0.5, "land_use": 0,
                   "seismic_pga (g)": 0.2, "aspect_deg": 195, "curvature": 0.2,
                   "distance_to_road (km)": 0.6, "distance_to_stream (km)": 0.3},
    "Uttarkashi": {"slope_deg": 35, "elevation_m": 2500, "soil_depth_m": 1.5,
                   "geology_susceptibility": 0.7, "ndvi": 0.35, "land_use": 2,
                   "seismic_pga (g)": 0.25, "aspect_deg": 170, "curvature": 0.25,
                   "distance_to_road (km)": 1.2, "distance_to_stream (km)": 0.6},
    "Kohima": {"slope_deg": 26, "elevation_m": 1500, "soil_depth_m": 2.2,
               "geology_susceptibility": 0.5, "ndvi": 0.55, "land_use": 0,
               "seismic_pga (g)": 0.1, "aspect_deg": 200, "curvature": 0.12,
               "distance_to_road (km)": 0.9, "distance_to_stream (km)": 0.45},
}

HEATWAVE_CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
}

FLOOD_CITIES = {
    "Guwahati": {"lat": 26.1445, "lon": 91.7362},
    "Patna": {"lat": 25.6093, "lon": 85.1376},
    "Kochi": {"lat": 9.9312, "lon": 76.2673},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Dehradun": {"lat": 30.3165, "lon": 78.0322},
}

LANDSLIDE_REGIONS = {
    "Shimla": {"lat": 31.1048, "lon": 77.1734},
    "Munnar": {"lat": 10.0889, "lon": 77.0595},
    "Darjeeling": {"lat": 27.0360, "lon": 88.2627},
    "Uttarkashi": {"lat": 30.7268, "lon": 78.4354},
    "Kohima": {"lat": 25.6751, "lon": 94.1086},
}


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_ml_models(hazard: str):
    """Load saved ML models + scaler for a given hazard type."""
    model_dir = BASE_DIR / "models" / hazard / "ml"
    models = {}
    scaler = None
    for pkl in model_dir.glob("*.pkl"):
        if pkl.stem == "scaler":
            scaler = joblib.load(pkl)
        else:
            models[pkl.stem] = joblib.load(pkl)
    return models, scaler


@st.cache_data
def load_featured_data(hazard: str) -> pd.DataFrame:
    """Load the pre-engineered featured dataset for a hazard."""
    paths = {
        "heatwave": BASE_DIR / "data" / "heatwave" / "features" / "featured_data_all_cities.csv",
        "flood": BASE_DIR / "data" / "flood" / "features" / "featured_flood_data_all_regions.csv",
        "landslide": BASE_DIR / "data" / "landslide" / "features" / "featured_landslide_data_all_regions.csv",
    }
    df = pd.read_csv(paths[hazard])
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    return df


@st.cache_data
def load_model_comparison(hazard: str) -> pd.DataFrame:
    """Load model comparison results CSV."""
    path = BASE_DIR / "results" / hazard / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def get_exclude_cols(hazard: str):
    """Return columns to exclude from features during prediction."""
    if hazard == "heatwave":
        return [
            "risk_level", "is_heatwave", "city", "is_hot_day",
            "is_severe_hot_day", "heatwave_candidate", "severe_heatwave_candidate",
        ]
    elif hazard == "flood":
        return [
            "risk_level", "is_flood", "city",
            "above_warning", "above_danger", "soil_saturated",
            "discharge_percentile", "rain_3d_sum", "rain_7d_sum",
        ]
    else:
        return [
            "risk_level", "is_landslide", "region", "hazard_score",
            "rain_3d_sum (mm)", "rain_7d_sum (mm)", "rain_14d_sum (mm)",
        ]


def get_feature_columns(df: pd.DataFrame, hazard: str):
    """Return numeric feature columns for a hazard, excluding meta/target."""
    exclude = get_exclude_cols(hazard)
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ("int64", "float64", "int32", "float32")
    ]


def predict_risk(models, scaler, X: pd.DataFrame, model_name: str):
    """Scale input features and predict risk level + probabilities."""
    # Align columns to match exactly what the scaler/model were trained on
    expected = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else list(X.columns)
    missing = [c for c in expected if c not in X.columns]
    extra = [c for c in X.columns if c not in expected]
    X_aligned = X.drop(columns=extra, errors="ignore")
    for c in missing:
        X_aligned[c] = 0.0
    X_aligned = X_aligned[expected]

    X_scaled = pd.DataFrame(scaler.transform(X_aligned), columns=expected, index=X.index)
    model = models[model_name]
    preds = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)
    return preds, probas


# ---------------------------------------------------------------------------
# Open-Meteo API fetching
# ---------------------------------------------------------------------------
def fetch_openmeteo_forecast(lat: float, lon: float, hazard: str) -> pd.DataFrame:
    """Fetch 7-day weather forecast from Open-Meteo for the given location."""
    import requests

    base_daily = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "rain_sum", "precipitation_hours",
        "apparent_temperature_mean", "shortwave_radiation_sum",
        "et0_fao_evapotranspiration", "dewpoint_2m_mean",
        "relative_humidity_2m_mean",
        "windgusts_10m_max", "windspeed_10m_mean",
        "cloudcover_mean", "pressure_msl_mean",
        "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
        "soil_temperature_0_to_7cm_mean",
        "vapor_pressure_deficit_max",
    ]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(base_daily),
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    df = pd.DataFrame(daily)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    # Rename columns to match project convention
    rename_map = {
        "temperature_2m_max": "temperature_2m_max (°C)",
        "temperature_2m_min": "temperature_2m_min (°C)",
        "temperature_2m_mean": "temperature_2m_mean (°C)",
        "precipitation_sum": "precipitation_sum (mm)",
        "rain_sum": "rain_sum (mm)",
        "precipitation_hours": "precipitation_hours (h)",
        "apparent_temperature_mean": "apparent_temperature_mean (°C)",
        "shortwave_radiation_sum": "shortwave_radiation_sum (MJ/m²)",
        "et0_fao_evapotranspiration": "et0_fao_evapotranspiration (mm)",
        "dewpoint_2m_mean": "dew_point_2m_mean (°C)",
        "relative_humidity_2m_mean": "relative_humidity_2m_mean (%)",
        "windgusts_10m_max": "wind_gusts_10m_max (km/h)",
        "windspeed_10m_mean": "wind_speed_10m_mean (km/h)",
        "cloudcover_mean": "cloud_cover_mean (%)",
        "pressure_msl_mean": "pressure_msl_mean (hPa)",
        "soil_moisture_0_to_7cm_mean": "soil_moisture_0_to_7cm_mean (m³/m³)",
        "soil_moisture_7_to_28cm_mean": "soil_moisture_7_to_28cm_mean (m³/m³)",
        "soil_temperature_0_to_7cm_mean": "soil_temperature_0_to_7cm_mean (°C)",
        "vapor_pressure_deficit_max": "vapour_pressure_deficit_max (kPa)",
    }
    df = df.rename(columns=rename_map)
    return df


def fetch_openmeteo_historical(lat: float, lon: float, days_back: int = 60) -> pd.DataFrame:
    """Fetch historical weather from Open-Meteo Archive API."""
    import requests

    end = datetime.now() - timedelta(days=6)
    start = end - timedelta(days=days_back)

    base_daily = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "rain_sum", "precipitation_hours",
        "apparent_temperature_mean", "shortwave_radiation_sum",
        "et0_fao_evapotranspiration", "dewpoint_2m_mean",
        "relative_humidity_2m_mean",
        "windgusts_10m_max", "windspeed_10m_mean",
        "cloudcover_mean", "pressure_msl_mean",
        "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
        "soil_temperature_0_to_7cm_mean",
        "vapor_pressure_deficit_max",
    ]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(base_daily),
        "timezone": "Asia/Kolkata",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    df = pd.DataFrame(daily)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    rename_map = {
        "temperature_2m_max": "temperature_2m_max (°C)",
        "temperature_2m_min": "temperature_2m_min (°C)",
        "temperature_2m_mean": "temperature_2m_mean (°C)",
        "precipitation_sum": "precipitation_sum (mm)",
        "rain_sum": "rain_sum (mm)",
        "precipitation_hours": "precipitation_hours (h)",
        "apparent_temperature_mean": "apparent_temperature_mean (°C)",
        "shortwave_radiation_sum": "shortwave_radiation_sum (MJ/m²)",
        "et0_fao_evapotranspiration": "et0_fao_evapotranspiration (mm)",
        "dewpoint_2m_mean": "dew_point_2m_mean (°C)",
        "relative_humidity_2m_mean": "relative_humidity_2m_mean (%)",
        "windgusts_10m_max": "wind_gusts_10m_max (km/h)",
        "windspeed_10m_mean": "wind_speed_10m_mean (km/h)",
        "cloudcover_mean": "cloud_cover_mean (%)",
        "pressure_msl_mean": "pressure_msl_mean (hPa)",
        "soil_moisture_0_to_7cm_mean": "soil_moisture_0_to_7cm_mean (m³/m³)",
        "soil_moisture_7_to_28cm_mean": "soil_moisture_7_to_28cm_mean (m³/m³)",
        "soil_temperature_0_to_7cm_mean": "soil_temperature_0_to_7cm_mean (°C)",
        "vapor_pressure_deficit_max": "vapour_pressure_deficit_max (kPa)",
    }
    df = df.rename(columns=rename_map)
    return df


# ---------------------------------------------------------------------------
# Live Prediction Helpers — build derived columns for each hazard
# ---------------------------------------------------------------------------
def _calculate_api(precip_series, k=0.85):
    """Antecedent Precipitation Index with exponential decay factor k."""
    api = np.zeros(len(precip_series))
    vals = precip_series.values
    for i in range(1, len(vals)):
        api[i] = k * api[i - 1] + vals[i]
    return api


def _add_heat_derived(df, city):
    """Add heat-specific derived columns needed by the feature engineering pipeline."""
    from heatwave.data_preprocessing import calculate_heat_index, CITY_METADATA

    # heat_index
    df["heat_index"] = df.apply(
        lambda r: calculate_heat_index(
            r["temperature_2m_mean (°C)"], r["relative_humidity_2m_mean (%)"]
        ),
        axis=1,
    )

    # temp_departure: use 30-day rolling mean as proxy for climatology
    climatology = df["temperature_2m_max (°C)"].rolling(30, min_periods=7).mean()
    df["temp_departure"] = df["temperature_2m_max (°C)"] - climatology
    df["temp_departure"] = df["temp_departure"].fillna(0)

    # Other derived features matching add_derived_features()
    df["temp_range"] = df["temperature_2m_max (°C)"] - df["temperature_2m_min (°C)"]
    df["apparent_temp_diff"] = (
        df["apparent_temperature_mean (°C)"] - df["temperature_2m_mean (°C)"]
    )
    if "soil_temperature_0_to_7cm_mean (°C)" in df.columns:
        df["soil_temp_deviation"] = (
            df["soil_temperature_0_to_7cm_mean (°C)"] - df["temperature_2m_mean (°C)"]
        )
    df["moisture_deficit"] = df.get("vapour_pressure_deficit_max (kPa)", 0)
    if "wind_speed_10m_mean (km/h)" in df.columns and "dew_point_2m_mean (°C)" in df.columns:
        df["wind_cooling_effect"] = df["wind_speed_10m_mean (km/h)"] * (
            df["temperature_2m_max (°C)"] - df["dew_point_2m_mean (°C)"]
        )
    df["has_precipitation"] = (df["precipitation_sum (mm)"] > 0).astype(int)
    return df


def _add_flood_derived(df, city):
    """Add flood-specific columns not available from weather API using simple proxies."""
    precip = df["precipitation_sum (mm)"]

    # Antecedent Precipitation Index
    df["antecedent_precip_index (mm)"] = _calculate_api(precip)

    # Catchment wetness index from soil moisture
    if "soil_moisture_0_to_7cm_mean (m³/m³)" in df.columns:
        df["catchment_wetness_index"] = df["soil_moisture_0_to_7cm_mean (m³/m³)"] * 2
    else:
        df["catchment_wetness_index"] = 0.5

    # River discharge proxy (API * catchment coefficient + direct rainfall contribution)
    api = df["antecedent_precip_index (mm)"]
    df["river_discharge (cumecs)"] = api * 5 + precip * 10
    max_discharge = df["river_discharge (cumecs)"].max()
    if max_discharge == 0:
        max_discharge = 1

    # Water level proxy
    df["water_level (m)"] = 3.0 + (df["river_discharge (cumecs)"] / max_discharge) * 5

    # Reservoir storage proxy (seasonal)
    month = df.index.month
    df["reservoir_storage_pct (%)"] = 50.0
    df.loc[month.isin([7, 8, 9]), "reservoir_storage_pct (%)"] = 80.0
    df.loc[month.isin([6, 10]), "reservoir_storage_pct (%)"] = 65.0

    return df


def _add_landslide_derived(df, region):
    """Add landslide-specific static geographic and hydrological proxy columns."""
    static = LANDSLIDE_STATIC_DEFAULTS.get(region, LANDSLIDE_STATIC_DEFAULTS["Shimla"])
    for col, val in static.items():
        df[col] = val

    precip = df["precipitation_sum (mm)"]

    # Cumulative rain sums
    df["rain_3d_sum (mm)"] = precip.rolling(3, min_periods=1).sum()
    df["rain_7d_sum (mm)"] = precip.rolling(7, min_periods=1).sum()
    df["rain_14d_sum (mm)"] = precip.rolling(14, min_periods=1).sum()

    # Antecedent precipitation index
    df["antecedent_precip_index (mm)"] = _calculate_api(precip)

    # Pore water pressure proxy (soil moisture and rain driven)
    sm = df.get("soil_moisture_0_to_7cm_mean (m³/m³)", pd.Series(0.3, index=df.index))
    df["pore_water_pressure (kPa)"] = sm * 20 + precip * 0.1

    return df


def run_live_prediction(hazard, city, coords, models, scaler, model_name):
    """Fetch data, engineer features, and predict risk for the next 7 days."""
    import io, contextlib

    progress = st.progress(0, text="Fetching historical weather data (60 days)...")
    hist_df = fetch_openmeteo_historical(coords["lat"], coords["lon"], days_back=60)
    progress.progress(25, text="Fetching 7-day weather forecast...")
    forecast_df = fetch_openmeteo_forecast(coords["lat"], coords["lon"], hazard)

    # Combine and deduplicate
    combined = pd.concat([hist_df, forecast_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    st.success(f"Fetched **{len(hist_df)}** days history + **{len(forecast_df)}** days forecast")

    progress.progress(40, text="Adding derived features...")

    # Add location column
    loc_col = "city" if hazard != "landslide" else "region"
    combined[loc_col] = city

    # Add hazard-specific derived columns
    if hazard == "heatwave":
        combined = _add_heat_derived(combined, city)
    elif hazard == "flood":
        combined = _add_flood_derived(combined, city)
    else:
        combined = _add_landslide_derived(combined, city)

    progress.progress(55, text="Running feature engineering pipeline...")

    # Import and run feature engineering (suppress prints)
    if hazard == "heatwave":
        from heatwave.feature_engineering import create_all_features
    elif hazard == "flood":
        from flood.feature_engineering import create_all_features
    else:
        from landslide.feature_engineering import create_all_features

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        featured = create_all_features(combined)

    progress.progress(80, text="Running model prediction...")

    # Extract only the forecast period
    forecast_dates = forecast_df.index
    featured_forecast = featured.loc[featured.index.isin(forecast_dates)].copy()

    if featured_forecast.empty:
        st.warning("Could not align forecast dates with feature-engineered data.")
        return None

    # Get feature columns and predict
    feat_cols = get_feature_columns(featured_forecast, hazard)
    X = featured_forecast[feat_cols].ffill().fillna(0)

    preds, probas = predict_risk(models, scaler, X, model_name)

    featured_forecast["predicted_risk"] = preds
    featured_forecast["predicted_label"] = [RISK_LABELS.get(p, "?") for p in preds]
    featured_forecast["confidence"] = probas.max(axis=1)

    progress.progress(100, text="Done!")
    return featured_forecast


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("Disaster Risk")
        page = st.radio(
            "Go to",
            [
                "Dashboard",
                "Heatwave Prediction",
                "Flood Prediction",
                "Landslide Prediction",
                "Model Performance",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Multi-hazard EWS v2 | India")
    return page


# ---------------------------------------------------------------------------
# Dashboard Page
# ---------------------------------------------------------------------------
def page_dashboard():
    st.title("Disaster Risk Prediction")
    st.markdown("Multi-hazard early warning system covering 15 cities and regions across India.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Heatwave")
        comp = load_model_comparison("heatwave")
        if not comp.empty:
            best = comp["f1_weighted"].idxmax()
            st.metric("Best Model", best.replace("_", " ").title())
            st.metric("F1 Score", f"{comp.loc[best, 'f1_weighted']:.4f}")
            st.metric("CSI (Tuned)", f"{comp.loc[best, 'Tuned_CSI']:.4f}" if "Tuned_CSI" in comp.columns else "N/A")
        st.caption(f"Cities: {', '.join(HEATWAVE_CITIES.keys())}")

    with col2:
        st.markdown("### Flood")
        comp = load_model_comparison("flood")
        if not comp.empty:
            best_col = "Tuned_CSI" if "Tuned_CSI" in comp.columns else "f1_weighted"
            best = comp[best_col].idxmax()
            st.metric("Best Model", best.replace("_", " ").title())
            st.metric("F1 Score", f"{comp.loc[best, 'f1_weighted']:.4f}")
            st.metric("CSI (Tuned)", f"{comp.loc[best, 'Tuned_CSI']:.4f}" if "Tuned_CSI" in comp.columns else "N/A")
        st.caption(f"Cities: {', '.join(FLOOD_CITIES.keys())}")

    with col3:
        st.markdown("### Landslide")
        comp = load_model_comparison("landslide")
        if not comp.empty:
            best_col = "Tuned_CSI" if "Tuned_CSI" in comp.columns else "f1_weighted"
            best = comp[best_col].idxmax()
            st.metric("Best Model", best.replace("_", " ").title())
            st.metric("F1 Score", f"{comp.loc[best, 'f1_weighted']:.4f}")
            st.metric("CSI (Tuned)", f"{comp.loc[best, 'Tuned_CSI']:.4f}" if "Tuned_CSI" in comp.columns else "N/A")
        st.caption(f"Regions: {', '.join(LANDSLIDE_REGIONS.keys())}")

    st.divider()

    st.subheader("Monitored Locations")

    col_h, col_f, col_l = st.columns(3)
    with col_h:
        st.markdown("**Heatwave**")
        st.markdown(
            "- Delhi, Delhi\n"
            "- Jaipur, Rajasthan\n"
            "- Hyderabad, Telangana\n"
            "- Bhubaneswar, Odisha\n"
            "- Chennai, Tamil Nadu"
        )
    with col_f:
        st.markdown("**Flood**")
        st.markdown(
            "- Guwahati, Assam\n"
            "- Patna, Bihar\n"
            "- Kochi, Kerala\n"
            "- Mumbai, Maharashtra\n"
            "- Dehradun, Uttarakhand"
        )
    with col_l:
        st.markdown("**Landslide**")
        st.markdown(
            "- Shimla, Himachal Pradesh\n"
            "- Munnar, Kerala\n"
            "- Darjeeling, West Bengal\n"
            "- Uttarkashi, Uttarakhand\n"
            "- Kohima, Nagaland"
        )

    st.divider()
    st.subheader("How it works")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**Data Collection**\n\n16 years of daily weather observations (Open-Meteo) combined with hydrological and geomorphic variables")
    c2.markdown("**Feature Engineering**\n\n120-200 features per hazard: temporal lags, rolling statistics, compound interaction indicators")
    c3.markdown("**Model Training**\n\nRandom Forest, XGBoost, LightGBM, Gradient Boosting with SMOTE oversampling and custom class weights")
    c4.markdown("**Evaluation**\n\nWMO-standard verification: POD, FAR, CSI, HSS with optimal probability threshold tuning")


# ---------------------------------------------------------------------------
# Generic Prediction Page
# ---------------------------------------------------------------------------
def page_prediction(hazard: str, cities_dict: dict, emoji: str):
    st.title(f"{hazard.title()} Risk Prediction")

    models, scaler = load_ml_models(hazard)
    if not models:
        st.error(f"No trained models found for {hazard}. Run the training pipeline first.")
        return

    # Model selector
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        model_name = st.selectbox(
            "Select Model",
            list(models.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )
    with col_cfg2:
        city = st.selectbox(
            "Select City / Region",
            list(cities_dict.keys()),
        )

    coords = cities_dict[city]

    tab_live, tab_hist = st.tabs(["Live Forecast", "Historical Analysis"])

    # ------ LIVE FORECAST TAB ------
    with tab_live:
        st.markdown(
            f"Fetches **60 days of history + 7-day forecast** for **{city}** from Open-Meteo, "
            f"runs the full feature-engineering pipeline, and predicts disaster risk using **{model_name.replace('_', ' ').title()}**."
        )
        if st.button(f"Run Forecast — {city}", type="primary"):
            try:
                result = run_live_prediction(hazard, city, coords, models, scaler, model_name)

                if result is not None and not result.empty:
                    # ---- Key metric cards ----
                    max_risk = int(result["predicted_risk"].max())
                    avg_conf = result["confidence"].mean()
                    high_days = int((result["predicted_risk"] >= 3).sum())

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Peak Risk", RISK_LABELS.get(max_risk, "?"))
                    m2.metric("High+ Risk Days", f"{high_days} / {len(result)}")
                    m3.metric("Avg Confidence", f"{avg_conf:.1%}")
                    m4.metric("Forecast Period", f"{result.index[0].strftime('%d %b')} – {result.index[-1].strftime('%d %b')}")

                    # ---- 7-Day Risk Forecast Chart ----
                    st.subheader("7-Day Risk Forecast")
                    fig = go.Figure()
                    colors = [RISK_COLORS_LIST[int(r)] for r in result["predicted_risk"]]
                    fig.add_trace(go.Bar(
                        x=result.index.strftime("%a %d %b"),
                        y=result["predicted_risk"],
                        marker_color=colors,
                        text=result["predicted_label"],
                        textposition="outside",
                        hovertemplate="Date: %{x}<br>Risk: %{text}<br>Confidence: %{customdata:.1%}",
                        customdata=result["confidence"],
                    ))
                    fig.update_layout(
                        yaxis=dict(
                            title="Risk Level",
                            tickvals=[0, 1, 2, 3, 4],
                            ticktext=["No Risk", "Low", "Moderate", "High", "Extreme"],
                            range=[-0.3, 4.8],
                        ),
                        height=380,
                        margin=dict(t=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ---- Weather Context Chart ----
                    st.subheader("Weather Context")
                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                    if "temperature_2m_max (°C)" in result.columns:
                        fig2.add_trace(go.Scatter(
                            x=result.index, y=result["temperature_2m_max (°C)"],
                            name="Max Temp (°C)", line=dict(color="#e74c3c", width=2),
                        ), secondary_y=False)
                    if "temperature_2m_min (°C)" in result.columns:
                        fig2.add_trace(go.Scatter(
                            x=result.index, y=result["temperature_2m_min (°C)"],
                            name="Min Temp (°C)", line=dict(color="#3498db", width=2),
                        ), secondary_y=False)
                    if "precipitation_sum (mm)" in result.columns:
                        fig2.add_trace(go.Bar(
                            x=result.index, y=result["precipitation_sum (mm)"],
                            name="Precipitation (mm)", marker_color="#2ecc71", opacity=0.5,
                        ), secondary_y=True)
                    fig2.update_layout(
                        yaxis=dict(title="Temperature (°C)"),
                        yaxis2=dict(title="Precipitation (mm)"),
                        height=350,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    # ---- Confidence per-day ----
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader("Prediction Confidence")
                        fig_c = go.Figure(go.Bar(
                            x=result.index.strftime("%a %d %b"),
                            y=result["confidence"],
                            marker_color="#3498db",
                            text=[f"{c:.0%}" for c in result["confidence"]],
                            textposition="outside",
                        ))
                        fig_c.update_layout(yaxis=dict(title="Confidence", range=[0, 1.15]), height=300)
                        st.plotly_chart(fig_c, use_container_width=True)

                    with col_b:
                        st.subheader("Risk Distribution")
                        risk_counts = result["predicted_label"].value_counts()
                        fig_d = px.pie(
                            names=risk_counts.index, values=risk_counts.values,
                            color=risk_counts.index, color_discrete_map=RISK_COLORS,
                        )
                        fig_d.update_layout(height=300)
                        st.plotly_chart(fig_d, use_container_width=True)

                    with st.expander("Detailed Daily Predictions"):
                        show_cols = ["predicted_label", "confidence"]
                        weather_cols = [c for c in ["temperature_2m_max (°C)", "temperature_2m_min (°C)",
                                                     "precipitation_sum (mm)", "relative_humidity_2m_mean (%)"]
                                        if c in result.columns]
                        show_cols = weather_cols + show_cols
                        st.dataframe(
                            result[show_cols].style.format(
                                {c: "{:.1f}" for c in weather_cols} | {"confidence": "{:.1%}"}
                            ),
                            use_container_width=True,
                        )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ------ HISTORICAL ANALYSIS TAB ------
    with tab_hist:
        featured_df = load_featured_data(hazard)

        # Filter by city/region
        loc_col = "city" if hazard != "landslide" else "region"
        if loc_col in featured_df.columns:
            city_df = featured_df[featured_df[loc_col] == city].copy()
        else:
            city_df = featured_df.copy()

        if city_df.empty:
            st.warning(f"No data found for {city}.")
            return

        st.markdown(f"**{len(city_df):,}** records for **{city}** | Features: **{len(city_df.columns)}**")

        # Get feature columns & run prediction
        feat_cols = get_feature_columns(city_df, hazard)
        X = city_df[feat_cols].copy()

        # Handle NaN via forward-fill then zero-fill for remaining
        X = X.ffill().fillna(0)

        preds, probas = predict_risk(models, scaler, X, model_name)

        city_df = city_df.copy()
        city_df["predicted_risk"] = preds
        city_df["predicted_label"] = [RISK_LABELS.get(p, "?") for p in preds]
        city_df["confidence"] = probas.max(axis=1)

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        actual = city_df["risk_level"] if "risk_level" in city_df.columns else None
        if actual is not None:
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(actual, preds)
            f1 = f1_score(actual, preds, average="weighted", zero_division=0)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("Weighted F1", f"{f1:.4f}")
        c3.metric("High+ Risk Days", int((preds >= 3).sum()))
        c4.metric("Avg Confidence", f"{city_df['confidence'].mean():.2%}")

        # Risk timeline
        st.subheader("Risk Level Timeline")
        fig = go.Figure()
        if actual is not None:
            fig.add_trace(go.Scatter(
                x=city_df.index, y=actual,
                name="Actual", mode="lines", line=dict(color="#3498db", width=1.5),
            ))
        fig.add_trace(go.Scatter(
            x=city_df.index, y=preds,
            name="Predicted", mode="lines", line=dict(color="#e74c3c", width=1.5, dash="dot"),
        ))
        fig.update_layout(
            yaxis=dict(title="Risk Level", tickvals=[0, 1, 2, 3, 4],
                       ticktext=["No Risk", "Low", "Moderate", "High", "Extreme"]),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk distribution
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Predicted Risk Distribution")
            risk_counts = pd.Series(preds).map(RISK_LABELS).value_counts()
            fig_dist = px.pie(
                names=risk_counts.index, values=risk_counts.values,
                color=risk_counts.index, color_discrete_map=RISK_COLORS,
            )
            fig_dist.update_layout(height=350)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_b:
            st.subheader("Prediction Confidence")
            fig_conf = px.histogram(city_df, x="confidence", nbins=30, color_discrete_sequence=["#3498db"])
            fig_conf.update_layout(height=350, xaxis_title="Confidence Score", yaxis_title="Count")
            st.plotly_chart(fig_conf, use_container_width=True)

        with st.expander("View Detailed Predictions (last 30 days)"):
            cols_show = ["predicted_label", "confidence"]
            if actual is not None:
                cols_show = ["risk_level", "predicted_label", "confidence"]
            st.dataframe(city_df[cols_show].tail(30).style.format({"confidence": "{:.2%}"}), use_container_width=True)


# ---------------------------------------------------------------------------
# Model Performance Page
# ---------------------------------------------------------------------------
def page_performance():
    st.title("Model Performance Comparison")

    hazard = st.selectbox("Select Hazard Type", ["heatwave", "flood", "landslide"])
    comp = load_model_comparison(hazard)

    if comp.empty:
        st.warning("No model comparison data available. Run the pipeline first.")
        return

    st.subheader(f"{hazard.title()} — Default Metrics")
    default_metrics = ["accuracy", "f1_weighted", "POD", "FAR", "CSI", "HSS"]
    available = [m for m in default_metrics if m in comp.columns]
    st.dataframe(comp[available].style.format("{:.4f}").background_gradient(cmap="YlGn", axis=0), use_container_width=True)

    # Tuned metrics
    tuned = [c for c in comp.columns if c.startswith("Tuned_")]
    if tuned:
        st.subheader("Threshold-Tuned Metrics")
        st.dataframe(comp[tuned].style.format("{:.4f}").background_gradient(cmap="YlGn", axis=0), use_container_width=True)

    # Bar chart comparison
    st.subheader("Visual Comparison")
    chart_metrics = ["accuracy", "f1_weighted", "POD", "CSI"]
    avail_chart = [m for m in chart_metrics if m in comp.columns]
    if avail_chart:
        reset = comp[avail_chart].reset_index()
        id_col = reset.columns[0]  # first column after reset_index (model name)
        melted = reset.melt(id_vars=id_col, var_name="Metric", value_name="Score")
        melted = melted.rename(columns={id_col: "Model"})
        fig = px.bar(
            melted, x="Model", y="Score", color="Metric",
            barmode="group", color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Tuned comparison
    if "Tuned_CSI" in comp.columns and "Tuned_POD" in comp.columns:
        st.subheader("Tuned: POD vs CSI")
        fig2 = go.Figure()
        for model in comp.index:
            fig2.add_trace(go.Scatter(
                x=[comp.loc[model, "Tuned_POD"]], y=[comp.loc[model, "Tuned_CSI"]],
                mode="markers+text", text=[model.replace("_", " ").title()],
                textposition="top center", marker=dict(size=14),
                name=model.replace("_", " ").title(),
            ))
        fig2.update_layout(
            xaxis_title="Tuned POD (Probability of Detection)",
            yaxis_title="Tuned CSI (Critical Success Index)",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    page = render_sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Heatwave Prediction":
        page_prediction("heatwave", HEATWAVE_CITIES, "")
    elif page == "Flood Prediction":
        page_prediction("flood", FLOOD_CITIES, "")
    elif page == "Landslide Prediction":
        page_prediction("landslide", LANDSLIDE_REGIONS, "")
    elif page == "Model Performance":
        page_performance()


if __name__ == "__main__":
    main()
