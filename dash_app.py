# =============================================================================
# Disaster Risk Prediction — Premium Dash Dashboard
# Multi-hazard early warning: Heatwave, Flood, Landslide (India)
# =============================================================================

import io
import sys
import contextlib
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, no_update
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Disaster Risk Prediction",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_LABELS = {0: "No Risk", 1: "Low", 2: "Moderate", 3: "High", 4: "Extreme"}
RISK_COLORS = {
    "No Risk": "#16a34a", "Low": "#ca8a04",
    "Moderate": "#ea580c", "High": "#dc2626", "Extreme": "#7c3aed",
}
RISK_COLORS_LIST = ["#16a34a", "#ca8a04", "#ea580c", "#dc2626", "#7c3aed"]
RISK_CSS = {
    "No Risk": "risk-none", "Low": "risk-low",
    "Moderate": "risk-moderate", "High": "risk-high", "Extreme": "risk-extreme",
}
MODEL_NAMES = ["random_forest", "xgboost", "lightgbm", "gradient_boosting"]

HEATWAVE_CITIES = {
    "Delhi":       {"lat": 28.6139, "lon": 77.2090},
    "Jaipur":      {"lat": 26.9124, "lon": 75.7873},
    "Hyderabad":   {"lat": 17.3850, "lon": 78.4867},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245},
    "Chennai":     {"lat": 13.0827, "lon": 80.2707},
}
FLOOD_CITIES = {
    "Guwahati":  {"lat": 26.1445, "lon": 91.7362},
    "Patna":     {"lat": 25.6093, "lon": 85.1376},
    "Kochi":     {"lat":  9.9312, "lon": 76.2673},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Dehradun":  {"lat": 30.3165, "lon": 78.0322},
}
LANDSLIDE_REGIONS = {
    "Shimla":     {"lat": 31.1048, "lon": 77.1734},
    "Munnar":     {"lat": 10.0889, "lon": 77.0595},
    "Darjeeling": {"lat": 27.0360, "lon": 88.2627},
    "Uttarkashi": {"lat": 30.7268, "lon": 78.4354},
    "Kohima":     {"lat": 25.6751, "lon": 94.1086},
}
LANDSLIDE_STATIC_DEFAULTS = {
    "Shimla":     {"slope_deg": 28, "elevation_m": 2200, "soil_depth_m": 2.5,  "geology_susceptibility": 0.6,  "ndvi": 0.45, "land_use": 0, "seismic_pga (g)": 0.15, "aspect_deg": 180, "curvature": 0.10, "distance_to_road (km)": 0.8, "distance_to_stream (km)": 0.5},
    "Munnar":     {"slope_deg": 32, "elevation_m": 1600, "soil_depth_m": 2.0,  "geology_susceptibility": 0.55, "ndvi": 0.60, "land_use": 0, "seismic_pga (g)": 0.05, "aspect_deg": 210, "curvature": 0.15, "distance_to_road (km)": 1.0, "distance_to_stream (km)": 0.4},
    "Darjeeling": {"slope_deg": 30, "elevation_m": 2050, "soil_depth_m": 1.8,  "geology_susceptibility": 0.65, "ndvi": 0.50, "land_use": 0, "seismic_pga (g)": 0.20, "aspect_deg": 195, "curvature": 0.20, "distance_to_road (km)": 0.6, "distance_to_stream (km)": 0.3},
    "Uttarkashi": {"slope_deg": 35, "elevation_m": 2500, "soil_depth_m": 1.5,  "geology_susceptibility": 0.70, "ndvi": 0.35, "land_use": 2, "seismic_pga (g)": 0.25, "aspect_deg": 170, "curvature": 0.25, "distance_to_road (km)": 1.2, "distance_to_stream (km)": 0.6},
    "Kohima":     {"slope_deg": 26, "elevation_m": 1500, "soil_depth_m": 2.2,  "geology_susceptibility": 0.50, "ndvi": 0.55, "land_use": 0, "seismic_pga (g)": 0.10, "aspect_deg": 200, "curvature": 0.12, "distance_to_road (km)": 0.9, "distance_to_stream (km)": 0.45},
}

# Base layout — NO xaxis/yaxis keys so callers can pass them without conflict
PLOTLY_TEMPLATE = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "font": {"color": "#555", "family": "Inter, sans-serif", "size": 12},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#333"}},
    "margin": {"t": 36, "r": 12, "b": 40, "l": 12},
}

# Axis defaults for charts that need custom axes
AXIS_DEFAULTS = {
    "gridcolor": "rgba(0,0,0,0.06)",
    "zerolinecolor": "rgba(0,0,0,0.12)",
    "linecolor": "rgba(0,0,0,0.12)",
    "tickcolor": "#999",
    "tickfont": {"color": "#666", "size": 11},
}

# ---------------------------------------------------------------------------
# Data / Model helpers (identical logic to app.py — no Streamlit deps)
# ---------------------------------------------------------------------------
_model_cache = {}
_data_cache  = {}
_comp_cache  = {}


def load_ml_models(hazard: str):
    if hazard in _model_cache:
        return _model_cache[hazard]
    model_dir = BASE_DIR / "models" / hazard / "ml"
    models, scaler = {}, None
    for pkl in model_dir.glob("*.pkl"):
        if pkl.stem == "scaler":
            scaler = joblib.load(pkl)
        else:
            models[pkl.stem] = joblib.load(pkl)
    _model_cache[hazard] = (models, scaler)
    return models, scaler


def load_featured_data(hazard: str) -> pd.DataFrame:
    if hazard in _data_cache:
        return _data_cache[hazard]
    paths = {
        "heatwave":  BASE_DIR / "data" / "heatwave"  / "features" / "featured_data_all_cities.csv",
        "flood":     BASE_DIR / "data" / "flood"     / "features" / "featured_flood_data_all_regions.csv",
        "landslide": BASE_DIR / "data" / "landslide" / "features" / "featured_landslide_data_all_regions.csv",
    }
    df = pd.read_csv(paths[hazard])
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    _data_cache[hazard] = df
    return df


def load_model_comparison(hazard: str) -> pd.DataFrame:
    if hazard in _comp_cache:
        return _comp_cache[hazard]
    path = BASE_DIR / "results" / hazard / "model_comparison.csv"
    df = pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame()
    _comp_cache[hazard] = df
    return df


def get_exclude_cols(hazard: str):
    if hazard == "heatwave":
        return ["risk_level","is_heatwave","city","is_hot_day","is_severe_hot_day","heatwave_candidate","severe_heatwave_candidate"]
    elif hazard == "flood":
        return ["risk_level","is_flood","city","above_warning","above_danger","soil_saturated","discharge_percentile","rain_3d_sum","rain_7d_sum"]
    return ["risk_level","is_landslide","region","hazard_score","rain_3d_sum (mm)","rain_7d_sum (mm)","rain_14d_sum (mm)"]


def get_feature_columns(df, hazard):
    exclude = get_exclude_cols(hazard)
    return [c for c in df.columns if c not in exclude and df[c].dtype in ("int64","float64","int32","float32")]


def predict_risk(models, scaler, X, model_name):
    expected = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else list(X.columns)
    X_aligned = X.drop(columns=[c for c in X.columns if c not in expected], errors="ignore")
    for c in [c for c in expected if c not in X.columns]:
        X_aligned[c] = 0.0
    X_aligned = X_aligned[expected]
    X_scaled = pd.DataFrame(scaler.transform(X_aligned), columns=expected, index=X.index)
    model = models[model_name]
    return model.predict(X_scaled), model.predict_proba(X_scaled)


def _calculate_api(precip_series, k=0.85):
    api = np.zeros(len(precip_series))
    vals = precip_series.values
    for i in range(1, len(vals)):
        api[i] = k * api[i - 1] + vals[i]
    return api


def _add_heat_derived(df, city):
    from heatwave.data_preprocessing import calculate_heat_index
    df["heat_index"] = df.apply(lambda r: calculate_heat_index(r["temperature_2m_mean (°C)"], r["relative_humidity_2m_mean (%)"]), axis=1)
    climatology = df["temperature_2m_max (°C)"].rolling(30, min_periods=7).mean()
    df["temp_departure"] = (df["temperature_2m_max (°C)"] - climatology).fillna(0)
    df["temp_range"] = df["temperature_2m_max (°C)"] - df["temperature_2m_min (°C)"]
    df["apparent_temp_diff"] = df["apparent_temperature_mean (°C)"] - df["temperature_2m_mean (°C)"]
    if "soil_temperature_0_to_7cm_mean (°C)" in df.columns:
        df["soil_temp_deviation"] = df["soil_temperature_0_to_7cm_mean (°C)"] - df["temperature_2m_mean (°C)"]
    df["moisture_deficit"] = df.get("vapour_pressure_deficit_max (kPa)", 0)
    if "wind_speed_10m_mean (km/h)" in df.columns and "dew_point_2m_mean (°C)" in df.columns:
        df["wind_cooling_effect"] = df["wind_speed_10m_mean (km/h)"] * (df["temperature_2m_max (°C)"] - df["dew_point_2m_mean (°C)"])
    df["has_precipitation"] = (df["precipitation_sum (mm)"] > 0).astype(int)
    return df


def _add_flood_derived(df, city):
    precip = df["precipitation_sum (mm)"]
    df["antecedent_precip_index (mm)"] = _calculate_api(precip)
    df["catchment_wetness_index"] = df["soil_moisture_0_to_7cm_mean (m³/m³)"].multiply(2) if "soil_moisture_0_to_7cm_mean (m³/m³)" in df.columns else 0.5
    api = df["antecedent_precip_index (mm)"]
    df["river_discharge (cumecs)"] = api * 5 + precip * 10
    mx = df["river_discharge (cumecs)"].max() or 1
    df["water_level (m)"] = 3.0 + (df["river_discharge (cumecs)"] / mx) * 5
    month = df.index.month
    df["reservoir_storage_pct (%)"] = 50.0
    df.loc[month.isin([7, 8, 9]), "reservoir_storage_pct (%)"] = 80.0
    df.loc[month.isin([6, 10]),   "reservoir_storage_pct (%)"] = 65.0
    return df


def _add_landslide_derived(df, region):
    for col, val in LANDSLIDE_STATIC_DEFAULTS.get(region, LANDSLIDE_STATIC_DEFAULTS["Shimla"]).items():
        df[col] = val
    precip = df["precipitation_sum (mm)"]
    df["rain_3d_sum (mm)"]  = precip.rolling(3,  min_periods=1).sum()
    df["rain_7d_sum (mm)"]  = precip.rolling(7,  min_periods=1).sum()
    df["rain_14d_sum (mm)"] = precip.rolling(14, min_periods=1).sum()
    df["antecedent_precip_index (mm)"] = _calculate_api(precip)
    sm = df.get("soil_moisture_0_to_7cm_mean (m³/m³)", pd.Series(0.3, index=df.index))
    df["pore_water_pressure (kPa)"] = sm * 20 + precip * 0.1
    return df


RENAME_MAP = {
    "temperature_2m_max": "temperature_2m_max (°C)", "temperature_2m_min": "temperature_2m_min (°C)",
    "temperature_2m_mean": "temperature_2m_mean (°C)", "precipitation_sum": "precipitation_sum (mm)",
    "rain_sum": "rain_sum (mm)", "precipitation_hours": "precipitation_hours (h)",
    "apparent_temperature_mean": "apparent_temperature_mean (°C)",
    "shortwave_radiation_sum": "shortwave_radiation_sum (MJ/m²)",
    "et0_fao_evapotranspiration": "et0_fao_evapotranspiration (mm)",
    "dewpoint_2m_mean": "dew_point_2m_mean (°C)", "relative_humidity_2m_mean": "relative_humidity_2m_mean (%)",
    "windgusts_10m_max": "wind_gusts_10m_max (km/h)", "windspeed_10m_mean": "wind_speed_10m_mean (km/h)",
    "cloudcover_mean": "cloud_cover_mean (%)", "pressure_msl_mean": "pressure_msl_mean (hPa)",
    "soil_moisture_0_to_7cm_mean": "soil_moisture_0_to_7cm_mean (m³/m³)",
    "soil_moisture_7_to_28cm_mean": "soil_moisture_7_to_28cm_mean (m³/m³)",
    "soil_temperature_0_to_7cm_mean": "soil_temperature_0_to_7cm_mean (°C)",
    "vapor_pressure_deficit_max": "vapour_pressure_deficit_max (kPa)",
}

BASE_DAILY = [
    "temperature_2m_max","temperature_2m_min","temperature_2m_mean","precipitation_sum",
    "rain_sum","precipitation_hours","apparent_temperature_mean","shortwave_radiation_sum",
    "et0_fao_evapotranspiration","dewpoint_2m_mean","relative_humidity_2m_mean",
    "windgusts_10m_max","windspeed_10m_mean","cloudcover_mean","pressure_msl_mean",
    "soil_moisture_0_to_7cm_mean","soil_moisture_7_to_28cm_mean",
    "soil_temperature_0_to_7cm_mean","vapor_pressure_deficit_max",
]


def _fetch_meteo(url, params):
    import requests
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("daily", {}))
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    return df.rename(columns=RENAME_MAP)


def fetch_forecast(lat, lon):
    return _fetch_meteo("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat, "longitude": lon, "daily": ",".join(BASE_DAILY),
        "timezone": "Asia/Kolkata", "forecast_days": 7,
    })


def fetch_historical(lat, lon, days_back=60):
    end   = datetime.now() - timedelta(days=6)
    start = end - timedelta(days=days_back)
    return _fetch_meteo("https://archive-api.open-meteo.com/v1/archive", {
        "latitude": lat, "longitude": lon, "daily": ",".join(BASE_DAILY),
        "timezone": "Asia/Kolkata",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date":   end.strftime("%Y-%m-%d"),
    })


def run_live_prediction(hazard, city, models, scaler, model_name):
    coords = {"heatwave": HEATWAVE_CITIES, "flood": FLOOD_CITIES, "landslide": LANDSLIDE_REGIONS}[hazard][city]
    hist_df     = fetch_historical(coords["lat"], coords["lon"])
    forecast_df = fetch_forecast(coords["lat"], coords["lon"])
    combined = pd.concat([hist_df, forecast_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    loc_col  = "city" if hazard != "landslide" else "region"
    combined[loc_col] = city
    if hazard == "heatwave":
        combined = _add_heat_derived(combined, city)
        from heatwave.feature_engineering import create_all_features
    elif hazard == "flood":
        combined = _add_flood_derived(combined, city)
        from flood.feature_engineering import create_all_features
    else:
        combined = _add_landslide_derived(combined, city)
        from landslide.feature_engineering import create_all_features
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        featured = create_all_features(combined)
    forecast_dates = forecast_df.index
    ff = featured.loc[featured.index.isin(forecast_dates)].copy()
    if ff.empty:
        return None, "Could not align forecast dates."
    fc = get_feature_columns(ff, hazard)
    X  = ff[fc].ffill().fillna(0)
    preds, probas = predict_risk(models, scaler, X, model_name)
    ff["predicted_risk"]  = preds
    ff["predicted_label"] = [RISK_LABELS.get(p, "?") for p in preds]
    ff["confidence"]      = probas.max(axis=1)
    return ff, None

# =============================================================================
# UI HELPERS
# =============================================================================

def _pt(t): return dict(**PLOTLY_TEMPLATE, title_text=t, title_font=dict(color="#f0f4ff", size=13))

def risk_badge(label):
    css = RISK_CSS.get(label, "risk-none")
    dot = {"No Risk":"","Low":"","Moderate":"","High":"","Extreme":""}.get(label,"")
    return html.Span([dot, " ", label], className=f"risk-badge {css}")

def metric_card(label, value, sub="", icon="", cls=""):
    return html.Div([
        html.Div(icon, className="metric-icon"),
        html.Div(label, className="metric-label"),
        html.Div(str(value), className="metric-value"),
        html.Div(sub, className="metric-sub"),
    ], className=f"metric-card {cls}")

def section_label(text):
    return html.Div(text, className="section-label")

def chart_card(title, fig, animate_cls=""):
    return html.Div([
        html.Div(title, className="chart-title"),
        dcc.Graph(
            figure=fig,
            config={"displayModeBar": False, "responsive": True},
            style={"height": "320px"},
            className="responsive-chart"
        ),
    ], className=f"chart-wrapper {animate_cls}")

def _empty_fig(msg="No data available"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(color="#64748b", size=14), xref="paper", yref="paper")
    fig.update_layout(**PLOTLY_TEMPLATE, height=300)
    return fig

# ---------------------------------------------------------------------------
# Navbar (n0x / Apple style minimalist)
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("/",            "", "Overview"),
    ("/heatwave",    "", "Heatwave"),
    ("/flood",       "", "Flood"),
    ("/landslide",   "", "Landslide"),
    ("/performance", "", "Model Performance"),
]

navbar = html.Div([
    html.Div([
        # Brand (Left)
        html.Div([
            html.Span("■", style={"color": "#fff", "marginRight": "10px", "fontSize": "1.1rem"}),
            html.Span("RiskSense", className="brand-text")
        ], className="top-nav-brand"),

        # Hamburger Button for Mobile
        html.Button([
            html.Span(),
            html.Span(),
            html.Span()
        ], id="hamburger-btn", className="hamburger-btn", n_clicks=0),

        # Links and Button (Right)
        html.Div([
            html.Div([
                dcc.Link(
                    name,
                    href=href,
                    className="top-nav-link",
                    id=f"nav-{name.lower().replace(' ','-')}"
                )
                for href, icon, name in NAV_ITEMS
            ], className="top-nav-links"),

            # Action button
            html.Div([
                html.Span(className="status-dot"), "System Online  →"
            ], className="top-nav-btn"),
        ], className="top-nav-right", id="nav-menu"),
    ], className="top-nav-container")
], className="top-navbar")

# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    html.Div(id="page-content", className="fade-in"),
], id="app-wrapper")

# =============================================================================
# DASHBOARD PAGE
# =============================================================================
def layout_dashboard():
    hazards = [
        ("heatwave", "", "Heatwave Risk", "heat", HEATWAVE_CITIES),
        ("flood",    "", "Flood Risk",     "flood", FLOOD_CITIES),
        ("landslide","", "Landslide Risk", "land",  LANDSLIDE_REGIONS),
    ]
    cards = []
    for haz, emoji, title, cls, cities in hazards:
        comp = load_model_comparison(haz)
        if not comp.empty and "f1_weighted" in comp.columns:
            best   = comp["f1_weighted"].idxmax()
            f1     = comp.loc[best, "f1_weighted"]
            csi    = comp.loc[best, "Tuned_CSI"] if "Tuned_CSI" in comp.columns else None
            model_label = best.replace("_"," ").title()
        else:
            best = model_label = f1 = csi = None
        cards.append(dbc.Col(html.Div([
            html.Span(emoji, className="hazard-emoji"),
            html.Div(title, className="hazard-title"),
            html.Div(model_label or "", className="hazard-model"),
            html.Div([
                html.Span("Best F1", className="hazard-stat-label"),
                html.Span(f"{f1:.4f}" if f1 else "", className="hazard-stat-value"),
            ], className="hazard-stat"),
            html.Div([
                html.Span("Tuned CSI", className="hazard-stat-label"),
                html.Span(f"{csi:.4f}" if csi else "", className="hazard-stat-value"),
            ], className="hazard-stat"),
            html.Div([
                html.Span("Locations", className="hazard-stat-label"),
                html.Span(str(len(cities)), className="hazard-stat-value"),
            ], className="hazard-stat"),
        ], className=f"hazard-card {cls}"), md=4))

    loc_cols = [
        (" Heatwave Cities", list(HEATWAVE_CITIES.keys())),
        (" Flood Cities",    list(FLOOD_CITIES.keys())),
        (" Landslide Zones", list(LANDSLIDE_REGIONS.keys())),
    ]

    pipeline_steps = [
        ("1", "Data Collection",    "16 yrs of daily weather from Open-Meteo + hydrological & geomorphic variables."),
        ("2", "Feature Engineering","120200 features per hazard: temporal lags, rolling stats, compound interaction indicators."),
        ("3", "Model Training",     "Random Forest, XGBoost, LightGBM, Gradient Boosting with SMOTE & custom class weights."),
        ("4", "Evaluation",         "WMO-standard verification: POD, FAR, CSI, HSS with optimal probability threshold tuning."),
    ]

    return html.Div([
        # Header
        html.Div([
            html.H1("Disaster Risk Overview", className="page-title"),
            html.P("Multi-hazard early warning covering 15 cities and regions across India.", className="page-subtitle"),
        ], className="page-header"),

        # Hazard cards
        dbc.Row(cards, className="g-4 mb-4"),

        # Locations
        section_label("Monitored Locations"),
        dbc.Row([
            dbc.Col(html.Div([
                html.Div(lbl, style={"fontWeight":"600","fontSize":"0.85rem","color":"#f0f4ff","marginBottom":"10px"}),
                html.Div([html.Span(c, className="location-tag") for c in locs]),
            ], className="glass-card"), md=4) for lbl, locs in loc_cols
        ], className="g-3 mb-4"),

        # Pipeline
        section_label("How It Works"),
        html.Div([
            html.Div([
                html.Div(num, className="step-number"),
                html.Div(t,   className="step-title"),
                html.Div(b,   className="step-body"),
            ], className="pipeline-step") for num, t, b in pipeline_steps
        ], className="pipeline-container"),
    ])

# =============================================================================
# PREDICTION PAGE (Heatwave / Flood / Landslide)
# =============================================================================

# Feature-input schemas per hazard
HEAT_FEATURES = [
    (" Temperature",    [("Max Temp (C)", 20, 50, 38, "temperature_2m_max (C)"), ("Min Temp (C)", 10, 40, 24, "temperature_2m_min (C)"), ("Apparent Temp (C)", 15, 55, 40, "apparent_temperature_mean (C)")]),
    (" Humidity & Dew", [("Relative Humidity (%)", 5, 100, 35, "relative_humidity_2m_mean (%)"), ("Dew Point (C)", 0, 35, 16, "dew_point_2m_mean (C)"), ("Vapour Pressure Deficit (kPa)", 0, 6, 2.5, "vapour_pressure_deficit_max (kPa)")]),
    (" Wind & Radiation",[("Wind Speed (km/h)", 0, 80, 12, "wind_speed_10m_mean (km/h)"), ("Shortwave Radiation (MJ/m)", 0, 40, 22, "shortwave_radiation_sum (MJ/m)"), ("Cloud Cover (%)", 0, 100, 20, "cloud_cover_mean (%)")]),
    (" Precipitation",  [("Precip (mm)", 0, 80, 0, "precipitation_sum (mm)"), ("Soil Moisture 0-7cm", 0.01, 0.5, 0.12, "soil_moisture_0_to_7cm_mean (m/m)")]),
]

FLOOD_FEATURES = [
    (" Rainfall",       [("Precipitation (mm)", 0, 200, 15, "precipitation_sum (mm)"), ("Rain Sum (mm)", 0, 200, 15, "rain_sum (mm)"), ("Precip Hours (h)", 0, 24, 3, "precipitation_hours (h)")]),
    (" Soil & Moisture",[("Soil Moisture 0-7cm (m/m)", 0.01, 0.6, 0.25, "soil_moisture_0_to_7cm_mean (m/m)"), ("Soil Moisture 7-28cm (m/m)", 0.01, 0.6, 0.2, "soil_moisture_7_to_28cm_mean (m/m)")]),
    (" Temperature",    [("Max Temp (C)", 10, 45, 30, "temperature_2m_max (C)"), ("Min Temp (C)", 5, 35, 20, "temperature_2m_min (C)")]),
    (" Atmospheric",   [("Pressure (hPa)", 980, 1030, 1005, "pressure_msl_mean (hPa)"), ("Wind Speed (km/h)", 0, 80, 15, "wind_speed_10m_mean (km/h)"), ("Cloud Cover (%)", 0, 100, 60, "cloud_cover_mean (%)")]),
]

LAND_FEATURES = [
    (" Rainfall",       [("Precipitation (mm)", 0, 250, 30, "precipitation_sum (mm)")]),
    (" Terrain",         [("Slope ()", 5, 60, 28, "slope_deg"), ("Elevation (m)", 500, 4000, 2200, "elevation_m"), ("Soil Depth (m)", 0.5, 5.0, 2.0, "soil_depth_m"), ("Curvature", -1.0, 1.0, 0.1, "curvature")]),
    (" Land & Geology", [("Geology Susceptibility", 0, 1, 0.6, "geology_susceptibility"), ("NDVI", 0, 1, 0.45, "ndvi"), ("Seismic PGA (g)", 0, 0.5, 0.15, "seismic_pga (g)")]),
    (" Distances",      [("Dist. to Road (km)", 0, 5, 0.8, "distance_to_road (km)"), ("Dist. to Stream (km)", 0, 3, 0.5, "distance_to_stream (km)")]),
]

HAZARD_FEATURES = {"heatwave": HEAT_FEATURES, "flood": FLOOD_FEATURES, "landslide": LAND_FEATURES}


def make_feature_sliders(hazard):
    """Build the interactive feature panel for a hazard with grid layout."""
    groups = HAZARD_FEATURES[hazard]
    sections = []
    for group_name, feats in groups:
        items = []
        for label, mn, mx, default, fid in feats:
            step = round((mx - mn) / 100, 4)
            items.append(html.Div([
                html.Div([
                    html.Span(label, style={"fontSize": "11px"}),
                    html.Span(str(default), id=f"val-{hazard}-{fid}", className="feature-item-value")
                ], className="feature-item-label"),
                dcc.Slider(
                    id=f"sl-{hazard}-{fid}",
                    min=mn, max=mx, value=default, step=step,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], className="feature-item"))

        sections.append(html.Div([
            html.Div(group_name, className="feature-section-heading"),
            html.Div(items, className="feature-grid-section"),
        ]))
    return sections


def layout_prediction(hazard):
    cities_dict = {"heatwave": HEATWAVE_CITIES, "flood": FLOOD_CITIES, "landslide": LANDSLIDE_REGIONS}[hazard]
    emoji       = {"heatwave": "", "flood": "", "landslide": ""}[hazard]
    color_accent= {"heatwave": "#ef4444", "flood": "#3b82f6", "landslide": "#8b5cf6"}[hazard]
    desc        = {"heatwave":"Predict heatwave risk using temperature, humidity, radiation and wind features.",
                   "flood":   "Predict flood risk using rainfall intensity, soil saturation and atmospheric variables.",
                   "landslide":"Predict landslide risk combining real-time rainfall with terrain and geological factors."}[hazard]

    try:
        models, scaler = load_ml_models(hazard)
        model_opts = [{"label": m.replace("_"," ").title(), "value": m} for m in models.keys()]
        default_model = next(iter(models.keys())) if models else None
    except Exception:
        model_opts = []
        default_model = None

    city_opts  = [{"label": c, "value": c} for c in cities_dict.keys()]
    default_city = list(cities_dict.keys())[0]

    page_cls = f"prediction-page prediction-page--{hazard}"

    # Left Column - Model Control Card
    model_card = html.Div([
        html.Div([
            html.Div([
                html.Div("Model", style={"fontSize": "11px", "fontWeight": "500", "color": "#666", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id=f"dd-model-{hazard}",
                    options=model_opts,
                    value=default_model,
                    clearable=False,
                    className="dash-dropdown"
                ),
            ]),
            html.Div([
                html.Div("Location", style={"fontSize": "11px", "fontWeight": "500", "color": "#666", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id=f"dd-city-{hazard}",
                    options=city_opts,
                    value=default_city,
                    clearable=False,
                    className="dash-dropdown"
                ),
            ]),
        ], className="model-control-grid"),
        html.Button(
            [" Run Live Forecast — ", html.Span(default_city, id=f"btn-city-label-{hazard}", style={"fontWeight": "700"})],
            id=f"btn-forecast-{hazard}",
            n_clicks=0,
            className="btn-run"
        ),
    ], className=f"model-control-card model-control-card--{hazard}")

    # Left Column - Feature Explorer
    feature_panel = html.Div([
        html.Div([
            html.Div(" Feature Explorer", className="feature-panel-header"),
            html.Div("Adjust parameters to explore manual predictions below.", className="feature-panel-desc"),
        ]),
        *make_feature_sliders(hazard),
        html.Div(style={"height": "10px"}),
        html.Button(" Predict from Sliders", id=f"btn-manual-{hazard}", n_clicks=0, className="btn-run"),
        html.Div(id=f"manual-result-{hazard}", style={"marginTop": "12px"}),
    ], className=f"feature-panel feature-panel--{hazard}")

    # Right Column - Results Panel
    results_panel = html.Div([
        html.Div([
            dcc.Tabs(id=f"tabs-{hazard}", value="live", children=[
                dcc.Tab(label=" Live Forecast", value="live", className="tab", selected_className="tab--selected"),
                dcc.Tab(label=" Historical Analysis", value="historical", className="tab", selected_className="tab--selected"),
            ], className="custom-tabs"),
        ], className="results-panel-tabs"),
        html.Div(
            html.Div(
                [
                    html.Div("○", className="results-empty-icon"),
                    html.Div("Run a forecast to see results here", className="results-empty-text"),
                ],
                id=f"tab-content-{hazard}",
                className="results-panel-inner",
            ),
            className="results-panel-content",
        ),
    ], className=f"results-panel results-panel--{hazard}")

    return html.Div([
        # Header row
        html.Div([
            html.H1([emoji, " ", hazard.title(), " Risk Prediction"], className="page-title"),
            html.P(desc, className="page-subtitle"),
        ], className="page-header"),

        # Two-column grid
        html.Div([
            # Left column
            html.Div([
                model_card,
                feature_panel,
            ], className=f"prediction-left-col prediction-left-col--{hazard}"),

            # Right column
            results_panel,
        ], className=f"prediction-grid prediction-grid--{hazard}"),
    ], className=page_cls)

# =============================================================================
# MODEL PERFORMANCE PAGE
# =============================================================================
def layout_performance():
    return html.Div([
        html.Div([
            html.H1(" Model Performance", className="page-title"),
            html.P("Compare all trained models across accuracy, F1, POD, FAR, CSI and HSS metrics.", className="page-subtitle"),
        ], className="page-header"),
        html.Div([
            html.Div("Hazard Type", className="feature-label"),
            dcc.Dropdown(id="dd-perf-hazard",
                         options=[{"label":"  Heatwave","value":"heatwave"},
                                  {"label":"  Flood",    "value":"flood"},
                                  {"label":"  Landslide","value":"landslide"}],
                         value="heatwave", clearable=False, className="dash-dropdown",
                         style={"maxWidth":"320px"}),
        ], className="glass-card mb-4", style={"maxWidth":"400px"}),
        html.Div(id="perf-content"),
    ])

# =============================================================================
# CALLBACKS
# =============================================================================

#  1. Routing
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname in (None, "/", ""):
        return layout_dashboard()
    elif pathname == "/heatwave":
        return layout_prediction("heatwave")
    elif pathname == "/flood":
        return layout_prediction("flood")
    elif pathname == "/landslide":
        return layout_prediction("landslide")
    elif pathname == "/performance":
        return layout_performance()
    return html.Div([html.H2("404  Page not found", className="page-title")], style={"padding":"60px"})


# 1.1 Mobile Menu Toggle
@app.callback(
    [Output("nav-menu", "className"),
     Output("hamburger-btn", "className")],
    Input("hamburger-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_mobile_menu(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return "top-nav-right active", "hamburger-btn active"
    return "top-nav-right", "hamburger-btn"


# 1.2 Close mobile menu when navigating
app.clientside_callback(
    """
    function(pathname) {
        return ['top-nav-right', 'hamburger-btn'];
    }
    """,
    [Output("nav-menu", "className", allow_duplicate=True),
     Output("hamburger-btn", "className", allow_duplicate=True)],
    Input("url", "pathname"),
    prevent_initial_call=True
)


#  2. Forecast + Historical tabs  for each hazard 
def _forecast_charts(result, hazard):
    max_risk  = int(result["predicted_risk"].max())
    high_days = int((result["predicted_risk"] >= 3).sum())
    avg_conf  = result["confidence"].mean()
    period    = f"{result.index[0].strftime('%d %b')}  {result.index[-1].strftime('%d %b')}"

    metrics_row = dbc.Row([
        dbc.Col(metric_card("Peak Risk",       RISK_LABELS.get(max_risk,"?"), icon=""), md=3),
        dbc.Col(metric_card("High+ Risk Days", f"{high_days}/{len(result)}",  icon=""), md=3),
        dbc.Col(metric_card("Avg Confidence",  f"{avg_conf:.0%}",             icon=""), md=3),
        dbc.Col(metric_card("Forecast Period", period,                        icon=""), md=3),
    ], className="g-3 mb-4")

    # Risk bar chart
    colors = [RISK_COLORS_LIST[int(r)] for r in result["predicted_risk"]]
    fig_risk = go.Figure(go.Bar(
        x=result.index.strftime("%a %d %b"), y=result["predicted_risk"],
        marker_color=colors, text=result["predicted_label"], textposition="outside",
        customdata=result["confidence"],
        hovertemplate="Date: %{x}<br>Risk: %{text}<br>Confidence: %{customdata:.1%}",
    ))
    fig_risk.update_layout(**PLOTLY_TEMPLATE, title_text="7-Day Risk Forecast",
                           xaxis={**AXIS_DEFAULTS},
                           yaxis={**AXIS_DEFAULTS, "title": "Risk Level", "tickvals": [0,1,2,3,4],
                                  "ticktext": ["No Risk","Low","Moderate","High","Extreme"], "range": [-0.3,5]},
                           height=300)

    if hazard == "heatwave":
        max_temp_col = "temperature_2m_max (C)"
        humidity_col = "relative_humidity_2m_mean (%)"
        wind_col = "wind_speed_10m_max (km/h)"

        weather_cards = dbc.Row([
            dbc.Col(
                metric_card(
                    "Max Temp",
                    f"{result[max_temp_col].max():.1f}°C" if max_temp_col in result.columns else "N/A",
                ),
                md=4
            ),
            dbc.Col(
                metric_card(
                    "Avg Humidity",
                    f"{result[humidity_col].mean():.0f}%" if humidity_col in result.columns else "N/A",
                ),
                md=4
            ),
            dbc.Col(
                metric_card(
                    "Max Wind",
                    f"{result[wind_col].max():.1f} km/h" if wind_col in result.columns else "N/A",
                ),
                md=4
            ),
        ], className="g-2")

        return html.Div([
            html.Div(metrics_row, className="results-section-metrics"),
            html.Div(chart_card("", fig_risk, "fade-in-1"), className="results-section-chart"),
            html.Div([
                html.Div("Weather Context", className="results-section-label"),
                weather_cards,
            ], className="results-section-weather"),
        ], className="results-live-content")

    # Weather context
    fig_wx = make_subplots(specs=[[{"secondary_y": True}]])
    if "temperature_2m_max (C)" in result.columns:
        fig_wx.add_trace(go.Scatter(x=result.index, y=result["temperature_2m_max (C)"], name="Max Temp C", line=dict(color="#ef4444", width=2)), secondary_y=False)
    if "temperature_2m_min (C)" in result.columns:
        fig_wx.add_trace(go.Scatter(x=result.index, y=result["temperature_2m_min (C)"], name="Min Temp C", line=dict(color="#3b82f6", width=2)), secondary_y=False)
    if "precipitation_sum (mm)" in result.columns:
        fig_wx.add_trace(go.Bar(x=result.index, y=result["precipitation_sum (mm)"], name="Precip mm", marker_color="#06b6d4", opacity=0.55), secondary_y=True)
    fig_wx.update_layout(**PLOTLY_TEMPLATE, title_text="Weather Context", height=280,
                         xaxis={**AXIS_DEFAULTS},
                         yaxis={**AXIS_DEFAULTS, "title": "Temp (°C)"},
                         yaxis2={**AXIS_DEFAULTS, "title": "Precip (mm)"})

    # Confidence bar
    fig_conf = go.Figure(go.Bar(
        x=result.index.strftime("%a %d %b"), y=result["confidence"],
        marker_color=["#6366f1"]*len(result),
        text=[f"{c:.0%}" for c in result["confidence"]], textposition="outside",
    ))
    fig_conf.update_layout(**PLOTLY_TEMPLATE, title_text="Prediction Confidence", height=260,
                           xaxis={**AXIS_DEFAULTS},
                           yaxis={**AXIS_DEFAULTS, "title": "Confidence", "range": [0, 1.2]})

    # Pie
    rc = result["predicted_label"].value_counts()
    fig_pie = px.pie(names=rc.index, values=rc.values,
                     color=rc.index, color_discrete_map=RISK_COLORS, hole=0.45)
    fig_pie.update_layout(**PLOTLY_TEMPLATE, title_text="Risk Distribution", height=260, showlegend=True)

    tbl_cols = ["predicted_label","confidence"] + [c for c in ["temperature_2m_max (C)","precipitation_sum (mm)","relative_humidity_2m_mean (%)"] if c in result.columns]
    tbl_data = result[tbl_cols].copy()
    tbl_data.index = tbl_data.index.strftime("%a %d %b")
    tbl_data = tbl_data.reset_index().rename(columns={"time":"Date","index":"Date","predicted_label":"Risk","confidence":"Confidence"})
    tbl_data["Confidence"] = tbl_data["Confidence"].map("{:.1%}".format)

    table = dash_table.DataTable(
        data=tbl_data.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tbl_data.columns],
        style_header={"backgroundColor":"rgba(99,102,241,0.12)","color":"#94a3b8","fontWeight":"700","fontSize":"11px","textTransform":"uppercase","border":"none"},
        style_cell={"backgroundColor":"rgba(255,255,255,0.03)","color":"#f0f4ff","border":"none","fontSize":"13px","padding":"10px 14px"},
        style_data_conditional=[{"if":{"filter_query":'{Risk} = "High" || {Risk} = "Extreme"'},"color":"#ef4444","fontWeight":"700"}],
        page_size=7, style_table={"overflowX":"auto"},
    )

    return html.Div([
        metrics_row,
        dbc.Row([
            dbc.Col(chart_card("", fig_risk, "fade-in-1"), md=12),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(chart_card("", fig_wx,   "fade-in-2"), md=12),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(chart_card("", fig_conf, "fade-in-3"), md=6),
            dbc.Col(chart_card("", fig_pie,  "fade-in-4"), md=6),
        ], className="mb-3"),
        section_label("Daily Breakdown"),
        table,
    ])


def _historical_charts(hazard, city, model_name):
    try:
        models, scaler = load_ml_models(hazard)
        df = load_featured_data(hazard)
    except Exception as e:
        return html.Div(f"Error loading data: {e}", className="stat-banner stat-banner-error")

    loc_col = "city" if hazard != "landslide" else "region"
    city_df = df[df[loc_col] == city].copy() if loc_col in df.columns else df.copy()
    if city_df.empty:
        return html.Div(f"No historical data for {city}.", className="stat-banner stat-banner-warn")

    feat_cols = get_feature_columns(city_df, hazard)
    X = city_df[feat_cols].ffill().fillna(0)
    try:
        preds, probas = predict_risk(models, scaler, X, model_name)
    except Exception as e:
        return html.Div(f"Prediction error: {e}", className="stat-banner stat-banner-error")

    city_df = city_df.copy()
    city_df["predicted_risk"]  = preds
    city_df["predicted_label"] = [RISK_LABELS.get(p,"?") for p in preds]
    city_df["confidence"]      = probas.max(axis=1)
    actual = city_df["risk_level"] if "risk_level" in city_df.columns else None

    # Metrics
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(actual, preds) if actual is not None else None
    f1  = f1_score(actual, preds, average="weighted", zero_division=0) if actual is not None else None

    metrics_row = dbc.Row([
        dbc.Col(metric_card("Accuracy",      f"{acc:.4f}" if acc else ""), md=3),
        dbc.Col(metric_card("Weighted F1",   f"{f1:.4f}"  if f1  else ""), md=3),
        dbc.Col(metric_card("High+ Risk Days", str(int((preds >= 3).sum()))), md=3),
        dbc.Col(metric_card("Avg Confidence", f"{city_df['confidence'].mean():.2%}"), md=3),
    ], className="g-3 mb-4")

    # Timeline
    fig_tl = go.Figure()
    if actual is not None:
        fig_tl.add_trace(go.Scatter(x=city_df.index, y=actual, name="Actual", line=dict(color="#3b82f6",width=1.5)))
    fig_tl.add_trace(go.Scatter(x=city_df.index, y=preds, name="Predicted", line=dict(color="#ef4444",width=1.5,dash="dot")))
    fig_tl.update_layout(**PLOTLY_TEMPLATE, title_text=f"Risk Timeline — {city}", height=300,
                         xaxis={**AXIS_DEFAULTS},
                         yaxis={**AXIS_DEFAULTS, "title": "Risk Level", "tickvals": [0,1,2,3,4],
                                "ticktext": ["No Risk","Low","Moderate","High","Extreme"]})

    # Dist
    rc = pd.Series(preds).map(RISK_LABELS).value_counts()
    fig_dist = px.pie(names=rc.index, values=rc.values, color=rc.index,
                      color_discrete_map=RISK_COLORS, hole=0.45)
    fig_dist.update_layout(**PLOTLY_TEMPLATE, title_text="Predicted Risk Distribution", height=280,
                           xaxis={**AXIS_DEFAULTS}, yaxis={**AXIS_DEFAULTS})

    # Hist
    fig_hist = px.histogram(city_df, x="confidence", nbins=25, color_discrete_sequence=["#111"])
    fig_hist.update_layout(**PLOTLY_TEMPLATE, title_text="Confidence Distribution", height=280,
                           xaxis={**AXIS_DEFAULTS, "title": "Confidence Score"},
                           yaxis={**AXIS_DEFAULTS, "title": "Days"})

    tbl = dash_table.DataTable(
        data=city_df[["predicted_label","confidence"]].tail(30).reset_index().rename(columns={"time":"Date","index":"Date","predicted_label":"Risk","confidence":"Conf"}).assign(Conf=lambda d: d["Conf"].map("{:.1%}".format)).to_dict("records"),
        columns=[{"name": c, "id": c} for c in ["Date","Risk","Conf"]],
        style_header={"backgroundColor":"rgba(99,102,241,0.12)","color":"#94a3b8","fontWeight":"700","fontSize":"11px","border":"none"},
        style_cell={"backgroundColor":"rgba(255,255,255,0.03)","color":"#f0f4ff","border":"none","fontSize":"13px","padding":"10px 14px"},
        page_size=10, style_table={"overflowX":"auto"},
    )

    return html.Div([
        metrics_row,
        chart_card("", fig_tl, "fade-in-1"),
        dbc.Row([
            dbc.Col(chart_card("", fig_dist, "fade-in-2"), md=6),
            dbc.Col(chart_card("", fig_hist, "fade-in-3"), md=6),
        ], className="mb-3"),
        section_label("Last 30 Days"),
        tbl,
    ])


def register_prediction_callbacks(hazard):
    # Update button label when city changes
    @app.callback(
        Output(f"btn-city-label-{hazard}", "children"),
        Input(f"dd-city-{hazard}", "value"),
    )
    def update_button_label(city):
        return city or ""

    @app.callback(
        Output(f"tab-content-{hazard}", "children"),
        Input(f"tabs-{hazard}", "value"),
        Input(f"btn-forecast-{hazard}", "n_clicks"),
        State(f"dd-model-{hazard}", "value"),
        State(f"dd-city-{hazard}",  "value"),
        prevent_initial_call=True,
    )
    def update_tab(tab, n_clicks, model_name, city):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        if tab == "historical":
            return _historical_charts(hazard, city, model_name)

        # Live tab - show empty state or forecast results
        if f"btn-forecast-{hazard}" not in triggered:
            return html.Div([
                html.Div("○", className="results-empty-icon"),
                html.Div("Run a forecast to see results here", className="results-empty-text"),
            ], className="results-empty-state")

        # Live forecast
        try:
            models, scaler = load_ml_models(hazard)
            result, err = run_live_prediction(hazard, city, models, scaler, model_name)
            if err:
                return html.Div(f" {err}", className="stat-banner stat-banner-warn", style={"margin": "0"})
            return _forecast_charts(result, hazard)
        except Exception as e:
            import traceback
            return html.Div([
                html.Div(f" Error: {str(e)}", className="stat-banner stat-banner-error", style={"margin": "0"}),
                html.Pre(traceback.format_exc(), style={"fontSize":"11px","color":"#64748b","marginTop":"8px"}),
            ])

    @app.callback(
        Output(f"manual-result-{hazard}", "children"),
        Input(f"btn-manual-{hazard}", "n_clicks"),
        State(f"dd-model-{hazard}", "value"),
        State(f"dd-city-{hazard}",  "value"),
        [State(f"sl-{hazard}-{fid}", "value") for grp in HAZARD_FEATURES[hazard] for lbl, mn, mx, dflt, fid in grp[1]],
        prevent_initial_call=True,
    )
    def manual_predict(n_clicks, model_name, city, *slider_vals):
        if not n_clicks:
            return no_update
        try:
            models, scaler = load_ml_models(hazard)
            feature_ids = [fid for grp in HAZARD_FEATURES[hazard] for lbl, mn, mx, dflt, fid in grp[1]]
            row = {fid: val for fid, val in zip(feature_ids, slider_vals)}
            expected = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else []
            for c in expected:
                if c not in row:
                    row[c] = 0.0
            X = pd.DataFrame([row])[expected] if expected else pd.DataFrame([row])
            X_sc  = pd.DataFrame(scaler.transform(X), columns=X.columns)
            model = models[model_name]
            pred  = int(model.predict(X_sc)[0])
            proba = model.predict_proba(X_sc)[0]
            label = RISK_LABELS.get(pred, "?")
            css   = RISK_CSS.get(label, "risk-none")
            return html.Div([
                html.Div("Manual Prediction Result", className="section-label"),
                html.Div([
                    html.Span("Risk Level: ", style={"color":"#94a3b8","fontSize":"0.85rem"}),
                    html.Span([label], className=f"risk-badge {css}"),
                ], style={"marginBottom":"10px"}),
                html.Div([
                    html.Span("Confidence: ", style={"color":"#94a3b8","fontSize":"0.85rem"}),
                    html.Span(f"{proba.max():.1%}", style={"color":"#6366f1","fontWeight":"700","fontFamily":"JetBrains Mono,monospace"}),
                ]),
                html.Div([
                    dbc.Progress(value=int(pred / 4 * 100), color={0:"success",1:"warning",2:"warning",3:"danger",4:"danger"}.get(pred,"info"),
                                 className="mt-2", style={"height":"6px","borderRadius":"99px","background":"rgba(255,255,255,0.08)"}),
                ]),
            ])
        except Exception as e:
            return html.Div(f"Error: {e}", className="stat-banner stat-banner-error")


for _h in ("heatwave", "flood", "landslide"):
    register_prediction_callbacks(_h)


#  3. Model Performance 
@app.callback(Output("perf-content", "children"), Input("dd-perf-hazard", "value"))
def update_performance(hazard):
    comp = load_model_comparison(hazard)
    if comp.empty:
        return html.Div("No model comparison data found. Run the pipeline first.", className="stat-banner stat-banner-warn")

    default_metrics = ["accuracy","f1_weighted","POD","FAR","CSI","HSS"]
    available = [m for m in default_metrics if m in comp.columns]
    tuned     = [c for c in comp.columns if c.startswith("Tuned_")]

    tbl_main = dash_table.DataTable(
        data=[{"Model": idx.replace("_"," ").title(), **{m: round(float(comp.loc[idx,m]),4) for m in available}} for idx in comp.index],
        columns=[{"name": c, "id": c} for c in ["Model"] + available],
        style_header={"backgroundColor":"rgba(99,102,241,0.15)","color":"#94a3b8","fontWeight":"700","fontSize":"11px","border":"none"},
        style_cell={"backgroundColor":"rgba(255,255,255,0.03)","color":"#f0f4ff","border":"none","fontSize":"13px","padding":"10px 16px"},
        style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"rgba(255,255,255,0.015)"}],
        style_table={"overflowX":"auto"},
    )

    # Bar chart
    chart_metrics = [m for m in ["accuracy","f1_weighted","POD","CSI"] if m in comp.columns]
    melted = comp[chart_metrics].reset_index().melt(id_vars=comp.index.name or "index", var_name="Metric", value_name="Score")
    melted = melted.rename(columns={melted.columns[0]: "Model"})
    melted["Model"] = melted["Model"].str.replace("_"," ").str.title()
    fig_bar = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=["#6366f1","#8b5cf6","#06b6d4","#10b981"])
    fig_bar.update_layout(**PLOTLY_TEMPLATE, title_text="Metric Comparison", height=340)

    blocks = [
        section_label("Default Metrics"),
        tbl_main,
        chart_card("", fig_bar, "fade-in-1"),
    ]

    if tuned:
        tbl_tuned = dash_table.DataTable(
            data=[{"Model": idx.replace("_"," ").title(), **{m: round(float(comp.loc[idx,m]),4) for m in tuned}} for idx in comp.index],
            columns=[{"name": c.replace("Tuned_",""), "id": c} for c in ["Model"] + tuned] if False else [{"name": c, "id": c} for c in ["Model"] + tuned],
            style_header={"backgroundColor":"rgba(99,102,241,0.15)","color":"#94a3b8","fontWeight":"700","fontSize":"11px","border":"none"},
            style_cell={"backgroundColor":"rgba(255,255,255,0.03)","color":"#f0f4ff","border":"none","fontSize":"13px","padding":"10px 16px"},
            style_table={"overflowX":"auto"},
        )
        blocks += [section_label("Threshold-Tuned Metrics"), tbl_tuned]

    if "Tuned_CSI" in comp.columns and "Tuned_POD" in comp.columns:
        fig_sc = go.Figure()
        for model in comp.index:
            fig_sc.add_trace(go.Scatter(
                x=[comp.loc[model,"Tuned_POD"]], y=[comp.loc[model,"Tuned_CSI"]],
                mode="markers+text", text=[model.replace("_"," ").title()],
                textposition="top center", marker=dict(size=16, symbol="circle"),
                name=model.replace("_"," ").title(),
            ))
        fig_sc.update_layout(**PLOTLY_TEMPLATE, title_text="Tuned: POD vs CSI", height=340,
                             xaxis_title="Probability of Detection (POD)",
                             yaxis_title="Critical Success Index (CSI)")
        blocks.append(chart_card("", fig_sc, "fade-in-2"))

    return html.Div(blocks)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app.run(debug=False, port=8050, host="127.0.0.1")
