"""
Feature Engineering Module for Landslide Risk Prediction

Creates ~180 features across seven categories:
  1. Rainfall dynamics          — intensity, antecedent, I-D thresholds
  2. Geomorphological features  — slope derivatives, curvature, aspect  
  3. Soil / hydrological        — moisture, pore pressure, saturation index
  4. Geological / seismic       — susceptibility, PGA, quake-rain interaction
  5. Vegetation / land use      — NDVI, deforestation proxy, land-use encoding
  6. Temporal / seasonal        — monsoon indicators, day-of-year, cyclicity
  7. Compound interaction       — rain×slope, API×soil_moisture, etc.

References:
  - Froude & Petley (2018) — Global fatal landslide occurrence
  - Dikshit et al. (2020) — Rainfall thresholds for Indian landslides
  - GSI (2009) — Macro-zonation methodology for India
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


def create_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rainfall intensity, antecedent, and duration features."""
    rain = 'precipitation_sum (mm)'

    # Rolling statistics (multiple windows)
    for w in [3, 5, 7, 14]:
        df[f'rain_roll_mean_{w}d'] = df[rain].rolling(w, min_periods=1).mean()
        df[f'rain_roll_max_{w}d'] = df[rain].rolling(w, min_periods=1).max()
        df[f'rain_roll_sum_{w}d'] = df[rain].rolling(w, min_periods=1).sum()
        df[f'rain_roll_std_{w}d'] = df[rain].rolling(w, min_periods=1).std().fillna(0)

    # Antecedent precipitation index (already in data, but add decay variants)
    api = df['antecedent_precip_index (mm)']
    df['api_normalized'] = api / (api.max() + 1e-6)
    df['api_squared'] = api ** 2

    # Rainfall rate / intensity classification (IMD scheme)
    df['rain_intensity_class'] = pd.cut(
        df[rain],
        bins=[-1, 2.5, 7.5, 35.5, 64.5, 115.5, 204.4, 9999],
        labels=[0, 1, 2, 3, 4, 5, 6]
    ).astype(int)
    # 0=no rain, 1=very light, 2=light, 3=moderate, 4=heavy,
    # 5=very heavy, 6=extremely heavy

    # Consecutive rain days
    is_rainy = (df[rain] > 2.5).astype(int)
    df['consecutive_rain_days'] = is_rainy.groupby(
        (is_rainy != is_rainy.shift()).cumsum()
    ).cumsum() * is_rainy

    # Dry spell before rain (important for desiccation cracks)
    is_dry = (df[rain] <= 2.5).astype(int)
    df['dry_spell_days'] = is_dry.groupby(
        (is_dry != is_dry.shift()).cumsum()
    ).cumsum() * is_dry

    # Rain change (acceleration)
    df['rain_change_1d'] = df[rain].diff(1).fillna(0)
    df['rain_change_3d'] = df[rain].diff(3).fillna(0)

    # Intensity-Duration product (proxy for I-D threshold exceedance)
    df['intensity_duration_3d'] = df[rain] * df['consecutive_rain_days'].clip(upper=7)
    df['intensity_duration_7d'] = df[f'rain_roll_mean_7d'] * np.minimum(
        df['consecutive_rain_days'], 14
    )

    return df


def create_geomorphological_features(df: pd.DataFrame) -> pd.DataFrame:
    """Slope, aspect, curvature, and terrain-derived features."""

    # Slope categories (GSI zonation thresholds)
    df['slope_class'] = pd.cut(
        df['slope_deg'],
        bins=[0, 15, 25, 35, 45, 90],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    # 0=gentle, 1=moderate, 2=moderately steep, 3=steep, 4=very steep

    df['slope_radians'] = np.radians(df['slope_deg'])
    df['slope_tangent'] = np.tan(df['slope_radians'])

    # Aspect-derived features (rainfall direction matters in monsoon)
    aspect_rad = np.radians(df['aspect_deg'])
    df['aspect_sin'] = np.sin(aspect_rad).round(4)
    df['aspect_cos'] = np.cos(aspect_rad).round(4)
    # Southward / SW-facing slopes get more monsoon rain in Himalayas
    df['monsoon_exposure'] = np.clip(
        np.cos(aspect_rad - np.radians(225)),  # SW direction
        0, 1
    ).round(4)

    # Curvature categories
    df['curvature_class'] = pd.cut(
        df['curvature'],
        bins=[-1.1, -0.3, 0.3, 1.1],
        labels=[0, 1, 2]  # 0=convex, 1=flat, 2=concave (water collecting)
    ).astype(int)

    # Elevation derivatives
    df['elevation_normalized'] = (
        (df['elevation_m'] - df['elevation_m'].min()) /
        (df['elevation_m'].max() - df['elevation_m'].min() + 1e-6)
    )

    # Topographic wetness proxy (TWI ∝ ln(A/tan(slope)))
    # We use curvature as contributing-area proxy
    df['twi_proxy'] = np.log(
        (df['curvature'].clip(lower=0.01) + 1) /
        (df['slope_tangent'].clip(lower=0.01))
    ).round(4)

    return df


def create_soil_hydro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Soil moisture, pore pressure, and hydrological features."""
    sm = 'soil_moisture_0_to_7cm_mean (m³/m³)'
    pwp = 'pore_water_pressure (kPa)'

    # Soil moisture rolling stats
    for w in [3, 5, 7]:
        df[f'sm_roll_mean_{w}d'] = df[sm].rolling(w, min_periods=1).mean()
        df[f'sm_roll_max_{w}d'] = df[sm].rolling(w, min_periods=1).max()

    # Saturation index (0-1, based on typical porosity ~0.5)
    df['saturation_index'] = (df[sm] / 0.50).clip(upper=1.0).round(4)

    # Soil moisture change rate
    df['sm_change_1d'] = df[sm].diff(1).fillna(0)
    df['sm_change_3d'] = df[sm].diff(3).fillna(0)

    # Pore pressure features
    df['pwp_normalized'] = (
        df[pwp] / (df[pwp].max() + 1e-6)
    ).round(4)
    for w in [3, 5]:
        df[f'pwp_roll_mean_{w}d'] = df[pwp].rolling(w, min_periods=1).mean()
        df[f'pwp_roll_max_{w}d'] = df[pwp].rolling(w, min_periods=1).max()

    # Factor-of-Safety proxy:
    # FoS ∝ (cohesion + (γ·z·cos²α - u)·tanφ) / (γ·z·sinα·cosα)
    # Simplified: lower → more unstable
    gamma_z = 18 * df['soil_depth_m']  # unit weight * depth (kN/m²)
    cos_a = np.cos(np.radians(df['slope_deg']))
    sin_a = np.sin(np.radians(df['slope_deg']))
    cohesion = 15  # kPa (typical residual for weathered rock)
    phi = np.radians(28)  # friction angle

    normal_stress = gamma_z * cos_a ** 2 - df[pwp]
    shear_resistance = cohesion + normal_stress.clip(lower=0) * np.tan(phi)
    driving_force = gamma_z * sin_a * cos_a

    df['fos_proxy'] = (shear_resistance / driving_force.clip(lower=0.1)).round(4)
    df['fos_proxy'] = df['fos_proxy'].clip(0, 10)

    # Effective stress proxy
    df['effective_stress'] = (gamma_z * cos_a ** 2 - df[pwp]).clip(lower=0).round(2)

    return df


def create_geological_seismic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Geological susceptibility and seismic features."""
    pga = 'seismic_pga (g)'

    # Seismic rolling max (aftershock window)
    df['pga_roll_max_3d'] = df[pga].rolling(3, min_periods=1).max()
    df['pga_roll_max_7d'] = df[pga].rolling(7, min_periods=1).max()

    # Was there a significant seismic event recently?
    df['recent_quake_3d'] = (df['pga_roll_max_3d'] > 0.05).astype(int)
    df['recent_quake_7d'] = (df['pga_roll_max_7d'] > 0.03).astype(int)

    # Geological susceptibility bins
    df['geo_susc_class'] = pd.cut(
        df['geology_susceptibility'],
        bins=[0, 0.3, 0.5, 0.7, 1.01],
        labels=[0, 1, 2, 3]
    ).astype(int)
    # 0=low, 1=moderate, 2=high, 3=very high

    return df


def create_vegetation_landuse_features(df: pd.DataFrame) -> pd.DataFrame:
    """NDVI and land-use derived features."""

    # NDVI rolling mean
    for w in [7, 14, 30]:
        df[f'ndvi_roll_mean_{w}d'] = df['ndvi'].rolling(w, min_periods=1).mean()

    # NDVI change (deforestation / degradation detector)
    df['ndvi_change_7d'] = df['ndvi'].diff(7).fillna(0)
    df['ndvi_change_30d'] = df['ndvi'].diff(30).fillna(0)

    # Low vegetation indicator
    df['low_vegetation'] = (df['ndvi'] < 0.3).astype(int)

    # One-hot encode land use
    for lu_val, lu_name in [(0, 'forest'), (1, 'agriculture'), (2, 'barren'), (3, 'settlement')]:
        df[f'lu_{lu_name}'] = (df['land_use'] == lu_val).astype(int)

    # Vegetation protective factor (forest > agriculture > settlement > barren)
    veg_protection = {0: 1.0, 1: 0.5, 2: 0.1, 3: 0.3}
    df['vegetation_protection'] = df['land_use'].map(veg_protection).fillna(0.5)

    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal and seasonal features."""
    idx = df.index

    df['month'] = idx.month
    df['day_of_year'] = idx.dayofyear
    df['year'] = idx.year

    # Monsoon indicator (JJAS for most Himalayan regions + JJASO for Western Ghats)
    df['is_monsoon'] = idx.month.isin([6, 7, 8, 9]).astype(int)
    df['is_pre_monsoon'] = idx.month.isin([4, 5]).astype(int)
    df['is_post_monsoon'] = idx.month.isin([10, 11]).astype(int)

    # Cyclic encoding for month / doy
    df['month_sin'] = np.sin(2 * np.pi * idx.month / 12).round(4)
    df['month_cos'] = np.cos(2 * np.pi * idx.month / 12).round(4)
    df['doy_sin'] = np.sin(2 * np.pi * idx.dayofyear / 365).round(4)
    df['doy_cos'] = np.cos(2 * np.pi * idx.dayofyear / 365).round(4)

    return df


def create_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance-based features."""
    # Inverse distance (closer = higher risk)
    df['inv_dist_road'] = (1.0 / (df['distance_to_road (km)'] + 0.1)).round(4)
    df['inv_dist_stream'] = (1.0 / (df['distance_to_stream (km)'] + 0.1)).round(4)

    # Near-road indicator
    df['near_road'] = (df['distance_to_road (km)'] < 0.5).astype(int)
    df['near_stream'] = (df['distance_to_stream (km)'] < 0.3).astype(int)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compound / interaction features — the most predictive for landslides.
    Rain × slope, API × soil_moisture, etc.
    """
    rain = 'precipitation_sum (mm)'
    sm = 'soil_moisture_0_to_7cm_mean (m³/m³)'
    api = 'antecedent_precip_index (mm)'
    pwp = 'pore_water_pressure (kPa)'

    # Rain × slope (steep + rain = most dangerous)
    df['rain_slope_interaction'] = (df[rain] * df['slope_deg'] / 100).round(4)
    df['rain7d_slope'] = (df['rain_roll_sum_7d'] * df['slope_deg'] / 1000).round(4)

    # Rain × soil moisture (rain on wet soil)
    df['rain_sm_interaction'] = (df[rain] * df[sm]).round(4)
    df['rain7d_sm'] = (df['rain_roll_sum_7d'] * df[sm]).round(4)

    # API × slope × soil moisture (triple interaction — key predictor)
    df['api_slope_sm'] = (
        df[api] * df['slope_deg'] * df[sm] / 1000
    ).round(4)

    # Geological susceptibility × rainfall
    df['geo_rain'] = (df['geology_susceptibility'] * df[rain]).round(4)
    df['geo_rain7d'] = (df['geology_susceptibility'] * df['rain_roll_sum_7d']).round(4)

    # Seismic × rain (earthquake + rain = worst combo)
    df['seismic_rain'] = (df['seismic_pga (g)'] * df[rain] * 100).round(4)

    # FoS below threshold indicator
    df['fos_critical'] = (df['fos_proxy'] < 1.2).astype(int)
    df['fos_very_critical'] = (df['fos_proxy'] < 1.0).astype(int)

    # Pore pressure × slope
    df['pwp_slope'] = (df[pwp] * df['slope_deg'] / 100).round(4)

    # Landslide potential index (composite)
    df['landslide_potential_index'] = (
        df['api_normalized'] * 0.25 +
        df['saturation_index'] * 0.20 +
        (df['slope_deg'] / 70) * 0.20 +
        df['geology_susceptibility'] * 0.15 +
        (1 - df['vegetation_protection']) * 0.10 +
        df['pwp_normalized'] * 0.10
    ).round(4)

    # Cumulative instability score (rolling 7d sum of LPI)
    df['instability_7d'] = df['landslide_potential_index'].rolling(7, min_periods=1).sum().round(4)

    # NDVI-rain interaction (low vegetation + rain = worse)
    df['rain_low_veg'] = (df[rain] * (1 - df['ndvi'])).round(4)

    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    Args:
        df: Combined DataFrame with all regions (from preprocessing)

    Returns:
        DataFrame with all features added
    """
    print("Creating landslide features...")
    n_original = df.shape[1]

    df = create_rainfall_features(df)
    print(f"  Rainfall features: {df.shape[1] - n_original} new → {df.shape[1]} total")
    n = df.shape[1]

    df = create_geomorphological_features(df)
    print(f"  Geomorphological features: {df.shape[1] - n} new → {df.shape[1]} total")
    n = df.shape[1]

    df = create_soil_hydro_features(df)
    print(f"  Soil/Hydro features: {df.shape[1] - n} new → {df.shape[1]} total")
    n = df.shape[1]

    df = create_geological_seismic_features(df)
    print(f"  Geological/Seismic features: {df.shape[1] - n} new → {df.shape[1]} total")
    n = df.shape[1]

    df = create_vegetation_landuse_features(df)
    print(f"  Vegetation/Land-use features: {df.shape[1] - n} new → {df.shape[1]} total")
    n = df.shape[1]

    df = create_temporal_features(df)
    print(f"  Temporal features: {df.shape[1] - n} new → {df.shape[1]} total")
    n = df.shape[1]

    df = create_proximity_features(df)
    print(f"  Proximity features: {df.shape[1] - n} new → {df.shape[1]} total")
    n = df.shape[1]

    df = create_interaction_features(df)
    print(f"  Interaction features: {df.shape[1] - n} new → {df.shape[1]} total")

    # Handle any remaining NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"\nTotal features: {df.shape[1]} columns ({df.shape[1] - n_original} engineered)")
    return df
