"""
Feature Engineering Module for Flood Risk Prediction System
Creates temporal, lag, rolling, and flood-specific features

Features are designed based on hydrological research:
- Antecedent moisture conditions (critical for runoff)
- Cumulative rainfall patterns (threshold exceedance)
- River discharge dynamics (rate of change, peak detection)
- Seasonal monsoon indicators (Indian monsoon context)
"""

import pandas as pd
import numpy as np
from typing import List
import warnings
warnings.filterwarnings('ignore')


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the datetime index
    """
    df = df.copy()
    
    # Basic temporal
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['day_of_week'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Indian season classification
    def get_season(month):
        if month in [3, 4, 5]:
            return 1  # Pre-monsoon (Summer)
        elif month in [6, 7, 8, 9]:
            return 2  # Monsoon (SW Monsoon - peak flood season)
        elif month in [10, 11]:
            return 3  # Post-monsoon (NE Monsoon / retreating)
        else:
            return 4  # Winter
    
    df['season'] = df['month'].apply(get_season)
    
    # Binary indicators
    df['is_monsoon'] = (df['season'] == 2).astype(int)
    df['is_pre_monsoon'] = (df['season'] == 1).astype(int)
    df['is_post_monsoon'] = (df['season'] == 3).astype(int)
    df['is_winter'] = (df['season'] == 4).astype(int)
    
    # Peak monsoon (July-August)
    df['is_peak_monsoon'] = df['month'].isin([7, 8]).astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for specified columns
    """
    df = df.copy()
    
    if 'city' in df.columns:
        df = df.sort_values(['city', df.index.name or 'time'])
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag{lag}'] = df.groupby('city')[col].shift(lag)
    else:
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Create rolling window statistics
    """
    df = df.copy()
    
    if 'city' in df.columns:
        df = df.sort_values(['city', df.index.name or 'time'])
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_roll_mean_{window}d'] = df.groupby('city')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    df[f'{col}_roll_std_{window}d'] = df.groupby('city')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                    df[f'{col}_roll_max_{window}d'] = df.groupby('city')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                    df[f'{col}_roll_sum_{window}d'] = df.groupby('city')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).sum()
                    )
    else:
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_roll_mean_{window}d'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_roll_std_{window}d'] = df[col].rolling(window=window, min_periods=1).std()
                    df[f'{col}_roll_max_{window}d'] = df[col].rolling(window=window, min_periods=1).max()
                    df[f'{col}_roll_sum_{window}d'] = df[col].rolling(window=window, min_periods=1).sum()
    
    return df


def create_flood_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create flood-specific hydrometeorological features
    
    Based on established flood forecasting indicators:
    - Antecedent Precipitation Index (API)
    - Standardized Precipitation Index (SPI) approximation
    - Rate of rise of water level
    - Soil saturation deficit
    - Compound flood indicators
    """
    df = df.copy()
    
    group_col = 'city' if 'city' in df.columns else None
    
    def _groupby_or_direct(series, func, **kwargs):
        if group_col and group_col in df.columns:
            return df.groupby(group_col)[series].transform(func, **kwargs)
        return func(df[series], **kwargs)
    
    # --- Rainfall Dynamics ---
    
    # Rate of change of rainfall
    if group_col:
        df['precip_change_1d'] = df.groupby('city')['precipitation_sum (mm)'].diff()
        df['precip_change_3d'] = df.groupby('city')['precipitation_sum (mm)'].diff(3)
    else:
        df['precip_change_1d'] = df['precipitation_sum (mm)'].diff()
        df['precip_change_3d'] = df['precipitation_sum (mm)'].diff(3)
    
    # Consecutive rain days
    def count_consecutive_rain(series, threshold=2.5):
        is_rain = (series >= threshold).astype(int)
        consecutive = is_rain.groupby((is_rain != is_rain.shift()).cumsum()).cumsum()
        return consecutive * is_rain
    
    if group_col:
        df['consecutive_rain_days'] = df.groupby('city')['precipitation_sum (mm)'].transform(
            count_consecutive_rain
        )
    else:
        df['consecutive_rain_days'] = count_consecutive_rain(df['precipitation_sum (mm)'])
    
    # Dry spell before rain (soil crust formation affects runoff)
    def count_dry_days(series, threshold=2.5):
        is_dry = (series < threshold).astype(int)
        consecutive = is_dry.groupby((is_dry != is_dry.shift()).cumsum()).cumsum()
        return consecutive * is_dry
    
    if group_col:
        df['preceding_dry_days'] = df.groupby('city')['precipitation_sum (mm)'].transform(count_dry_days)
    else:
        df['preceding_dry_days'] = count_dry_days(df['precipitation_sum (mm)'])
    
    # Rainfall departure from monthly normal
    if group_col:
        monthly_normal = df.groupby(['city', 'month'])['precipitation_sum (mm)'].transform('mean')
    else:
        monthly_normal = df.groupby('month')['precipitation_sum (mm)'].transform('mean')
    df['rain_departure'] = df['precipitation_sum (mm)'] - monthly_normal
    df['rain_departure_pct'] = np.where(
        monthly_normal > 0,
        (df['precipitation_sum (mm)'] - monthly_normal) / monthly_normal * 100,
        0
    )
    
    # --- Water Level Dynamics ---
    
    if 'water_level (m)' in df.columns:
        if group_col:
            df['water_level_change_1d'] = df.groupby('city')['water_level (m)'].diff()
            df['water_level_change_3d'] = df.groupby('city')['water_level (m)'].diff(3)
            df['water_level_change_7d'] = df.groupby('city')['water_level (m)'].diff(7)
        else:
            df['water_level_change_1d'] = df['water_level (m)'].diff()
            df['water_level_change_3d'] = df['water_level (m)'].diff(3)
            df['water_level_change_7d'] = df['water_level (m)'].diff(7)
        
        # Rate of rise (m/day) - critical flood indicator
        df['rate_of_rise'] = df['water_level_change_1d']
        df['is_rapid_rise'] = (df['rate_of_rise'] > 0.3).astype(int)
    
    # --- Discharge Dynamics ---
    
    if 'river_discharge (cumecs)' in df.columns:
        if group_col:
            df['discharge_change_1d'] = df.groupby('city')['river_discharge (cumecs)'].diff()
            df['discharge_change_3d'] = df.groupby('city')['river_discharge (cumecs)'].diff(3)
        else:
            df['discharge_change_1d'] = df['river_discharge (cumecs)'].diff()
            df['discharge_change_3d'] = df['river_discharge (cumecs)'].diff(3)
        
        # Discharge-to-normal ratio
        if group_col:
            discharge_normal = df.groupby(['city', 'month'])['river_discharge (cumecs)'].transform('mean')
        else:
            discharge_normal = df.groupby('month')['river_discharge (cumecs)'].transform('mean')
        
        df['discharge_ratio'] = np.where(
            discharge_normal > 0,
            df['river_discharge (cumecs)'] / discharge_normal,
            1
        )
    
    # --- Soil & Moisture Indicators ---
    
    if 'soil_moisture_0_to_7cm_mean (m³/m³)' in df.columns:
        # Soil moisture change rate
        if group_col:
            df['soil_moisture_change'] = df.groupby('city')['soil_moisture_0_to_7cm_mean (m³/m³)'].diff()
        else:
            df['soil_moisture_change'] = df['soil_moisture_0_to_7cm_mean (m³/m³)'].diff()
        
        # Soil saturation deficit (lower = more saturated = more runoff)
        df['saturation_deficit'] = 0.50 - df['soil_moisture_0_to_7cm_mean (m³/m³)']
        df['saturation_deficit'] = df['saturation_deficit'].clip(lower=0)
        
        # Infiltration capacity proxy (saturated soil = low infiltration = more runoff)
        df['infiltration_proxy'] = (
            df['saturation_deficit'] * 100 + 
            df['preceding_dry_days'] * 2
        )
    
    # --- Reservoir Dynamics ---
    
    if 'reservoir_storage_pct (%)' in df.columns:
        if group_col:
            df['reservoir_change_1d'] = df.groupby('city')['reservoir_storage_pct (%)'].diff()
            df['reservoir_change_7d'] = df.groupby('city')['reservoir_storage_pct (%)'].diff(7)
        else:
            df['reservoir_change_1d'] = df['reservoir_storage_pct (%)'].diff()
            df['reservoir_change_7d'] = df['reservoir_storage_pct (%)'].diff(7)
        
        # High reservoir = risk of release flooding
        df['reservoir_near_full'] = (df['reservoir_storage_pct (%)'] >= 85).astype(int)
    
    # --- Compound Flood Indicators ---
    
    # Flood Potential Index (combination of key indicators)
    fpi_components = []
    if 'antecedent_precip_index (mm)' in df.columns:
        fpi_components.append(df['antecedent_precip_index (mm)'] / df['antecedent_precip_index (mm)'].max())
    if 'soil_moisture_0_to_7cm_mean (m³/m³)' in df.columns:
        fpi_components.append(df['soil_moisture_0_to_7cm_mean (m³/m³)'] / 0.5)
    if 'precipitation_sum (mm)' in df.columns:
        fpi_components.append(df['precipitation_sum (mm)'] / df['precipitation_sum (mm)'].quantile(0.99))
    
    if fpi_components:
        df['flood_potential_index'] = np.mean(fpi_components, axis=0)
    
    # Rainfall-Discharge coupling strength
    if all(c in df.columns for c in ['precipitation_sum (mm)', 'river_discharge (cumecs)']):
        rain_norm = df['precipitation_sum (mm)'] / (df['precipitation_sum (mm)'].max() + 1)
        disc_norm = df['river_discharge (cumecs)'] / (df['river_discharge (cumecs)'].max() + 1)
        df['rain_discharge_coupling'] = rain_norm * disc_norm
    
    # Atmospheric instability proxy
    if all(c in df.columns for c in ['pressure_msl_mean (hPa)', 'relative_humidity_2m_mean (%)']):
        df['atm_instability'] = (
            (1020 - df['pressure_msl_mean (hPa)']) *
            df['relative_humidity_2m_mean (%)'] / 100
        )
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between important hydrometeorological variables
    """
    df = df.copy()
    
    # Rainfall × Soil moisture (runoff indicator)
    if all(c in df.columns for c in ['precipitation_sum (mm)', 'soil_moisture_0_to_7cm_mean (m³/m³)']):
        df['rain_soil_interaction'] = df['precipitation_sum (mm)'] * df['soil_moisture_0_to_7cm_mean (m³/m³)']
    
    # Rainfall × Humidity
    if all(c in df.columns for c in ['precipitation_sum (mm)', 'relative_humidity_2m_mean (%)']):
        df['rain_humidity_interaction'] = df['precipitation_sum (mm)'] * df['relative_humidity_2m_mean (%)'] / 100
    
    # Temperature × Precipitation (warm rain events)
    if all(c in df.columns for c in ['temperature_2m_mean (°C)', 'precipitation_sum (mm)']):
        df['temp_rain_interaction'] = df['temperature_2m_mean (°C)'] * df['precipitation_sum (mm)'] / 100
    
    # Wind × Rainfall (storm intensity)
    if all(c in df.columns for c in ['wind_speed_10m_mean (km/h)', 'precipitation_sum (mm)']):
        df['wind_rain_interaction'] = df['wind_speed_10m_mean (km/h)'] * df['precipitation_sum (mm)'] / 100
    
    # Pressure × Rainfall (depression-driven rain)
    if all(c in df.columns for c in ['pressure_msl_mean (hPa)', 'precipitation_sum (mm)']):
        df['pressure_rain_interaction'] = (1020 - df['pressure_msl_mean (hPa)']) * df['precipitation_sum (mm)']
    
    # Discharge × Soil moisture (basin response)
    if all(c in df.columns for c in ['river_discharge (cumecs)', 'soil_moisture_0_to_7cm_mean (m³/m³)']):
        df['discharge_soil_interaction'] = df['river_discharge (cumecs)'] * df['soil_moisture_0_to_7cm_mean (m³/m³)']
    
    # API × Current rainfall (antecedent + current event)
    if all(c in df.columns for c in ['antecedent_precip_index (mm)', 'precipitation_sum (mm)']):
        df['api_rain_interaction'] = df['antecedent_precip_index (mm)'] * df['precipitation_sum (mm)'] / 1000
    
    return df


def create_all_features(df: pd.DataFrame,
                       lag_periods: List[int] = [1, 2, 3, 5, 7],
                       rolling_windows: List[int] = [3, 5, 7, 14, 30]) -> pd.DataFrame:
    """
    Create all engineered features for flood prediction
    
    Args:
        df: Input preprocessed DataFrame
        lag_periods: List of lag periods for lag features
        rolling_windows: List of window sizes for rolling features
        
    Returns:
        DataFrame with all engineered features
    """
    print("\n" + "="*80)
    print("FLOOD FEATURE ENGINEERING")
    print("="*80)
    
    original_features = len(df.columns)
    print(f"Original features: {original_features}")
    
    # Temporal features
    print("\nCreating temporal features...")
    df = create_temporal_features(df)
    print(f"  Added {len(df.columns) - original_features} temporal features")
    
    # Key columns for lag and rolling features (flood-relevant)
    lag_columns = [
        'precipitation_sum (mm)',
        'river_discharge (cumecs)',
        'water_level (m)',
        'soil_moisture_0_to_7cm_mean (m³/m³)',
        'relative_humidity_2m_mean (%)',
        'antecedent_precip_index (mm)',
        'reservoir_storage_pct (%)'
    ]
    
    # Lag features
    current_features = len(df.columns)
    print(f"\nCreating lag features ({len(lag_columns)} columns × {len(lag_periods)} lags)...")
    df = create_lag_features(df, lag_columns, lag_periods)
    print(f"  Added {len(df.columns) - current_features} lag features")
    
    # Rolling features (fewer columns, more windows for flood prediction)
    rolling_columns = [
        'precipitation_sum (mm)',
        'river_discharge (cumecs)',
        'water_level (m)',
        'relative_humidity_2m_mean (%)'
    ]
    
    current_features = len(df.columns)
    print(f"\nCreating rolling statistics ({len(rolling_columns)} columns × {len(rolling_windows)} windows)...")
    df = create_rolling_features(df, rolling_columns, rolling_windows)
    print(f"  Added {len(df.columns) - current_features} rolling features")
    
    # Flood-specific features
    current_features = len(df.columns)
    print(f"\nCreating flood-specific features...")
    df = create_flood_specific_features(df)
    print(f"  Added {len(df.columns) - current_features} flood-specific features")
    
    # Interaction features
    current_features = len(df.columns)
    print(f"\nCreating interaction features...")
    df = create_interaction_features(df)
    print(f"  Added {len(df.columns) - current_features} interaction features")
    
    total_features = len(df.columns)
    print(f"\nTotal features: {total_features}")
    print(f"New features created: {total_features - original_features}")
    
    # Handle NaN values
    print(f"\nHandling NaN values from feature engineering...")
    nan_before = df.isnull().sum().sum()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='bfill').fillna(method='ffill')
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    nan_after = df.isnull().sum().sum()
    print(f"  NaN values: {nan_before} → {nan_after}")
    
    print("\nFlood feature engineering complete!")
    
    return df


if __name__ == "__main__":
    print("Flood feature engineering module loaded successfully")
    print("Use create_all_features() to engineer features from preprocessed flood data")
