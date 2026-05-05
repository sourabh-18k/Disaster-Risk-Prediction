"""
Feature Engineering Module for Heatwave Prediction System
Creates temporal features, lag features, and rolling statistics
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the datetime index
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        DataFrame with temporal features
    """
    df = df.copy()
    
    # Basic temporal features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['day_of_week'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    
    # Cyclical encoding for periodic features
    # Month (1-12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Day of year (1-365)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Day of week (0-6)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Season (Indian context)
    def get_season(month):
        if month in [3, 4, 5]:
            return 1  # Summer (Pre-monsoon)
        elif month in [6, 7, 8, 9]:
            return 2  # Monsoon
        elif month in [10, 11]:
            return 3  # Post-monsoon
        else:  # 12, 1, 2
            return 4  # Winter
    
    df['season'] = df['month'].apply(get_season)
    
    # Binary indicators
    df['is_summer'] = (df['season'] == 1).astype(int)
    df['is_monsoon'] = (df['season'] == 2).astype(int)
    df['is_winter'] = (df['season'] == 4).astype(int)
    
    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for specified columns
    
    Args:
        df: Input DataFrame
        columns: List of column names to create lags for
        lags: List of lag periods (in days)
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    # Sort by city and date to ensure proper lag calculation
    if 'city' in df.columns:
        df = df.sort_values(['city', df.index.name or 'time'])
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    # Create lag feature per city
                    df[f'{col}_lag{lag}'] = df.groupby('city')[col].shift(lag)
    else:
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Create rolling window statistics for specified columns
    
    Args:
        df: Input DataFrame
        columns: List of column names to create rolling features for
        windows: List of window sizes (in days)
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    # Sort by city and date
    if 'city' in df.columns:
        df = df.sort_values(['city', df.index.name or 'time'])
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # Rolling mean
                    df[f'{col}_rolling_mean_{window}d'] = df.groupby('city')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    # Rolling std
                    df[f'{col}_rolling_std_{window}d'] = df.groupby('city')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                    # Rolling max
                    df[f'{col}_rolling_max_{window}d'] = df.groupby('city')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
                    # Rolling min
                    df[f'{col}_rolling_min_{window}d'] = df.groupby('city')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min()
                    )
    else:
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}d'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}d'] = df[col].rolling(window=window, min_periods=1).std()
                    df[f'{col}_rolling_max_{window}d'] = df[col].rolling(window=window, min_periods=1).max()
                    df[f'{col}_rolling_min_{window}d'] = df[col].rolling(window=window, min_periods=1).min()
    
    return df


def create_heatwave_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create heatwave-specific features and indicators
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with heatwave-specific features
    """
    df = df.copy()
    
    # Temperature trend (change from previous day)
    if 'city' in df.columns:
        df['temp_change'] = df.groupby('city')['temperature_2m_max (°C)'].diff()
        df['temp_change_3d'] = df.groupby('city')['temperature_2m_max (°C)'].diff(3)
        df['temp_change_7d'] = df.groupby('city')['temperature_2m_max (°C)'].diff(7)
    else:
        df['temp_change'] = df['temperature_2m_max (°C)'].diff()
        df['temp_change_3d'] = df['temperature_2m_max (°C)'].diff(3)
        df['temp_change_7d'] = df['temperature_2m_max (°C)'].diff(7)
    
    # Consecutive hot days counter
    def count_consecutive_hot_days(series, threshold=40):
        """Count consecutive days above threshold"""
        # Create binary series for hot days
        is_hot = (series >= threshold).astype(int)
        # Count consecutive occurrences
        consecutive = is_hot.groupby((is_hot != is_hot.shift()).cumsum()).cumsum()
        return consecutive * is_hot  # Zero out non-hot days
    
    if 'city' in df.columns:
        df['consecutive_hot_days'] = df.groupby('city')['temperature_2m_max (°C)'].transform(
            count_consecutive_hot_days
        )
    else:
        df['consecutive_hot_days'] = count_consecutive_hot_days(df['temperature_2m_max (°C)'])
    
    # Cumulative heat exposure (sum of temperature departure over past week)
    if 'temp_departure' in df.columns:
        if 'city' in df.columns:
            df['cumulative_heat_7d'] = df.groupby('city')['temp_departure'].transform(
                lambda x: x.rolling(window=7, min_periods=1).sum()
            )
            df['cumulative_heat_14d'] = df.groupby('city')['temp_departure'].transform(
                lambda x: x.rolling(window=14, min_periods=1).sum()
            )
        else:
            df['cumulative_heat_7d'] = df['temp_departure'].rolling(window=7, min_periods=1).sum()
            df['cumulative_heat_14d'] = df['temp_departure'].rolling(window=14, min_periods=1).sum()
    
    # Heat stress indicator (combination of temperature and humidity)
    if 'heat_index' in df.columns:
        df['heat_stress'] = (df['heat_index'] - df['temperature_2m_mean (°C)'])
        df['is_high_heat_stress'] = (df['heat_stress'] > 5).astype(int)
    
    # Moisture availability (inverse of VPD)
    if 'vapour_pressure_deficit_max (kPa)' in df.columns:
        df['moisture_stress'] = df['vapour_pressure_deficit_max (kPa)']
        df['is_moisture_stress'] = (df['moisture_stress'] > 3).astype(int)
    
    # Nighttime heat (minimum temperature relative to climatology)
    if 'temperature_2m_min (°C)' in df.columns:
        if 'city' in df.columns:
            df['nighttime_temp_percentile'] = df.groupby(['city', 'day_of_year'])['temperature_2m_min (°C)'].rank(pct=True)
        else:
            df['nighttime_temp_percentile'] = df.groupby('day_of_year')['temperature_2m_min (°C)'].rank(pct=True)
    
    # Dry spell indicator (consecutive days without precipitation)
    if 'precipitation_sum (mm)' in df.columns:
        def count_dry_days(series):
            is_dry = (series <= 0.1).astype(int)
            consecutive = is_dry.groupby((is_dry != is_dry.shift()).cumsum()).cumsum()
            return consecutive * is_dry
        
        if 'city' in df.columns:
            df['consecutive_dry_days'] = df.groupby('city')['precipitation_sum (mm)'].transform(count_dry_days)
        else:
            df['consecutive_dry_days'] = count_dry_days(df['precipitation_sum (mm)'])
    
    # Soil moisture depletion rate
    if 'soil_moisture_0_to_7cm_mean (m³/m³)' in df.columns:
        if 'city' in df.columns:
            df['soil_moisture_change'] = df.groupby('city')['soil_moisture_0_to_7cm_mean (m³/m³)'].diff()
        else:
            df['soil_moisture_change'] = df['soil_moisture_0_to_7cm_mean (m³/m³)'].diff()
    
    # Atmospheric dryness indicator
    if all(col in df.columns for col in ['relative_humidity_2m_mean (%)', 'temperature_2m_max (°C)']):
        df['atmospheric_dryness'] = (100 - df['relative_humidity_2m_mean (%)']) * (df['temperature_2m_max (°C)'] / 40)
    
    # Wind intensity relative to normal
    if 'wind_speed_10m_mean (km/h)' in df.columns:
        if 'city' in df.columns:
            wind_normal = df.groupby(['city', 'month'])['wind_speed_10m_mean (km/h)'].transform('mean')
        else:
            wind_normal = df.groupby('month')['wind_speed_10m_mean (km/h)'].transform('mean')
        df['wind_anomaly'] = df['wind_speed_10m_mean (km/h)'] - wind_normal
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between important variables
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    # Temperature-Humidity interaction
    if all(col in df.columns for col in ['temperature_2m_max (°C)', 'relative_humidity_2m_mean (%)']):
        df['temp_humidity_interaction'] = df['temperature_2m_max (°C)'] * df['relative_humidity_2m_mean (%)']
    
    # Temperature-Wind interaction (cooling effect)
    if all(col in df.columns for col in ['temperature_2m_max (°C)', 'wind_speed_10m_mean (km/h)']):
        df['temp_wind_interaction'] = df['temperature_2m_max (°C)'] / (df['wind_speed_10m_mean (km/h)'] + 1)
    
    # Radiation-Cloud interaction
    if all(col in df.columns for col in ['shortwave_radiation_sum (MJ/m²)', 'cloud_cover_mean (%)']):
        df['radiation_cloud_interaction'] = df['shortwave_radiation_sum (MJ/m²)'] * (100 - df['cloud_cover_mean (%)']) / 100
    
    # Temperature-Soil moisture interaction
    if all(col in df.columns for col in ['temperature_2m_max (°C)', 'soil_moisture_0_to_7cm_mean (m³/m³)']):
        df['temp_soil_interaction'] = df['temperature_2m_max (°C)'] / (df['soil_moisture_0_to_7cm_mean (m³/m³)'] + 0.01)
    
    return df


def create_all_features(df: pd.DataFrame, 
                       lag_periods: List[int] = [1, 2, 3, 7],
                       rolling_windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
    """
    Create all engineered features
    
    Args:
        df: Input preprocessed DataFrame
        lag_periods: List of lag periods for lag features
        rolling_windows: List of window sizes for rolling features
        
    Returns:
        DataFrame with all engineered features
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    original_features = len(df.columns)
    print(f"Original features: {original_features}")
    
    # Temporal features
    print("\nCreating temporal features...")
    df = create_temporal_features(df)
    print(f"  Added {len(df.columns) - original_features} temporal features")
    
    # Key columns for lag and rolling features
    key_columns = [
        'temperature_2m_max (°C)',
        'temperature_2m_mean (°C)',
        'temperature_2m_min (°C)',
        'apparent_temperature_mean (°C)',
        'relative_humidity_2m_mean (%)',
        'heat_index',
        'temp_departure'
    ]
    
    # Lag features
    current_features = len(df.columns)
    print(f"\nCreating lag features for {len(key_columns)} columns...")
    df = create_lag_features(df, key_columns, lag_periods)
    print(f"  Added {len(df.columns) - current_features} lag features")
    
    # Rolling features
    current_features = len(df.columns)
    print(f"\nCreating rolling statistics for key columns...")
    rolling_columns = [
        'temperature_2m_max (°C)',
        'temperature_2m_mean (°C)',
        'relative_humidity_2m_mean (%)',
        'vapour_pressure_deficit_max (kPa)'
    ]
    df = create_rolling_features(df, rolling_columns, rolling_windows)
    print(f"  Added {len(df.columns) - current_features} rolling features")
    
    # Heatwave-specific features
    current_features = len(df.columns)
    print(f"\nCreating heatwave-specific features...")
    df = create_heatwave_specific_features(df)
    print(f"  Added {len(df.columns) - current_features} heatwave features")
    
    # Interaction features
    current_features = len(df.columns)
    print(f"\nCreating interaction features...")
    df = create_interaction_features(df)
    print(f"  Added {len(df.columns) - current_features} interaction features")
    
    total_features = len(df.columns)
    print(f"\nTotal features: {total_features}")
    print(f"New features created: {total_features - original_features}")
    
    # Handle any remaining NaN values from lag/rolling operations
    print(f"\nHandling NaN values from feature engineering...")
    nan_before = df.isnull().sum().sum()
    
    # Fill NaN values
    # For lag features, we'll use backward fill (since they're at the start)
    # For rolling features, they should already have min_periods=1
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='bfill').fillna(method='ffill')
    
    # If still any NaN, fill with 0
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    nan_after = df.isnull().sum().sum()
    print(f"  NaN values removed: {nan_before} -> {nan_after}")
    
    print("\nFeature engineering complete!")
    
    return df


if __name__ == "__main__":
    # This would be run after data_preprocessing
    print("Feature engineering module loaded successfully")
    print("Use create_all_features() to engineer features from preprocessed data")
