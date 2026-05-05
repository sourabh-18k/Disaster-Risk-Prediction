"""
Data Preprocessing Module for Flood Risk Prediction System
Handles data loading, cleaning, and labeling for Indian flood-prone regions

Flood risk labeling is based on:
- Central Water Commission (CWC) danger level criteria
- India Meteorological Department (IMD) heavy rainfall thresholds
- Historical flood event patterns from CWC Annual Flood Reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# City / Region Metadata (CWC danger levels, IMD thresholds)
# =============================================================================
CITY_METADATA = {
    'Guwahati': {
        'danger_level_m': 49.68,
        'warning_level_m': 48.68,
        'heavy_rain_threshold_mm': 115.6,   # IMD: Very heavy rainfall
        'extreme_rain_threshold_mm': 204.4,  # IMD: Extremely heavy rainfall
        'type': 'riverine',
        'river': 'Brahmaputra'
    },
    'Patna': {
        'danger_level_m': 50.00,
        'warning_level_m': 49.00,
        'heavy_rain_threshold_mm': 115.6,
        'extreme_rain_threshold_mm': 204.4,
        'type': 'riverine',
        'river': 'Ganga'
    },
    'Kochi': {
        'danger_level_m': 6.50,
        'warning_level_m': 5.50,
        'heavy_rain_threshold_mm': 115.6,
        'extreme_rain_threshold_mm': 204.4,
        'type': 'riverine_coastal',
        'river': 'Periyar'
    },
    'Mumbai': {
        'danger_level_m': 4.20,
        'warning_level_m': 3.50,
        'heavy_rain_threshold_mm': 115.6,
        'extreme_rain_threshold_mm': 204.4,
        'type': 'urban',
        'river': 'Mithi'
    },
    'Dehradun': {
        'danger_level_m': 7.00,
        'warning_level_m': 6.00,
        'heavy_rain_threshold_mm': 115.6,
        'extreme_rain_threshold_mm': 204.4,
        'type': 'flash_flood',
        'river': 'Song/Rispana'
    }
}


def load_city_data(file_path: str) -> pd.DataFrame:
    """
    Load data for a single city
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y')
    df = df.set_index('time')
    return df


def load_all_cities(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load flood data for all cities
    
    Args:
        data_dir: Directory containing the flood data files
        
    Returns:
        Dictionary of city names to DataFrames
    """
    cities = {
        'Guwahati': 'Guwahati_flood.csv',
        'Patna': 'Patna_flood.csv',
        'Kochi': 'Kochi_flood.csv',
        'Mumbai': 'Mumbai_flood.csv',
        'Dehradun': 'Dehradun_flood.csv'
    }
    
    city_data = {}
    for city, filename in cities.items():
        file_path = Path(data_dir) / filename
        print(f"Loading {city} data from {file_path}...")
        city_data[city] = load_city_data(str(file_path))
        print(f"  Loaded {len(city_data[city])} records ({city_data[city].index.min().date()} to {city_data[city].index.max().date()})")
    
    return city_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by handling missing values and unrealistic outliers
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Handle missing values: forward fill then backward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Clip physically unrealistic values
    for col in numeric_cols:
        if 'temperature' in col.lower():
            df[col] = df[col].clip(lower=-5, upper=50)
        elif 'humidity' in col.lower():
            df[col] = df[col].clip(lower=0, upper=100)
        elif 'precipitation' in col.lower() or 'rain' in col.lower():
            df[col] = df[col].clip(lower=0, upper=700)  # IMD record: ~944mm/day
        elif 'pressure' in col.lower():
            df[col] = df[col].clip(lower=950, upper=1050)
        elif 'discharge' in col.lower():
            df[col] = df[col].clip(lower=0)
        elif 'water_level' in col.lower():
            df[col] = df[col].clip(lower=0)
        elif 'soil_moisture' in col.lower():
            df[col] = df[col].clip(lower=0, upper=0.6)
        elif 'cloud' in col.lower():
            df[col] = df[col].clip(lower=0, upper=100)
        elif 'reservoir' in col.lower() or 'storage' in col.lower():
            df[col] = df[col].clip(lower=0, upper=100)
        else:
            # General IQR-based clipping
            Q1 = df[col].quantile(0.005)
            Q3 = df[col].quantile(0.995)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1 - 3 * IQR, upper=Q3 + 3 * IQR)
    
    return df


def define_flood_risk_labels(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Define flood risk labels based on CWC and IMD criteria
    
    Risk Levels:
        0 - No Risk:   Normal conditions
        1 - Low:       Above normal rainfall / rising water levels
        2 - Moderate:  Heavy rainfall + elevated water levels (approaching warning)
        3 - High:      Water level near/above warning level OR very heavy rain
        4 - Extreme:   Water level above danger level OR extremely heavy rain + saturated soil
    
    Args:
        df: Input DataFrame with hydrometeorological data
        city: City name for region-specific thresholds
        
    Returns:
        DataFrame with flood risk labels
    """
    df = df.copy()
    meta = CITY_METADATA[city]
    
    danger_level = meta['danger_level_m']
    warning_level = meta['warning_level_m']
    heavy_rain = meta['heavy_rain_threshold_mm']
    extreme_rain = meta['extreme_rain_threshold_mm']
    
    # --- Derived indicators ---
    
    # Cumulative rainfall (3-day and 7-day)
    df['rain_3d_sum'] = df['precipitation_sum (mm)'].rolling(3, min_periods=1).sum()
    df['rain_7d_sum'] = df['precipitation_sum (mm)'].rolling(7, min_periods=1).sum()
    
    # Discharge percentile (within this city's data)
    df['discharge_percentile'] = df['river_discharge (cumecs)'].rank(pct=True) * 100
    
    # Water level status
    df['above_warning'] = (df['water_level (m)'] >= warning_level).astype(int)
    df['above_danger'] = (df['water_level (m)'] >= danger_level).astype(int)
    
    # Soil saturation indicator
    df['soil_saturated'] = (df['soil_moisture_0_to_7cm_mean (m³/m³)'] >= 0.35).astype(int)
    
    # --- Risk Level Assignment ---
    df['risk_level'] = 0
    
    # Level 1 - Low Risk:
    # Moderate rainfall OR rising water levels OR elevated discharge
    low_rain_cond = (df['precipitation_sum (mm)'] >= 35.5) & (df['precipitation_sum (mm)'] < heavy_rain)
    rising_water = (df['water_level (m)'] >= warning_level * 0.85) & (df['water_level (m)'] < warning_level)
    elevated_discharge = (df['discharge_percentile'] >= 75) & (df['discharge_percentile'] < 90)
    df.loc[low_rain_cond | rising_water | elevated_discharge, 'risk_level'] = 1
    
    # Level 2 - Moderate Risk:
    # Heavy rainfall OR water near warning level OR high 3-day cumulative rain
    heavy_rain_cond = (df['precipitation_sum (mm)'] >= heavy_rain) & (df['precipitation_sum (mm)'] < extreme_rain)
    near_warning = (df['water_level (m)'] >= warning_level * 0.95) & (df['water_level (m)'] < warning_level)
    high_cum_rain = (df['rain_3d_sum'] >= heavy_rain * 1.5) & (df['rain_3d_sum'] < heavy_rain * 3)
    df.loc[heavy_rain_cond | near_warning | high_cum_rain, 'risk_level'] = 2
    
    # Level 3 - High Risk:
    # Water above warning or very heavy cumulative rain + saturated soil  
    above_warning_cond = (df['above_warning'] == 1) & (df['above_danger'] == 0)
    very_heavy_rain = (df['precipitation_sum (mm)'] >= extreme_rain) & (df['above_danger'] == 0)
    sustained_heavy = (df['rain_3d_sum'] >= heavy_rain * 3) & (df['soil_saturated'] == 1)
    high_discharge = (df['discharge_percentile'] >= 95) & (df['above_danger'] == 0)
    df.loc[above_warning_cond | very_heavy_rain | sustained_heavy | high_discharge, 'risk_level'] = 3
    
    # Level 4 - Extreme Risk:
    # Water above danger level OR extreme accumulated rainfall
    above_danger_cond = (df['above_danger'] == 1)
    extreme_event = (df['precipitation_sum (mm)'] >= extreme_rain) & (df['soil_saturated'] == 1) & (df['discharge_percentile'] >= 97)
    df.loc[above_danger_cond | extreme_event, 'risk_level'] = 4
    
    # Binary flood indicator (risk >= 3 is a flood event)
    df['is_flood'] = (df['risk_level'] >= 3).astype(int)
    
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived hydrometeorological features
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Temperature range
    df['temp_range'] = df['temperature_2m_max (°C)'] - df['temperature_2m_min (°C)']
    
    # Precipitation intensity (mm/hour when raining)
    df['precip_intensity'] = np.where(
        df['precipitation_hours (h)'] > 0,
        df['precipitation_sum (mm)'] / df['precipitation_hours (h)'],
        0
    )
    
    # Rain-to-ET ratio (excess water indicator)
    df['rain_et_ratio'] = np.where(
        df['et0_fao_evapotranspiration (mm)'] > 0,
        df['precipitation_sum (mm)'] / df['et0_fao_evapotranspiration (mm)'],
        0
    )
    
    # Has precipitation flag
    df['has_precipitation'] = (df['precipitation_sum (mm)'] > 0.1).astype(int)
    
    # Runoff potential indicator
    df['runoff_potential'] = (
        df['precipitation_sum (mm)'] * 
        df['soil_moisture_0_to_7cm_mean (m³/m³)'] *
        (100 - df['cloud_cover_mean (%)'].clip(0, 100)) / 100
    )
    
    # Pressure drop (depression indicator)
    df['pressure_change'] = df['pressure_msl_mean (hPa)'].diff()
    
    # Moisture surplus
    df['moisture_surplus'] = (
        df['relative_humidity_2m_mean (%)'] / 100 * 
        df['precipitation_sum (mm)'] -
        df['et0_fao_evapotranspiration (mm)']
    )
    
    return df


def preprocess_city_data(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for a single city
    
    Args:
        df: Raw DataFrame
        city: City name
        
    Returns:
        Preprocessed DataFrame with labels
    """
    print(f"\nPreprocessing {city}...")
    
    # Clean data
    df = clean_data(df)
    print(f"  Data cleaned: {len(df)} records")
    
    # Add derived features
    df = add_derived_features(df)
    print(f"  Derived features added")
    
    # Define flood risk labels
    df = define_flood_risk_labels(df, city)
    print(f"  Flood risk labels defined")
    
    # Add city identifier
    df['city'] = city
    
    # Summary
    flood_days = df['is_flood'].sum()
    flood_pct = (flood_days / len(df)) * 100
    print(f"  Flood risk days (High+Extreme): {flood_days} ({flood_pct:.2f}%)")
    print(f"  Risk level distribution:")
    risk_names = {0: 'No Risk', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Extreme'}
    for level, count in df['risk_level'].value_counts().sort_index().items():
        print(f"    {risk_names.get(level, level)}: {count} ({count/len(df)*100:.2f}%)")
    
    return df


def combine_cities_data(city_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine preprocessed data from all cities
    """
    combined = pd.concat(city_data_dict.values(), axis=0)
    combined = combined.sort_index()
    
    print(f"\nCombined dataset: {len(combined)} records from {len(city_data_dict)} cities")
    print(f"Date range: {combined.index.min().date()} to {combined.index.max().date()}")
    
    return combined


def load_and_preprocess_data(data_dir: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Complete data loading and preprocessing pipeline
    
    Args:
        data_dir: Directory containing the flood data files
        
    Returns:
        Tuple of (individual city data dict, combined DataFrame)
    """
    print("="*80)
    print("LOADING FLOOD DATA")
    print("="*80)
    city_data = load_all_cities(data_dir)
    
    print("\n" + "="*80)
    print("PREPROCESSING FLOOD DATA")
    print("="*80)
    city_data_processed = {}
    for city, df in city_data.items():
        city_data_processed[city] = preprocess_city_data(df, city)
    
    print("\n" + "="*80)
    print("COMBINING DATA")
    print("="*80)
    combined_data = combine_cities_data(city_data_processed)
    
    print("\nPreprocessing complete!")
    
    return city_data_processed, combined_data


if __name__ == "__main__":
    city_data, combined_data = load_and_preprocess_data("flood/")
    
    print("\nSaving processed data...")
    combined_data.to_csv("data/flood/processed/combined_flood_data.csv")
    print("Saved to: data/flood/processed/combined_flood_data.csv")
