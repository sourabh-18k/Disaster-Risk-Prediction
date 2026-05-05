"""
Data Preprocessing Module for Heatwave Prediction System
Handles data loading, cleaning, and preprocessing for all cities

Heatwave risk labeling is based on:
- India Meteorological Department (IMD) heatwave criteria
- National Disaster Management Authority (NDMA) guidelines
- US NWS Heat Index formulation adapted for Indian context
- City-specific climatological normals
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# City Metadata — IMD / NDMA heatwave thresholds
# =============================================================================
CITY_METADATA = {
    'Delhi': {
        'type': 'plains',
        'base_temp_threshold': 40,      # IMD: heatwave threshold for plains
        'severe_temp_threshold': 47,    # IMD: severe heatwave
        'departure_moderate': 4.5,      # departure from normal for heatwave
        'departure_severe': 6.5,        # departure for severe heatwave
        'normal_max_summer': 42.0,      # approximate climatological max (May-Jun)
    },
    'Jaipur': {
        'type': 'plains_arid',
        'base_temp_threshold': 40,
        'severe_temp_threshold': 47,
        'departure_moderate': 4.5,
        'departure_severe': 6.5,
        'normal_max_summer': 43.0,
    },
    'Hyderabad': {
        'type': 'plains',
        'base_temp_threshold': 40,
        'severe_temp_threshold': 47,
        'departure_moderate': 4.5,
        'departure_severe': 6.5,
        'normal_max_summer': 40.5,
    },
    'Bhubaneswar': {
        'type': 'coastal',
        'base_temp_threshold': 37,      # IMD: coastal station threshold
        'severe_temp_threshold': 44,
        'departure_moderate': 4.5,
        'departure_severe': 6.5,
        'normal_max_summer': 38.0,
    },
    'Chennai': {
        'type': 'coastal',
        'base_temp_threshold': 37,
        'severe_temp_threshold': 44,
        'departure_moderate': 4.5,
        'departure_severe': 6.5,
        'normal_max_summer': 37.5,
    },
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
    Load data for all cities
    
    Args:
        data_dir: Directory containing the heat data files
        
    Returns:
        Dictionary of city names to DataFrames
    """
    cities = {
        'Bhubaneswar': 'Bhubaneswar_h.csv',
        'Chennai': 'Chennai_h.csv',
        'Delhi': 'Delhi_h.csv',
        'Hyderabad': 'Hyderabad_h.csv',
        'Jaipur': 'Jaipur_h.csv'
    }
    
    city_data = {}
    for city, filename in cities.items():
        file_path = Path(data_dir) / filename
        print(f"Loading {city} data from {file_path}...")
        city_data[city] = load_city_data(str(file_path))
        print(f"  Loaded {len(city_data[city])} records")
    
    return city_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by handling missing values and outliers
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove duplicate et0_fao_evapotranspiration column if exists
    if 'et0_fao_evapotranspiration_sum (mm)' in df.columns:
        # Keep the _sum version and drop the duplicate
        if 'et0_fao_evapotranspiration (mm)' in df.columns:
            # Check if they have same values
            if df['et0_fao_evapotranspiration (mm)'].equals(df['et0_fao_evapotranspiration_sum (mm)']):
                df = df.drop(columns=['et0_fao_evapotranspiration_sum (mm)'])
    
    # Handle missing values
    # For numerical columns, use forward fill then backward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Handle outliers using IQR method (but be careful with extreme weather events)
    # We'll only cap extremely unrealistic values
    for col in numeric_cols:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        IQR = Q3 - Q1
        
        # For temperature, use physical limits
        if 'temperature' in col.lower():
            df[col] = df[col].clip(lower=-10, upper=55)
        elif 'humidity' in col.lower():
            df[col] = df[col].clip(lower=0, upper=100)
        elif 'pressure' in col.lower():
            df[col] = df[col].clip(lower=950, upper=1050)
        else:
            # For other variables, use IQR with wider bounds
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


def calculate_heat_index(temp_c: float, rh: float) -> float:
    """
    Calculate Heat Index using the formula from US National Weather Service
    Adapted for Celsius
    
    Args:
        temp_c: Temperature in Celsius
        rh: Relative Humidity (0-100)
        
    Returns:
        Heat Index in Celsius
    """
    # Convert to Fahrenheit for calculation
    temp_f = temp_c * 9/5 + 32
    
    # Simple formula for initial estimate
    hi_f = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh * 0.094))
    
    # If heat index is above 80°F, use Rothfusz regression
    if hi_f >= 80:
        hi_f = (-42.379 + 2.04901523 * temp_f + 10.14333127 * rh 
                - 0.22475541 * temp_f * rh - 0.00683783 * temp_f**2 
                - 0.05481717 * rh**2 + 0.00122874 * temp_f**2 * rh 
                + 0.00085282 * temp_f * rh**2 - 0.00000199 * temp_f**2 * rh**2)
    
    # Convert back to Celsius
    hi_c = (hi_f - 32) * 5/9
    
    return hi_c


def define_heatwave_labels(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Define heatwave labels based on IMD (India Meteorological Department) criteria
    using city-specific metadata.
    
    IMD Criteria:
    - Heatwave: Max temp ≥40°C (plains) / ≥37°C (coastal) and departure from normal ≥4.5-6.4°C
    - Severe Heatwave: Departure ≥6.5°C OR Max temp ≥47°C (plains) / ≥44°C (coastal)
    - Duration: At least 2 consecutive days
    
    Args:
        df: Input DataFrame with temperature data
        city: City name
        
    Returns:
        DataFrame with heatwave labels
    """
    df = df.copy()
    meta = CITY_METADATA.get(city, CITY_METADATA['Delhi'])
    
    base_temp_threshold = meta['base_temp_threshold']
    severe_temp_threshold = meta['severe_temp_threshold']
    departure_moderate = meta['departure_moderate']
    departure_severe = meta['departure_severe']
    
    # Calculate normal temperature (30-year climatology approximation using rolling mean)
    # Using entire dataset's day-of-year mean as proxy for climatology
    df['day_of_year'] = df.index.dayofyear
    
    # Calculate climatological normal for each day of year
    climatology = df.groupby('day_of_year')['temperature_2m_max (°C)'].transform('mean')
    
    # Calculate departure from normal
    df['temp_departure'] = df['temperature_2m_max (°C)'] - climatology
    
    # Heatwave conditions
    df['is_hot_day'] = (df['temperature_2m_max (°C)'] >= base_temp_threshold) & \
                       (df['temp_departure'] >= departure_moderate)
    
    df['is_severe_hot_day'] = ((df['temp_departure'] >= departure_severe) | \
                               (df['temperature_2m_max (°C)'] >= severe_temp_threshold))
    
    # Require 2 consecutive days for heatwave declaration
    df['heatwave_candidate'] = df['is_hot_day'].rolling(window=2, min_periods=2).sum() >= 2
    df['severe_heatwave_candidate'] = df['is_severe_hot_day'].rolling(window=2, min_periods=2).sum() >= 2
    
    # Create risk levels (0: No risk, 1: Low, 2: Moderate, 3: High, 4: Extreme)
    df['risk_level'] = 0
    
    # Low Risk: Approaching threshold (within 2°C of threshold and positive departure)
    df.loc[(df['temperature_2m_max (°C)'] >= base_temp_threshold - 2) & \
           (df['temp_departure'] >= 2) & \
           (df['temp_departure'] < departure_moderate), 'risk_level'] = 1
    
    # Moderate Risk: Single hot day or approaching heatwave
    df.loc[(df['is_hot_day']) & (~df['heatwave_candidate']), 'risk_level'] = 2
    
    # High Risk: Heatwave conditions (2+ consecutive days)
    df.loc[df['heatwave_candidate'], 'risk_level'] = 3
    
    # Extreme Risk: Severe heatwave
    df.loc[df['severe_heatwave_candidate'], 'risk_level'] = 4
    
    # Add binary heatwave indicator
    df['is_heatwave'] = (df['risk_level'] >= 3).astype(int)
    
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that don't require temporal information
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Heat Index
    df['heat_index'] = df.apply(
        lambda row: calculate_heat_index(
            row['temperature_2m_mean (°C)'],
            row['relative_humidity_2m_mean (%)']
        ), axis=1
    )
    
    # Temperature range
    df['temp_range'] = df['temperature_2m_max (°C)'] - df['temperature_2m_min (°C)']
    
    # Discomfort measures
    df['apparent_temp_diff'] = df['apparent_temperature_mean (°C)'] - df['temperature_2m_mean (°C)']
    
    # Soil heat content
    df['soil_temp_deviation'] = df['soil_temperature_0_to_7cm_mean (°C)'] - df['temperature_2m_mean (°C)']
    
    # Moisture deficit
    df['moisture_deficit'] = df['vapour_pressure_deficit_max (kPa)']
    
    # Wind chill / cooling effect
    df['wind_cooling_effect'] = df['wind_speed_10m_mean (km/h)'] * (df['temperature_2m_max (°C)'] - df['dew_point_2m_mean (°C)'])
    
    # Precipitation indicator
    df['has_precipitation'] = (df['precipitation_sum (mm)'] > 0).astype(int)
    
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
    
    # Define heatwave labels
    df = define_heatwave_labels(df, city)
    print(f"  Heatwave labels defined")
    
    # Add city identifier
    df['city'] = city
    
    # Summary statistics
    heatwave_days = df['is_heatwave'].sum()
    heatwave_pct = (heatwave_days / len(df)) * 100
    print(f"  Heatwave days: {heatwave_days} ({heatwave_pct:.2f}%)")
    print(f"  Risk level distribution:")
    for level, count in df['risk_level'].value_counts().sort_index().items():
        risk_names = {0: 'No Risk', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Extreme'}
        print(f"    {risk_names.get(level, level)}: {count} ({count/len(df)*100:.2f}%)")
    
    return df


def combine_cities_data(city_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine preprocessed data from all cities
    
    Args:
        city_data_dict: Dictionary of city names to preprocessed DataFrames
        
    Returns:
        Combined DataFrame
    """
    combined = pd.concat(city_data_dict.values(), axis=0)
    combined = combined.sort_index()
    
    print(f"\nCombined dataset: {len(combined)} records from {len(city_data_dict)} cities")
    print(f"Date range: {combined.index.min()} to {combined.index.max()}")
    
    return combined


def load_and_preprocess_data(data_dir: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Complete data loading and preprocessing pipeline
    
    Args:
        data_dir: Directory containing the heat data files
        
    Returns:
        Tuple of (individual city data dict, combined DataFrame)
    """
    # Load data
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    city_data = load_all_cities(data_dir)
    
    # Preprocess each city
    print("\n" + "="*80)
    print("PREPROCESSING DATA")
    print("="*80)
    city_data_processed = {}
    for city, df in city_data.items():
        city_data_processed[city] = preprocess_city_data(df, city)
    
    # Combine data
    print("\n" + "="*80)
    print("COMBINING DATA")
    print("="*80)
    combined_data = combine_cities_data(city_data_processed)
    
    print("\nPreprocessing complete!")
    
    return city_data_processed, combined_data


if __name__ == "__main__":
    # Test the preprocessing pipeline
    city_data, combined_data = load_and_preprocess_data("heat/")
    
    # Save processed data
    print("\nSaving processed data...")
    combined_data.to_csv("data/heatwave/processed/combined_data.csv")
    print("Saved to: data/heatwave/processed/combined_data.csv")
