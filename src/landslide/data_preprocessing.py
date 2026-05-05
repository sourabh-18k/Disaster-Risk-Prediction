"""
Data Preprocessing Module for Landslide Risk Prediction System
Handles data loading, cleaning, and risk labeling for Indian landslide-prone regions.

Landslide risk labeling is based on:
- Geological Survey of India (GSI) Landslide Susceptibility Zonation guidelines
- NDMA National Landslide Risk Management Strategy
- IMD rainfall thresholds for landslide triggering
- Published intensity-duration (I-D) thresholds for Indian Himalayas
  (e.g., Kanungo & Sharma 2014; Dikshit et al. 2020)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Region Metadata — GSI / NDMA parameters
# =============================================================================
REGION_METADATA = {
    'Shimla': {
        'geology': 'Lesser Himalaya — Phyllite/Slate/Dolomite',
        'seismic_zone': 'IV',
        # Rainfall intensity-duration threshold for landslide triggering (mm/day)
        'rain_trigger_moderate': 60,   # > 60 mm/day
        'rain_trigger_high': 100,      # > 100 mm/day (IMD "very heavy")
        'rain_trigger_extreme': 200,   # > 200 mm/day (IMD "extremely heavy")
        'slope_high_risk': 35,
        'api_trigger': 80,  # antecedent precip index threshold
    },
    'Munnar': {
        'geology': 'Western Ghats — Charnockite/Gneiss',
        'seismic_zone': 'III',
        'rain_trigger_moderate': 65,
        'rain_trigger_high': 120,
        'rain_trigger_extreme': 250,
        'slope_high_risk': 30,
        'api_trigger': 100,
    },
    'Darjeeling': {
        'geology': 'Eastern Himalaya — Darjeeling Gneiss/Daling Phyllite',
        'seismic_zone': 'IV',
        'rain_trigger_moderate': 55,
        'rain_trigger_high': 90,
        'rain_trigger_extreme': 180,
        'slope_high_risk': 35,
        'api_trigger': 75,
    },
    'Uttarkashi': {
        'geology': 'Greater Himalaya — Central Crystallines/MCT zone',
        'seismic_zone': 'V',
        'rain_trigger_moderate': 50,
        'rain_trigger_high': 80,
        'rain_trigger_extreme': 170,
        'slope_high_risk': 35,
        'api_trigger': 65,
    },
    'Kohima': {
        'geology': 'Purvanchal Hills — Disang Shale/Barail Sandstone',
        'seismic_zone': 'V',
        'rain_trigger_moderate': 60,
        'rain_trigger_high': 100,
        'rain_trigger_extreme': 200,
        'slope_high_risk': 30,
        'api_trigger': 85,
    },
}


def load_region_data(file_path: str) -> pd.DataFrame:
    """Load data for a single region."""
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y')
    df = df.set_index('time')
    return df


def load_all_regions(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load landslide data for all regions."""
    regions = {
        'Shimla': 'Shimla_landslide.csv',
        'Munnar': 'Munnar_landslide.csv',
        'Darjeeling': 'Darjeeling_landslide.csv',
        'Uttarkashi': 'Uttarkashi_landslide.csv',
        'Kohima': 'Kohima_landslide.csv'
    }
    region_data = {}
    data_path = Path(data_dir)
    for region_name, filename in regions.items():
        fp = data_path / filename
        if fp.exists():
            region_data[region_name] = load_region_data(str(fp))
            print(f"  Loaded {region_name}: {len(region_data[region_name])} records")
        else:
            print(f"  WARNING: {fp} not found — skipping {region_name}")
    return region_data


def assign_landslide_risk(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    """
    Assign landslide risk levels (0-4) based on multi-criteria scoring.

    Uses a weighted factor-of-safety inspired approach combining:
      1) Rainfall intensity         (daily + antecedent)
      2) Slope steepness
      3) Soil moisture saturation
      4) Geological susceptibility
      5) Seismic loading (PGA)
      6) Land use (deforested/barren = higher risk)

    Risk levels:
      0 = No Risk    — stable conditions
      1 = Low Risk   — marginally stable
      2 = Moderate   — susceptible under prolonged rain
      3 = High Risk  — likely failure (trigger conditions met)
      4 = Extreme    — imminent / multiple triggers active

    Thresholds inspired by GSI LSZ (Landslide Susceptibility Zonation) and
    published I-D curves for the Himalayas.
    """
    meta = REGION_METADATA.get(region_name, REGION_METADATA['Shimla'])
    df = df.copy()

    # --- Compute composite landslide hazard score (0-100 scale) ---
    score = np.zeros(len(df))

    # 1) Rainfall factor (max 30 points)
    rain = df['precipitation_sum (mm)'].values
    rain_score = np.zeros(len(df))
    rain_score[rain > meta['rain_trigger_extreme']] = 30
    rain_score[(rain > meta['rain_trigger_high']) & (rain <= meta['rain_trigger_extreme'])] = 22
    rain_score[(rain > meta['rain_trigger_moderate']) & (rain <= meta['rain_trigger_high'])] = 14
    rain_score[(rain > 20) & (rain <= meta['rain_trigger_moderate'])] = 5
    score += rain_score

    # 2) Antecedent precipitation factor (max 15 points)
    api = df['antecedent_precip_index (mm)'].values
    api_score = np.zeros(len(df))
    api_score[api > meta['api_trigger'] * 2] = 15
    api_score[(api > meta['api_trigger']) & (api <= meta['api_trigger'] * 2)] = 10
    api_score[(api > meta['api_trigger'] * 0.5) & (api <= meta['api_trigger'])] = 5
    score += api_score

    # 3) Slope factor (max 20 points)
    slope = df['slope_deg'].values
    slope_score = np.zeros(len(df))
    slope_score[slope > 45] = 20
    slope_score[(slope > meta['slope_high_risk']) & (slope <= 45)] = 15
    slope_score[(slope > 25) & (slope <= meta['slope_high_risk'])] = 8
    slope_score[(slope > 15) & (slope <= 25)] = 3
    score += slope_score

    # 4) Soil moisture factor (max 15 points)
    sm = df['soil_moisture_0_to_7cm_mean (m³/m³)'].values
    sm_score = np.zeros(len(df))
    sm_score[sm > 0.45] = 15
    sm_score[(sm > 0.35) & (sm <= 0.45)] = 10
    sm_score[(sm > 0.25) & (sm <= 0.35)] = 5
    score += sm_score

    # 5) Geological susceptibility (max 10 points)
    geo = df['geology_susceptibility'].values
    geo_score = geo * 10  # already 0-1
    score += geo_score

    # 6) Seismic loading (max 10 points)
    pga = df['seismic_pga (g)'].values
    seis_score = np.zeros(len(df))
    seis_score[pga > 0.1] = 10   # significant earthquake
    seis_score[(pga > 0.05) & (pga <= 0.1)] = 7
    seis_score[(pga > 0.02) & (pga <= 0.05)] = 3
    score += seis_score

    # 7) Land use factor (barren/settlement add risk)
    lu = df['land_use'].values
    lu_bonus = np.zeros(len(df))
    lu_bonus[lu == 2] = 4   # barren
    lu_bonus[lu == 3] = 3   # settlement (cut slopes)
    lu_bonus[lu == 1] = 1   # agriculture (moderate)
    lu_bonus[lu == 0] = 0   # forest (stable)
    score += lu_bonus

    # Add noise
    score += np.random.normal(0, 2, len(df))
    score = np.clip(score, 0, 100)

    # --- Map score to risk levels ---
    risk = np.zeros(len(df), dtype=int)
    risk[score >= 60] = 4  # Extreme
    risk[(score >= 45) & (score < 60)] = 3  # High
    risk[(score >= 30) & (score < 45)] = 2  # Moderate
    risk[(score >= 15) & (score < 30)] = 1  # Low
    # 0 = No Risk (< 15)

    df['risk_level'] = risk
    df['hazard_score'] = np.round(score, 2)
    df['is_landslide'] = (risk >= 3).astype(int)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and handle missing values."""
    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill numeric NaN with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def load_and_preprocess_data(data_dir: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Full preprocessing pipeline: load → clean → label → combine.

    Args:
        data_dir: Directory containing region CSV files

    Returns:
        (region_data_dict, combined_dataframe)
    """
    print("Loading landslide data for all regions...")
    region_data = load_all_regions(data_dir)

    if not region_data:
        raise FileNotFoundError(f"No data files found in '{data_dir}'")

    print("\nAssigning landslide risk levels...")
    labeled_data = {}
    for region_name, df in region_data.items():
        df = clean_data(df)
        df = assign_landslide_risk(df, region_name)
        df['region'] = region_name
        labeled_data[region_name] = df

        ls_days = df['is_landslide'].sum()
        pct = ls_days / len(df) * 100
        print(f"  {region_name}: {ls_days} landslide-risk days ({pct:.1f}%)")

        for level in sorted(df['risk_level'].unique()):
            names = {0: 'No Risk', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Extreme'}
            cnt = (df['risk_level'] == level).sum()
            print(f"    {names.get(level, level)}: {cnt} ({cnt / len(df) * 100:.1f}%)")

    # Combine all regions
    combined = pd.concat(labeled_data.values(), axis=0)
    combined = combined.sort_index()

    print(f"\nCombined dataset: {len(combined):,} records, "
          f"{combined['region'].nunique()} regions")

    return labeled_data, combined
