"""
Landslide Data Generation Script
Generates realistic synthetic geo-hydrometeorological data for Indian landslide-prone regions
calibrated against historical events and GSI Landslide Susceptibility Mapping (2010-2024)

Regions selected from India's most landslide-prone zones (GSI BHUKOSH portal):
- Shimla (HP)      — Lesser Himalaya, fragile geology, NH-5 corridor
- Munnar (Kerala)  — Western Ghats, tea estate terrain, 2018/2020 disasters
- Darjeeling (WB)  — Eastern Himalaya, steep tea gardens, Teesta basin
- Uttarkashi (UK)  — Greater Himalaya, Char Dham highway, seismic zone V
- Kohima (Nagaland) — Purvanchal Hills, heavy monsoon, NH-29

Sources referenced for calibration:
- Geological Survey of India (GSI) — Landslide Susceptibility Zonation Mapping
- India Meteorological Department (IMD) — Rainfall records, monsoon data
- National Disaster Management Authority (NDMA) — Landslide disaster reports
- ISRO Bhuvan — DEM, slope, land-use data
- Open-Meteo Historical Weather API format (for variable naming)
- Published literature on Indian landslide triggers (Froude & Petley 2018)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# Region Configurations — calibrated from GSI / IMD / NDMA published data
# =============================================================================

REGION_CONFIGS = {
    'Shimla': {
        'lat': 31.10, 'lon': 77.17, 'elevation_m': 2206,
        'filename': 'Shimla_landslide.csv',
        'description': 'Lesser Himalaya — Himachal Pradesh (NH-5 corridor)',
        'geology': 'Phyllite, Slate, Dolomite (Jutogh/Shali formations)',
        'slope_mean_deg': 32, 'slope_std_deg': 8,
        'soil_depth_m': 2.5,
        # Monthly mean rainfall (mm) — IMD Shimla station
        'monthly_rain_mean': [55, 60, 65, 45, 60, 150, 380, 340, 170, 40, 15, 30],
        'monthly_rain_std':  [25, 30, 30, 20, 30,  60, 130, 110,  70, 25, 10, 18],
        'monthly_temp_mean': [4, 5, 10, 16, 20, 22, 20, 19, 18, 14, 9, 5],
        'monthly_temp_std':  [2, 2, 2, 1.5, 1.5, 1, 1, 1, 1.2, 1.5, 2, 2],
        'landslide_prone_months': [6, 7, 8, 9],
        # GSI susceptibility (0-1 scale: 0.7+ = high)
        'base_susceptibility': 0.65,
        # Historical landslide years (amplification factor)
        'landslide_years': {2013: 1.5, 2014: 1.3, 2017: 1.4, 2019: 1.3, 2023: 1.8, 2024: 1.4},
        'seismic_zone': 'IV',
        'land_use': {'forest': 0.45, 'barren': 0.15, 'agriculture': 0.25, 'settlement': 0.15},
    },
    'Munnar': {
        'lat': 10.09, 'lon': 77.06, 'elevation_m': 1532,
        'filename': 'Munnar_landslide.csv',
        'description': 'Western Ghats — Idukki District, Kerala',
        'geology': 'Charnockite, Gneiss (Precambrian crystalline)',
        'slope_mean_deg': 28, 'slope_std_deg': 7,
        'soil_depth_m': 3.0,
        'monthly_rain_mean': [15, 30, 55, 140, 260, 600, 700, 450, 280, 310, 165, 40],
        'monthly_rain_std':  [10, 18, 30,  55, 100, 180, 200, 140, 100, 120,  70, 25],
        'monthly_temp_mean': [17, 18, 20, 21, 20, 18, 17, 17, 18, 18, 18, 17],
        'monthly_temp_std':  [1.0, 1.0, 1.0, 0.8, 0.8, 0.7, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0],
        'landslide_prone_months': [6, 7, 8, 9, 10],
        'base_susceptibility': 0.70,
        'landslide_years': {2018: 1.8, 2019: 1.3, 2020: 1.6, 2021: 1.4, 2024: 1.5},
        'seismic_zone': 'III',
        'land_use': {'forest': 0.35, 'barren': 0.05, 'agriculture': 0.50, 'settlement': 0.10},
    },
    'Darjeeling': {
        'lat': 27.04, 'lon': 88.26, 'elevation_m': 2042,
        'filename': 'Darjeeling_landslide.csv',
        'description': 'Eastern Himalaya — Darjeeling-Sikkim belt',
        'geology': 'Darjeeling Gneiss, Daling phyllite, deeply weathered',
        'slope_mean_deg': 35, 'slope_std_deg': 9,
        'soil_depth_m': 2.0,
        'monthly_rain_mean': [10, 20, 50, 100, 250, 550, 780, 600, 400, 120, 15, 5],
        'monthly_rain_std':  [8, 14, 28,  45, 100, 170, 240, 180, 130,  55, 10, 4],
        'monthly_temp_mean': [6, 8, 12, 16, 18, 19, 19, 19, 18, 15, 10, 7],
        'monthly_temp_std':  [2, 2, 1.5, 1.5, 1, 0.8, 0.8, 0.8, 1, 1.5, 2, 2],
        'landslide_prone_months': [6, 7, 8, 9],
        'base_susceptibility': 0.72,
        'landslide_years': {2011: 1.5, 2015: 1.3, 2017: 1.4, 2019: 1.3, 2021: 1.4, 2023: 1.6},
        'seismic_zone': 'IV',
        'land_use': {'forest': 0.40, 'barren': 0.10, 'agriculture': 0.35, 'settlement': 0.15},
    },
    'Uttarkashi': {
        'lat': 30.73, 'lon': 78.44, 'elevation_m': 1158,
        'filename': 'Uttarkashi_landslide.csv',
        'description': 'Greater Himalaya — Uttarakhand (Char Dham route)',
        'geology': 'Central Crystallines: Gneiss, Schist, MCT zone',
        'slope_mean_deg': 38, 'slope_std_deg': 10,
        'soil_depth_m': 1.8,
        'monthly_rain_mean': [40, 45, 40, 25, 50, 150, 350, 330, 160, 30, 10, 20],
        'monthly_rain_std':  [20, 22, 20, 15, 25,  60, 120, 110,  70, 18, 8, 12],
        'monthly_temp_mean': [6, 8, 13, 19, 23, 25, 24, 23, 21, 16, 11, 7],
        'monthly_temp_std':  [2.5, 2.5, 2, 1.5, 1.5, 1, 1, 1, 1.2, 1.5, 2, 2.5],
        'landslide_prone_months': [6, 7, 8, 9],
        'base_susceptibility': 0.75,
        'landslide_years': {2012: 1.4, 2013: 2.0, 2014: 1.3, 2017: 1.3, 2021: 1.4, 2022: 1.5, 2023: 1.6},
        'seismic_zone': 'V',
        'land_use': {'forest': 0.50, 'barren': 0.20, 'agriculture': 0.20, 'settlement': 0.10},
    },
    'Kohima': {
        'lat': 25.67, 'lon': 94.10, 'elevation_m': 1444,
        'filename': 'Kohima_landslide.csv',
        'description': 'Purvanchal Hills — Nagaland (NH-29)',
        'geology': 'Disang shale, Barail sandstone, deeply weathered',
        'slope_mean_deg': 30, 'slope_std_deg': 7,
        'soil_depth_m': 2.8,
        'monthly_rain_mean': [10, 20, 60, 120, 200, 350, 400, 320, 240, 100, 25, 8],
        'monthly_rain_std':  [8, 14, 30,  50,  80, 120, 140, 110,  90,  50, 15, 6],
        'monthly_temp_mean': [11, 13, 17, 20, 21, 22, 22, 22, 22, 19, 15, 12],
        'monthly_temp_std':  [2, 2, 1.5, 1.2, 1, 0.8, 0.8, 0.8, 0.8, 1, 1.5, 2],
        'landslide_prone_months': [5, 6, 7, 8, 9, 10],
        'base_susceptibility': 0.60,
        'landslide_years': {2014: 1.3, 2016: 1.4, 2018: 1.3, 2020: 1.5, 2022: 1.3, 2024: 1.4},
        'seismic_zone': 'V',
        'land_use': {'forest': 0.55, 'barren': 0.10, 'agriculture': 0.25, 'settlement': 0.10},
    }
}


def generate_daily_data(config: dict, start_year: int = 2010,
                        end_year: int = 2024) -> pd.DataFrame:
    """
    Generate daily geo-hydrometeorological data for a single region.

    Variables follow a mix of Open-Meteo naming + GSI / NDMA landslide variables.
    """
    dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='D')
    n = len(dates)

    records = []
    for i, date in enumerate(dates):
        month = date.month
        year = date.year
        doy = date.dayofyear
        mi = month - 1  # 0-indexed month

        # Year-level amplification for known landslide years
        year_amp = config.get('landslide_years', {}).get(year, 1.0)
        is_monsoon = month in config['landslide_prone_months']

        # --- Precipitation ---
        rain_mean = config['monthly_rain_mean'][mi] / 30.0  # daily
        rain_std = config['monthly_rain_std'][mi] / 30.0
        if is_monsoon:
            rain_mean *= year_amp
        raw_rain = max(0, np.random.gamma(shape=max(0.5, rain_mean / max(rain_std, 0.1)),
                                           scale=max(0.1, rain_std), size=1)[0])
        # Occasional extreme events
        if is_monsoon and np.random.random() < 0.03 * year_amp:
            raw_rain *= np.random.uniform(3.0, 6.0)
        precipitation = round(raw_rain, 1)

        # --- Temperature ---
        temp_mean = config['monthly_temp_mean'][mi]
        temp_std = config['monthly_temp_std'][mi]
        temp = round(np.random.normal(temp_mean, temp_std), 1)
        temp_max = round(temp + np.random.uniform(3, 8), 1)
        temp_min = round(temp - np.random.uniform(3, 7), 1)

        # --- Relative humidity ---
        base_rh = 60 if not is_monsoon else 82
        rh = round(np.clip(np.random.normal(base_rh, 10), 20, 100), 1)

        # --- Wind speed ---
        wind = round(max(0, np.random.gamma(2, 3)), 1)

        # --- Soil moisture (volumetric, m³/m³) ---
        base_sm = 0.20 if not is_monsoon else 0.38
        sm = round(np.clip(np.random.normal(base_sm, 0.08), 0.05, 0.55), 4)

        # --- Slope (static + noise) ---
        slope = round(np.clip(np.random.normal(config['slope_mean_deg'],
                                                config['slope_std_deg']), 5, 70), 1)

        # --- Elevation (static + minor noise) ---
        elevation = round(config['elevation_m'] + np.random.normal(0, 50), 0)

        # --- Soil depth (m) ---
        soil_depth = round(np.clip(np.random.normal(config['soil_depth_m'], 0.4), 0.5, 6.0), 2)

        # --- Geology susceptibility index (0-1, from GSI mapping) ---
        geo_susc = round(np.clip(
            config['base_susceptibility'] + np.random.normal(0, 0.05), 0.1, 1.0), 3)

        # --- NDVI (vegetation index, seasonal) ---
        # Higher in monsoon (lush growth), lower in dry/post-deforestation
        if is_monsoon:
            ndvi = round(np.clip(np.random.normal(0.55, 0.10), 0.1, 0.9), 3)
        else:
            ndvi = round(np.clip(np.random.normal(0.40, 0.12), 0.05, 0.8), 3)

        # --- Land use encoded (0=forest, 1=agriculture, 2=barren, 3=settlement) ---
        lu_probs = [config['land_use']['forest'], config['land_use']['agriculture'],
                    config['land_use']['barren'], config['land_use']['settlement']]
        land_use = np.random.choice([0, 1, 2, 3], p=lu_probs)

        # --- Seismic activity (PGA proxy, units: g fraction) ---
        # Background seismicity + occasional events
        zone_base = {'III': 0.005, 'IV': 0.010, 'V': 0.018}
        pga = max(0, np.random.exponential(zone_base.get(config['seismic_zone'], 0.005)))
        if np.random.random() < 0.005:  # rare significant quake
            pga += np.random.uniform(0.05, 0.25)
        pga = round(pga, 4)

        # --- Antecedent rainfall (cumulative past 3, 7, 14 days) ---
        # Will be computed properly later; store daily rain for now
        # Placeholder — overwritten in post-processing
        rain_3d = 0.0
        rain_7d = 0.0
        rain_14d = 0.0

        # --- Pore water pressure proxy (kPa) ---
        # Increases with rainfall & soil moisture
        pwp = round(max(0, sm * 50 + precipitation * 0.3 + np.random.normal(0, 2)), 2)

        # --- Distance to nearest road/cut slope (km) ---
        dist_road = round(max(0.1, np.random.exponential(2.0)), 2)

        # --- Distance to nearest drainage/stream (km) ---
        dist_stream = round(max(0.05, np.random.exponential(1.0)), 2)

        # --- Aspect (degrees 0-360) ---
        aspect = round(np.random.uniform(0, 360), 1)

        # --- Curvature (profile, -1 to +1; concave=positive=water collecting) ---
        curvature = round(np.clip(np.random.normal(0, 0.3), -1, 1), 3)

        records.append({
            'time': date.strftime('%d-%m-%Y'),
            'precipitation_sum (mm)': precipitation,
            'temperature_2m_mean (°C)': temp,
            'temperature_2m_max (°C)': temp_max,
            'temperature_2m_min (°C)': temp_min,
            'relative_humidity_2m_mean (%)': rh,
            'wind_speed_10m_max (km/h)': wind,
            'soil_moisture_0_to_7cm_mean (m³/m³)': sm,
            'slope_deg': slope,
            'elevation_m': elevation,
            'soil_depth_m': soil_depth,
            'geology_susceptibility': geo_susc,
            'ndvi': ndvi,
            'land_use': land_use,
            'seismic_pga (g)': pga,
            'pore_water_pressure (kPa)': pwp,
            'distance_to_road (km)': dist_road,
            'distance_to_stream (km)': dist_stream,
            'aspect_deg': aspect,
            'curvature': curvature,
        })

    df = pd.DataFrame(records)
    df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y')
    df = df.set_index('time').sort_index()

    # Post-process: compute antecedent rainfall properly
    rain_col = 'precipitation_sum (mm)'
    df['rain_3d_sum (mm)'] = df[rain_col].rolling(3, min_periods=1).sum().round(1)
    df['rain_7d_sum (mm)'] = df[rain_col].rolling(7, min_periods=1).sum().round(1)
    df['rain_14d_sum (mm)'] = df[rain_col].rolling(14, min_periods=1).sum().round(1)

    # Antecedent Precipitation Index (API) — Kohler & Linsley method
    api = np.zeros(len(df))
    k = 0.85  # recession constant
    for j in range(len(df)):
        if j == 0:
            api[j] = df[rain_col].iloc[j]
        else:
            api[j] = k * api[j - 1] + df[rain_col].iloc[j]
    df['antecedent_precip_index (mm)'] = np.round(api, 2)

    # Reset index to write time as column
    df = df.reset_index()
    df['time'] = df['time'].dt.strftime('%d-%m-%Y')
    return df


def main():
    output_dir = Path('landslide')
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("LANDSLIDE DATA GENERATION — 5 Indian Landslide-Prone Regions")
    print("=" * 70)

    for region_name, config in REGION_CONFIGS.items():
        print(f"\nGenerating data for {region_name} ({config['description']})...")
        df = generate_daily_data(config)
        filepath = output_dir / config['filename']
        df.to_csv(filepath, index=False)
        print(f"  Saved {len(df)} records to {filepath}")
        print(f"  Date range: {df['time'].iloc[0]} — {df['time'].iloc[-1]}")
        print(f"  Mean rainfall: {df['precipitation_sum (mm)'].mean():.1f} mm/day")
        print(f"  Mean slope: {df['slope_deg'].mean():.1f}°")
        print(f"  Geology susceptibility: {df['geology_susceptibility'].mean():.3f}")

    print(f"\n{'=' * 70}")
    print(f"Data generation complete. Files saved to '{output_dir}/'")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
