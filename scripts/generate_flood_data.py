"""
Flood Data Generation Script
Generates realistic synthetic hydrometeorological data for Indian flood-prone regions
calibrated against historical flood events (2010-2024)

Sources referenced for calibration:
- India Meteorological Department (IMD) rainfall records
- Central Water Commission (CWC) flood reports
- India-WRIS (Water Resources Information System)
- Open-Meteo Historical Weather API format
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)  # Reproducibility


# =============================================================================
# City/Region Configurations - Based on real climatology of Indian flood-prone areas
# =============================================================================

CITY_CONFIGS = {
    'Guwahati': {
        'lat': 26.14, 'lon': 91.74,
        'filename': 'Guwahati_flood.csv',
        'description': 'Brahmaputra River Basin - Assam',
        # Monthly mean rainfall (mm) - based on IMD normals
        'monthly_rain_mean': [8, 16, 45, 120, 240, 330, 370, 310, 240, 110, 18, 5],
        'monthly_rain_std':  [6, 12, 30,  55, 110, 120, 140, 100,  90,  60, 12, 4],
        # Monthly mean temperature (°C)
        'monthly_temp_mean': [16, 19, 23, 26, 28, 29, 30, 30, 29, 27, 22, 17],
        'monthly_temp_std':  [2.0, 2.5, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0, 1.2, 1.5, 2.0, 2.0],
        # Base river discharge (cumecs) - Brahmaputra
        'monthly_discharge_mean': [2000, 2200, 3500, 7000, 15000, 25000, 35000, 30000, 22000, 12000, 5000, 2800],
        'monthly_discharge_std':   [400,  500,  800, 2000,  5000,  8000, 12000,  9000,  7000,  3000, 1000,  500],
        'flood_discharge_threshold': 40000,
        'danger_level_m': 49.68,  # CWC danger level
        'warning_level_m': 48.68,
        'base_water_level': 45.0,
        'monsoon_intensity': 1.3,
        'flood_prone_months': [5, 6, 7, 8, 9],
        # Known major flood years (amplified rainfall)
        'flood_years': {2012: 1.4, 2014: 1.3, 2016: 1.5, 2017: 1.4, 2019: 1.3, 2020: 1.4, 2022: 1.5, 2024: 1.3}
    },
    'Patna': {
        'lat': 25.60, 'lon': 85.10,
        'filename': 'Patna_flood.csv',
        'description': 'Ganga-Kosi River Basin - Bihar',
        'monthly_rain_mean': [12, 12, 8, 15, 60, 160, 310, 270, 210, 60, 8, 5],
        'monthly_rain_std':  [8,  8,  6, 10, 35,  70, 110,  90,  80, 35, 6, 4],
        'monthly_temp_mean': [15, 19, 26, 33, 36, 34, 31, 30, 30, 28, 22, 16],
        'monthly_temp_std':  [2.5, 2.5, 2.0, 1.5, 1.5, 1.2, 1.0, 1.0, 1.2, 1.5, 2.0, 2.5],
        'monthly_discharge_mean': [1500, 1500, 1800, 3000, 8000, 15000, 28000, 25000, 18000, 8000, 3000, 1800],
        'monthly_discharge_std':   [300,  300,  400,  800, 2500,  5000, 10000,  8000,  6000, 2500,  600,  300],
        'flood_discharge_threshold': 30000,
        'danger_level_m': 50.00,
        'warning_level_m': 49.00,
        'base_water_level': 45.5,
        'monsoon_intensity': 1.2,
        'flood_prone_months': [6, 7, 8, 9],
        'flood_years': {2013: 1.3, 2016: 1.4, 2017: 1.5, 2019: 1.6, 2020: 1.3, 2021: 1.3, 2023: 1.3, 2024: 1.4}
    },
    'Kochi': {
        'lat': 9.93, 'lon': 76.27,
        'filename': 'Kochi_flood.csv',
        'description': 'Periyar River Basin - Kerala',
        'monthly_rain_mean': [18, 25, 45, 120, 280, 600, 580, 380, 280, 290, 160, 40],
        'monthly_rain_std':  [12, 18, 30,  55, 100, 180, 170, 120, 100, 110,  70, 25],
        'monthly_temp_mean': [27, 28, 29, 30, 29, 27, 26, 26, 27, 27, 27, 27],
        'monthly_temp_std':  [0.8, 0.8, 0.8, 0.7, 0.8, 0.6, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8],
        'monthly_discharge_mean': [50, 40, 35, 80, 250, 600, 550, 400, 300, 280, 150, 70],
        'monthly_discharge_std':   [15, 12, 10, 30,  80, 200, 180, 130, 100,  90,  50, 20],
        'flood_discharge_threshold': 800,
        'danger_level_m': 6.50,
        'warning_level_m': 5.50,
        'base_water_level': 2.5,
        'monsoon_intensity': 1.4,
        'flood_prone_months': [6, 7, 8, 9, 10],
        'flood_years': {2018: 2.0, 2019: 1.5, 2020: 1.3, 2021: 1.4, 2024: 1.3}  # 2018 was catastrophic
    },
    'Mumbai': {
        'lat': 19.08, 'lon': 72.88,
        'filename': 'Mumbai_flood.csv',
        'description': 'Mithi River / Urban Catchment - Maharashtra',
        'monthly_rain_mean': [1, 1, 0, 2, 20, 530, 840, 560, 320, 60, 12, 3],
        'monthly_rain_std':  [2, 2, 1, 3, 15, 180, 250, 180, 120, 40, 10, 3],
        'monthly_temp_mean': [24, 25, 27, 29, 31, 30, 28, 28, 28, 29, 27, 25],
        'monthly_temp_std':  [1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 1.0],
        'monthly_discharge_mean': [5, 5, 3, 5, 20, 200, 350, 250, 150, 30, 10, 5],
        'monthly_discharge_std':   [3, 3, 2, 3, 10,  80, 120,  90,  60, 15,  5, 3],
        'flood_discharge_threshold': 450,
        'danger_level_m': 4.20,
        'warning_level_m': 3.50,
        'base_water_level': 1.0,
        'monsoon_intensity': 1.5,
        'flood_prone_months': [6, 7, 8, 9],
        'flood_years': {2014: 1.3, 2017: 1.5, 2019: 1.4, 2020: 1.3, 2023: 1.4, 2024: 1.5}
    },
    'Dehradun': {
        'lat': 30.32, 'lon': 78.03,
        'filename': 'Dehradun_flood.csv',
        'description': 'Himalayan Catchment - Uttarakhand (Flash Floods)',
        'monthly_rain_mean': [40, 50, 40, 20, 50, 200, 400, 380, 200, 40, 8, 20],
        'monthly_rain_std':  [25, 30, 25, 15, 30,  80, 150, 140,  80, 25, 6, 15],
        'monthly_temp_mean': [9, 12, 17, 22, 27, 28, 26, 25, 24, 20, 14, 10],
        'monthly_temp_std':  [2.0, 2.5, 2.0, 1.5, 1.5, 1.2, 1.0, 1.0, 1.2, 1.5, 2.0, 2.0],
        'monthly_discharge_mean': [30, 35, 60, 50, 80, 250, 500, 480, 300, 80, 40, 30],
        'monthly_discharge_std':   [10, 12, 20, 15, 30,  90, 180, 170, 100, 30, 12, 10],
        'flood_discharge_threshold': 650,
        'danger_level_m': 7.00,
        'warning_level_m': 6.00,
        'base_water_level': 3.0,
        'monsoon_intensity': 1.3,
        'flood_prone_months': [6, 7, 8, 9],
        'flood_years': {2010: 1.3, 2013: 2.0, 2014: 1.3, 2016: 1.3, 2021: 1.4, 2023: 1.5}  # 2013 Uttarakhand disaster
    }
}


def generate_base_weather(city: str, config: dict, start_date: str = '2010-01-01',
                          end_date: str = '2024-12-31') -> pd.DataFrame:
    """Generate base daily weather data using seasonal climatology + noise"""
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    df = pd.DataFrame({'time': dates})
    
    months = df['time'].dt.month.values
    years = df['time'].dt.year.values
    doy = df['time'].dt.dayofyear.values
    
    # --- Temperature ---
    temp_mean = np.array([config['monthly_temp_mean'][m-1] for m in months])
    temp_std = np.array([config['monthly_temp_std'][m-1] for m in months])
    
    # Add inter-annual variability + daily noise
    daily_temp_mean = temp_mean + np.random.normal(0, temp_std * 0.6, n)
    # Smooth with 3-day filter for realism
    daily_temp_mean = pd.Series(daily_temp_mean).rolling(3, center=True, min_periods=1).mean().values
    
    temp_range = 7 + 3 * np.sin(2 * np.pi * doy / 365) + np.random.normal(0, 1.5, n)
    temp_range = np.clip(temp_range, 3, 15)
    
    df['temperature_2m_mean (°C)'] = np.round(daily_temp_mean, 1)
    df['temperature_2m_max (°C)'] = np.round(daily_temp_mean + temp_range / 2, 1)
    df['temperature_2m_min (°C)'] = np.round(daily_temp_mean - temp_range / 2, 1)
    
    # --- Precipitation ---
    monthly_rain_mean = np.array([config['monthly_rain_mean'][m-1] for m in months])
    monthly_rain_std = np.array([config['monthly_rain_std'][m-1] for m in months])
    
    # Apply flood year multipliers
    multiplier = np.ones(n)
    for yr, mult in config.get('flood_years', {}).items():
        yr_mask = (years == yr)
        flood_months = config['flood_prone_months']
        month_mask = np.isin(months, flood_months)
        multiplier[yr_mask & month_mask] = mult
    
    # Daily rainfall: use gamma distribution for realistic rainfall patterns
    # P(rain on day) depends on month (monsoon vs dry)
    rain_prob = np.array([
        min(0.95, max(0.05, config['monthly_rain_mean'][m-1] / 300))
        for m in months
    ])
    
    rain_occurs = np.random.random(n) < rain_prob
    
    # Rainfall amount on rainy days (gamma distribution)
    daily_rain_mean = monthly_rain_mean / 30 * multiplier
    shape_param = 0.8  # Gamma shape (skewed - many light, few heavy rain days)
    scale_param = daily_rain_mean / shape_param
    scale_param = np.clip(scale_param, 0.1, None)
    
    rainfall = np.zeros(n)
    rainfall[rain_occurs] = np.random.gamma(shape_param, scale_param[rain_occurs])
    
    # Inject extreme rainfall events during flood years in monsoon months
    for yr, mult in config.get('flood_years', {}).items():
        yr_mask = (years == yr)
        flood_months = config['flood_prone_months']
        for fm in flood_months:
            fm_mask = yr_mask & (months == fm)
            n_extreme = max(1, int(np.sum(fm_mask) * 0.1 * (mult - 1) * 5))
            extreme_days = np.where(fm_mask)[0]
            if len(extreme_days) > 0:
                selected = np.random.choice(extreme_days, min(n_extreme, len(extreme_days)), replace=False)
                rainfall[selected] += np.random.exponential(50 * mult, len(selected))
    
    df['precipitation_sum (mm)'] = np.round(np.clip(rainfall, 0, 600), 1)
    df['rain_sum (mm)'] = df['precipitation_sum (mm)']
    
    # Rain hours
    df['precipitation_hours (h)'] = np.where(
        df['precipitation_sum (mm)'] > 0,
        np.clip(np.round(df['precipitation_sum (mm)'] / 8 + np.random.normal(0, 2, n), 0), 1, 24),
        0
    )
    
    # --- Humidity ---
    base_humidity = 40 + 35 * np.sin(2 * np.pi * (doy - 90) / 365)  # peaks in monsoon
    humidity = base_humidity + np.random.normal(0, 8, n)
    # Higher humidity on rainy days
    humidity[rain_occurs] += 10
    df['relative_humidity_2m_mean (%)'] = np.round(np.clip(humidity, 20, 100), 1)
    
    # --- Dew Point ---
    df['dew_point_2m_mean (°C)'] = np.round(
        daily_temp_mean - (100 - df['relative_humidity_2m_mean (%)'].values) / 5, 1
    )
    
    # --- Cloud Cover ---
    cloud_base = 20 + 40 * np.sin(2 * np.pi * (doy - 60) / 365)
    cloud = cloud_base + np.random.normal(0, 15, n)
    cloud[rain_occurs] += 20
    df['cloud_cover_mean (%)'] = np.round(np.clip(cloud, 0, 100), 1)
    
    # --- Wind Speed ---
    wind_base = 8 + 4 * np.sin(2 * np.pi * (doy - 30) / 365)
    wind = wind_base + np.random.normal(0, 3, n)
    df['wind_speed_10m_mean (km/h)'] = np.round(np.clip(wind, 1, 50), 1)
    df['wind_gusts_10m_max (km/h)'] = np.round(df['wind_speed_10m_mean (km/h)'] * (1.5 + np.random.exponential(0.3, n)), 1)
    
    # --- Pressure ---
    pressure_base = 1010 - 5 * np.sin(2 * np.pi * (doy - 200) / 365)
    pressure = pressure_base + np.random.normal(0, 3, n)
    # Drop pressure during heavy rain/cyclonic events
    heavy_rain_mask = df['precipitation_sum (mm)'] > 50
    pressure[heavy_rain_mask] -= np.random.uniform(3, 10, np.sum(heavy_rain_mask))
    df['pressure_msl_mean (hPa)'] = np.round(np.clip(pressure, 980, 1030), 1)
    
    # --- Soil Moisture ---
    soil_base = 0.15 + 0.15 * np.sin(2 * np.pi * (doy - 120) / 365)
    soil = soil_base + np.random.normal(0, 0.03, n)
    # Increase soil moisture after rain (with lag)
    rain_effect = pd.Series(df['precipitation_sum (mm)'].values / 500).rolling(3, min_periods=1).mean().values
    soil += rain_effect
    df['soil_moisture_0_to_7cm_mean (m³/m³)'] = np.round(np.clip(soil, 0.05, 0.55), 3)
    df['soil_moisture_7_to_28cm_mean (m³/m³)'] = np.round(
        np.clip(soil * 0.9 + np.random.normal(0, 0.01, n), 0.05, 0.50), 3
    )
    
    # --- Soil Temperature ---
    df['soil_temperature_0_to_7cm_mean (°C)'] = np.round(daily_temp_mean * 0.85 + 3, 1)
    
    # --- Radiation ---
    rad_base = 12 + 8 * np.cos(2 * np.pi * (doy - 172) / 365)
    rad = rad_base + np.random.normal(0, 2, n)
    rad[rain_occurs] *= 0.5
    df['shortwave_radiation_sum (MJ/m²)'] = np.round(np.clip(rad, 1, 30), 2)
    
    # --- ET0 ---
    df['et0_fao_evapotranspiration (mm)'] = np.round(
        np.clip(0.3 * df['shortwave_radiation_sum (MJ/m²)'] + 0.05 * daily_temp_mean - 0.01 * humidity + np.random.normal(0, 0.5, n), 0.5, 10), 2
    )
    
    # --- VPD ---
    vpd = 0.6108 * np.exp(17.27 * daily_temp_mean / (daily_temp_mean + 237.3)) * (1 - humidity / 100)
    df['vapour_pressure_deficit_max (kPa)'] = np.round(np.clip(vpd, 0, 5), 2)
    
    # --- River Discharge (cumecs) ---
    discharge_mean = np.array([config['monthly_discharge_mean'][m-1] for m in months])
    discharge_std = np.array([config['monthly_discharge_std'][m-1] for m in months])
    
    discharge = discharge_mean + np.random.normal(0, discharge_std * 0.5, n)
    
    # Discharge responds to cumulative rainfall (3-day lag)
    cum_rain_3d = pd.Series(df['precipitation_sum (mm)'].values).rolling(3, min_periods=1).sum().values
    discharge += cum_rain_3d * config['monsoon_intensity'] * 50
    
    # Apply flood year multipliers
    for yr, mult in config.get('flood_years', {}).items():
        yr_mask = (years == yr)
        flood_months = config['flood_prone_months']
        month_mask = np.isin(months, flood_months)
        discharge[yr_mask & month_mask] *= mult
    
    df['river_discharge (cumecs)'] = np.round(np.clip(discharge, 10, None), 1)
    
    # --- Water Level ---
    # Water level correlates with discharge (log relationship)
    base_level = config['base_water_level']
    level_range = config['danger_level_m'] - base_level
    
    discharge_normalized = (df['river_discharge (cumecs)'] - df['river_discharge (cumecs)'].min()) / \
                          (df['river_discharge (cumecs)'].max() - df['river_discharge (cumecs)'].min() + 1)
    
    water_level = base_level + level_range * np.power(discharge_normalized, 0.6)
    water_level += np.random.normal(0, 0.2, n)
    df['water_level (m)'] = np.round(water_level, 2)
    
    # --- Reservoir Storage (% of capacity) ---
    storage_base = 30 + 40 * np.sin(2 * np.pi * (doy - 90) / 365)
    storage = storage_base + np.random.normal(0, 8, n)
    rain_cum_30d = pd.Series(df['precipitation_sum (mm)'].values).rolling(30, min_periods=1).sum().values
    storage += rain_cum_30d / 100
    df['reservoir_storage_pct (%)'] = np.round(np.clip(storage, 5, 100), 1)
    
    # --- Antecedent Precipitation Index (API) ---
    # Weighted sum of past rainfall (decaying weights)
    api = np.zeros(n)
    decay = 0.85
    for i in range(1, n):
        api[i] = decay * api[i-1] + df['precipitation_sum (mm)'].iloc[i]
    df['antecedent_precip_index (mm)'] = np.round(api, 1)
    
    # --- Catchment Wetness Index ---
    df['catchment_wetness_index'] = np.round(
        (df['soil_moisture_0_to_7cm_mean (m³/m³)'] * 100 + 
         df['antecedent_precip_index (mm)'] / 10 +
         df['relative_humidity_2m_mean (%)'] / 10), 2
    )
    
    # Format time column
    df['time'] = df['time'].dt.strftime('%d-%m-%Y')
    
    return df


def generate_flood_data_for_city(city: str, config: dict, output_dir: str = 'flood/') -> pd.DataFrame:
    """Generate complete flood dataset for a city and save to CSV"""
    
    print(f"\nGenerating flood data for {city} ({config['description']})...")
    
    df = generate_base_weather(city, config)
    
    print(f"  Generated {len(df)} daily records (2010-2024)")
    print(f"  Columns: {len(df.columns)}")
    
    # Stats
    print(f"  Max daily rainfall: {df['precipitation_sum (mm)'].max():.1f} mm")
    print(f"  Max river discharge: {df['river_discharge (cumecs)'].max():.1f} cumecs")
    print(f"  Max water level: {df['water_level (m)'].max():.2f} m")
    
    # Save
    output_path = Path(output_dir) / config['filename']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    
    return df


def main():
    """Generate flood data for all cities"""
    
    print("="*80)
    print("FLOOD DATA GENERATION")
    print("Indian Flood-Prone Regions - Synthetic Hydrometeorological Data (2010-2024)")
    print("="*80)
    
    output_dir = 'flood/'
    all_data = {}
    
    for city, config in CITY_CONFIGS.items():
        all_data[city] = generate_flood_data_for_city(city, config, output_dir)
    
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE")
    print("="*80)
    print(f"\nCities: {', '.join(all_data.keys())}")
    print(f"Files saved to: {output_dir}")
    total_records = sum(len(df) for df in all_data.values())
    print(f"Total records: {total_records:,}")
    
    return all_data


if __name__ == "__main__":
    main()
