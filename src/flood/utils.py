"""
Utility Functions for Flood Risk Prediction System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def setup_directories():
    """Create necessary project directories for flood module"""
    directories = [
        'data/flood/raw',
        'data/flood/processed',
        'data/flood/features',
        'models/flood/ml',
        'models/flood/dl',
        'results/flood',
        'visualizations/flood'
    ]
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("Flood project directories created")


def save_dataframe(df: pd.DataFrame, filepath: str, description: str = ""):
    """Save DataFrame with logging"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(f"Saved {description or 'data'} to {filepath} (shape: {df.shape})")


def load_dataframe(filepath: str, description: str = "") -> pd.DataFrame:
    """Load DataFrame with logging"""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded {description or 'data'} from {filepath} (shape: {df.shape})")
    return df


def get_risk_level_name(level: int) -> str:
    """Convert risk level to name"""
    names = {0: 'No Risk', 1: 'Low Risk', 2: 'Moderate Risk', 3: 'High Risk', 4: 'Extreme Risk'}
    return names.get(level, f'Level {level}')


def get_risk_color(level: int) -> str:
    """Get color for flood risk level"""
    colors = {
        0: '#4CAF50',  # Green
        1: '#FFC107',  # Amber
        2: '#FF9800',  # Orange
        3: '#F44336',  # Red
        4: '#9C27B0'   # Purple
    }
    return colors.get(level, '#9E9E9E')


def plot_risk_distribution(df: pd.DataFrame, city: str = None):
    """Plot flood risk level distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    data = df[df['city'] == city] if city and 'city' in df.columns else df
    title_suffix = f" - {city}" if city else " - All Regions"
    
    risk_counts = data['risk_level'].value_counts().sort_index()
    colors = [get_risk_color(level) for level in risk_counts.index]
    
    ax1.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor='black')
    ax1.set_xlabel('Risk Level')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Flood Risk Distribution{title_suffix}')
    ax1.set_xticks(risk_counts.index)
    ax1.set_xticklabels([get_risk_level_name(i) for i in risk_counts.index], rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    risk_pct = risk_counts / len(data) * 100
    ax2.bar(risk_pct.index, risk_pct.values, color=colors, edgecolor='black')
    ax2.set_xlabel('Risk Level')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title(f'Flood Risk Percentage{title_suffix}')
    ax2.set_xticks(risk_pct.index)
    ax2.set_xticklabels([get_risk_level_name(i) for i in risk_pct.index], rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_rainfall_trends(df: pd.DataFrame, city: str = None):
    """Plot rainfall temporal trends"""
    data = df[df['city'] == city].copy() if city and 'city' in df.columns else df.copy()
    title_suffix = f" - {city}" if city else " - All Regions"
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Annual total rainfall
    data['year'] = data.index.year
    annual = data.groupby('year')['precipitation_sum (mm)'].agg(['sum', 'max', 'mean'])
    
    axes[0].bar(annual.index, annual['sum'], color='#2196F3', alpha=0.7, label='Annual Total')
    ax2 = axes[0].twinx()
    ax2.plot(annual.index, annual['max'], 'r-o', label='Max Daily', linewidth=2)
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Annual Rainfall (mm)', color='#2196F3')
    ax2.set_ylabel('Max Daily Rainfall (mm)', color='red')
    axes[0].set_title(f'Annual Rainfall Trends{title_suffix}')
    axes[0].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[0].grid(alpha=0.3)
    
    # Monthly climatology
    data['month'] = data.index.month
    monthly = data.groupby('month')['precipitation_sum (mm)'].agg(['mean', 'std'])
    
    axes[1].bar(monthly.index, monthly['mean'], color='#2196F3', alpha=0.7, yerr=monthly['std'], capsize=3)
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Mean Daily Rainfall (mm)')
    axes[1].set_title(f'Monthly Rainfall Climatology{title_suffix}')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_water_level_analysis(df: pd.DataFrame, city: str, danger_level: float, warning_level: float):
    """Plot water level analysis with danger/warning levels"""
    data = df[df['city'] == city].copy() if 'city' in df.columns else df.copy()
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    ax.plot(data.index, data['water_level (m)'], color='#2196F3', alpha=0.7, linewidth=0.8, label='Water Level')
    ax.axhline(y=danger_level, color='red', linestyle='--', linewidth=2, label=f'Danger Level ({danger_level}m)')
    ax.axhline(y=warning_level, color='orange', linestyle='--', linewidth=2, label=f'Warning Level ({warning_level}m)')
    
    # Highlight flood events
    flood_mask = data['water_level (m)'] >= warning_level
    if flood_mask.any():
        ax.fill_between(data.index, data['water_level (m)'], warning_level,
                       where=flood_mask, color='red', alpha=0.2, label='Above Warning')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Water Level (m)')
    ax.set_title(f'Water Level Time Series - {city}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def analyze_flood_events(df: pd.DataFrame, city: str = None):
    """Analyze flood events statistics"""
    data = df[df['city'] == city].copy() if city and 'city' in df.columns else df.copy()
    title_suffix = f" in {city}" if city else " (All Regions)"
    
    flood_days = data[data['is_flood'] == 1]
    
    print(f"\n{'='*80}")
    print(f"FLOOD ANALYSIS{title_suffix}")
    print(f"{'='*80}")
    
    print(f"\nTotal days analyzed: {len(data):,}")
    print(f"Flood risk days (High+Extreme): {len(flood_days)} ({len(flood_days)/len(data)*100:.2f}%)")
    
    if len(flood_days) > 0:
        print(f"\nRainfall during flood events:")
        print(f"  Mean daily rainfall: {flood_days['precipitation_sum (mm)'].mean():.1f} mm")
        print(f"  Max daily rainfall: {flood_days['precipitation_sum (mm)'].max():.1f} mm")
        
        if 'river_discharge (cumecs)' in flood_days.columns:
            print(f"\nRiver discharge during flood events:")
            print(f"  Mean discharge: {flood_days['river_discharge (cumecs)'].mean():.0f} cumecs")
            print(f"  Max discharge: {flood_days['river_discharge (cumecs)'].max():.0f} cumecs")
        
        if 'water_level (m)' in flood_days.columns:
            print(f"\nWater level during flood events:")
            print(f"  Mean water level: {flood_days['water_level (m)'].mean():.2f} m")
            print(f"  Max water level: {flood_days['water_level (m)'].max():.2f} m")
        
        flood_days_copy = flood_days.copy()
        flood_days_copy['year'] = flood_days_copy.index.year
        print(f"\nFlood days by year:")
        for year, count in flood_days_copy.groupby('year').size().items():
            print(f"  {year}: {count} days")
        
        flood_days_copy['month'] = flood_days_copy.index.month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print(f"\nFlood days by month:")
        for month, count in flood_days_copy.groupby('month').size().items():
            print(f"  {month_names[month-1]}: {count} days")
