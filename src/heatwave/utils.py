"""
Utility Functions for Heatwave Prediction System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def setup_directories():
    """Create necessary project directories for heatwave module"""
    directories = [
        'data/heatwave/raw',
        'data/heatwave/processed',
        'data/heatwave/features',
        'models/heatwave/ml',
        'models/heatwave/dl',
        'results/heatwave',
        'visualizations/heatwave'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Heatwave project directories created successfully")


def save_dataframe(df: pd.DataFrame, filepath: str, description: str = ""):
    """Save DataFrame with logging"""
    df.to_csv(filepath)
    print(f"Saved {description if description else 'data'} to {filepath}")
    print(f"  Shape: {df.shape}")


def load_dataframe(filepath: str, description: str = "") -> pd.DataFrame:
    """Load DataFrame with logging"""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded {description if description else 'data'} from {filepath}")
    print(f"  Shape: {df.shape}")
    return df


def get_risk_level_name(level: int) -> str:
    """Convert risk level to name"""
    risk_names = {
        0: 'No Risk',
        1: 'Low Risk',
        2: 'Moderate Risk',
        3: 'High Risk',
        4: 'Extreme Risk'
    }
    return risk_names.get(level, f'Level {level}')


def get_risk_color(level: int) -> str:
    """Get color for risk level visualization"""
    colors = {
        0: '#2ecc71',  # Green
        1: '#f1c40f',  # Yellow
        2: '#e67e22',  # Orange
        3: '#e74c3c',  # Red
        4: '#8e44ad'   # Purple
    }
    return colors.get(level, '#95a5a6')


def plot_risk_distribution(df: pd.DataFrame, city: str = None):
    """Plot risk level distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if city and 'city' in df.columns:
        data = df[df['city'] == city]
        title_suffix = f" - {city}"
    else:
        data = df
        title_suffix = " - All Cities"
    
    # Count plot
    risk_counts = data['risk_level'].value_counts().sort_index()
    colors = [get_risk_color(level) for level in risk_counts.index]
    
    ax1.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor='black')
    ax1.set_xlabel('Risk Level')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Risk Level Distribution{title_suffix}')
    ax1.set_xticks(risk_counts.index)
    ax1.set_xticklabels([get_risk_level_name(i) for i in risk_counts.index], rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Percentage plot
    risk_pct = (risk_counts / len(data) * 100)
    ax2.bar(risk_pct.index, risk_pct.values, color=colors, edgecolor='black')
    ax2.set_xlabel('Risk Level')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title(f'Risk Level Percentage{title_suffix}')
    ax2.set_xticks(risk_pct.index)
    ax2.set_xticklabels([get_risk_level_name(i) for i in risk_pct.index], rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_temporal_trends(df: pd.DataFrame, city: str = None, 
                        metric: str = 'temperature_2m_max (°C)'):
    """Plot temporal trends"""
    if city and 'city' in df.columns:
        data = df[df['city'] == city].copy()
        title_suffix = f" - {city}"
    else:
        data = df.copy()
        title_suffix = " - All Cities"
    
    # Yearly trend
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Annual trend
    data['year'] = data.index.year
    annual_data = data.groupby('year')[metric].agg(['mean', 'max', 'min'])
    
    axes[0].plot(annual_data.index, annual_data['mean'], marker='o', label='Mean', linewidth=2)
    axes[0].fill_between(annual_data.index, annual_data['min'], annual_data['max'], 
                         alpha=0.3, label='Min-Max Range')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel(metric)
    axes[0].set_title(f'Annual Trend: {metric}{title_suffix}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Monthly climatology
    data['month'] = data.index.month
    monthly_data = data.groupby('month')[metric].agg(['mean', 'std'])
    
    axes[1].plot(monthly_data.index, monthly_data['mean'], marker='o', linewidth=2)
    axes[1].fill_between(monthly_data.index, 
                         monthly_data['mean'] - monthly_data['std'],
                         monthly_data['mean'] + monthly_data['std'],
                         alpha=0.3)
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel(metric)
    axes[1].set_title(f'Monthly Climatology: {metric}{title_suffix}')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_heatwave_events(df: pd.DataFrame, city: str = None):
    """Analyze heatwave events"""
    if city and 'city' in df.columns:
        data = df[df['city'] == city].copy()
        title_suffix = f" in {city}"
    else:
        data = df.copy()
        title_suffix = " (All Cities)"
    
    # Identify heatwave events
    heatwave_days = data[data['is_heatwave'] == 1]
    
    print(f"\n{'='*80}")
    print(f"HEATWAVE ANALYSIS{title_suffix}")
    print(f"{'='*80}")
    
    print(f"\nTotal days analyzed: {len(data)}")
    print(f"Heatwave days: {len(heatwave_days)} ({len(heatwave_days)/len(data)*100:.2f}%)")
    
    if len(heatwave_days) > 0:
        print(f"\nTemperature statistics during heatwaves:")
        print(f"  Mean max temperature: {heatwave_days['temperature_2m_max (°C)'].mean():.2f}°C")
        print(f"  Peak temperature: {heatwave_days['temperature_2m_max (°C)'].max():.2f}°C")
        print(f"  Mean temperature departure: {heatwave_days['temp_departure'].mean():.2f}°C")
        
        # Analyze by year
        print(f"\nHeatwave days by year:")
        heatwave_days['year'] = heatwave_days.index.year
        yearly_counts = heatwave_days.groupby('year').size()
        for year, count in yearly_counts.items():
            print(f"  {year}: {count} days")
        
        # Analyze by month
        print(f"\nHeatwave days by month:")
        heatwave_days['month'] = heatwave_days.index.month
        monthly_counts = heatwave_days.groupby('month').size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, count in monthly_counts.items():
            print(f"  {month_names[month-1]}: {count} days")


def create_summary_statistics(df: pd.DataFrame, output_file: str = None):
    """Create summary statistics table"""
    summary = {
        'Metric': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        'Max': []
    }
    
    key_metrics = [
        'temperature_2m_max (°C)',
        'temperature_2m_mean (°C)',
        'relative_humidity_2m_mean (%)',
        'precipitation_sum (mm)',
        'wind_speed_10m_mean (km/h)',
        'heat_index'
    ]
    
    for metric in key_metrics:
        if metric in df.columns:
            summary['Metric'].append(metric)
            summary['Mean'].append(df[metric].mean())
            summary['Std'].append(df[metric].std())
            summary['Min'].append(df[metric].min())
            summary['Max'].append(df[metric].max())
    
    summary_df = pd.DataFrame(summary)
    
    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"Summary statistics saved to {output_file}")
    
    return summary_df


if __name__ == "__main__":
    print("Utility functions loaded successfully")
    setup_directories()
