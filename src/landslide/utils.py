"""
Utility functions for Landslide Risk Prediction Module
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def setup_directories():
    """Create all required directories for the landslide module."""
    dirs = [
        'data/landslide/raw',
        'data/landslide/processed',
        'data/landslide/features',
        'models/landslide/ml',
        'models/landslide/dl',
        'results/landslide',
        'visualizations/landslide',
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("Landslide directories ready.")


def analyze_landslide_events(df: pd.DataFrame, region_name: str):
    """Print summary analysis for a single region."""
    region_df = df[df['region'] == region_name] if 'region' in df.columns else df

    total = len(region_df)
    ls_days = region_df['is_landslide'].sum() if 'is_landslide' in region_df.columns else 0
    pct = ls_days / total * 100 if total > 0 else 0

    print(f"\n  {region_name}: {total} records, "
          f"{ls_days} landslide-risk days ({pct:.1f}%)")

    if 'risk_level' in region_df.columns:
        names = {0: 'No Risk', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Extreme'}
        for level in sorted(region_df['risk_level'].unique()):
            cnt = (region_df['risk_level'] == level).sum()
            print(f"    {names.get(level, level)}: {cnt} ({cnt / total * 100:.1f}%)")

    # Seasonal distribution of landslide-risk days
    if 'is_landslide' in region_df.columns:
        ls_data = region_df[region_df['is_landslide'] == 1]
        if len(ls_data) > 0:
            monthly = ls_data.groupby(ls_data.index.month).size()
            peak_month = monthly.idxmax()
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
                          6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                          11: 'Nov', 12: 'Dec'}
            print(f"    Peak month: {month_names.get(peak_month, peak_month)} "
                  f"({monthly.max()} events)")

    # Yearly trend
    if 'is_landslide' in region_df.columns:
        yearly = region_df.groupby(region_df.index.year)['is_landslide'].sum()
        if len(yearly) > 0:
            worst_year = yearly.idxmax()
            print(f"    Worst year: {worst_year} ({yearly.max()} landslide-risk days)")
