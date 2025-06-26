"""
Temporal analysis utilities for news publication patterns.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta

class TemporalAnalyzer:
    def __init__(self):
        """
        Initialize TemporalAnalyzer.
        """
        pass

    def parse_dates(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Parse dates in the dataframe and add temporal features.

        Args:
            df: Input dataframe
            date_column: Name of the date column

        Returns:
            DataFrame with additional temporal features
        """
        # Create a copy to avoid modifying the original
        df = df.copy()

        # Parse dates
        df[date_column] = pd.to_datetime(df[date_column])

        # Extract temporal features
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['hour'] = df[date_column].dt.hour
        df['day_of_week'] = df[date_column].dt.day_name()
        df['is_weekend'] = df[date_column].dt.day_name().isin(['Saturday', 'Sunday'])

        return df

    def analyze_temporal_patterns(self, df: pd.DataFrame, date_column: str = 'date') -> Dict[str, Any]:
        """
        Analyze temporal patterns in news publications.

        Args:
            df: Input dataframe with datetime index
            date_column: Name of the date column

        Returns:
            Dictionary containing temporal analysis results
        """
        # Ensure dates are parsed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df = self.parse_dates(df, date_column)

        # Daily frequency
        daily_counts = df.groupby(df[date_column].dt.date).size()

        # Hourly patterns
        hourly_counts = df.groupby(df[date_column].dt.hour).size()

        # Day of week patterns
        dow_counts = df.groupby('day_of_week').size()

        # Monthly patterns
        monthly_counts = df.groupby([df[date_column].dt.year, df[date_column].dt.month]).size()

        # Calculate statistics
        stats = {
            'total_days': len(daily_counts),
            'avg_daily_news': daily_counts.mean(),
            'max_daily_news': daily_counts.max(),
            'peak_hour': hourly_counts.idxmax(),
            'weekend_ratio': df['is_weekend'].mean(),
            'daily_counts': daily_counts,
            'hourly_counts': hourly_counts,
            'dow_counts': dow_counts,
            'monthly_counts': monthly_counts
        }

        return stats
