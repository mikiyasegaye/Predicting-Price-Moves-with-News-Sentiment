"""
Tests for analysis functions.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our modules
import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

import text_processing
import temporal_analysis
import market_analysis

@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    data = {
        'headline': [
            'Tesla Stock Surges 10%',
            'Apple Reports Strong Earnings',
            'Microsoft Announces New Product'
        ],
        'publisher': ['Reuters', 'Bloomberg', 'Reuters'],
        'date': [
            '2024-01-01 10:00:00-04:00',
            '2024-01-01 11:00:00-04:00',
            '2024-01-02 09:00:00-04:00'
        ],
        'stock': ['TSLA', 'AAPL', 'MSFT']
    }
    return pd.DataFrame(data)

def test_text_processing():
    """Test text processing functions."""
    text = "Tesla Stock Surges 10% After Strong Earnings"
    tokens = text_processing.process_text(text)
    assert len(tokens) > 0
    assert all(t.isalnum() for t in tokens)

def test_temporal_analysis(sample_df):
    """Test temporal analysis functions."""
    df = temporal_analysis.parse_dates(sample_df)
    assert 'year' in df.columns
    assert 'month' in df.columns
    assert 'day' in df.columns

    stats = temporal_analysis.analyze_temporal_patterns(df)
    assert 'total_days' in stats
    assert 'avg_daily_news' in stats

def test_market_analysis(sample_df):
    """Test market analysis functions."""
    publisher_stats = market_analysis.analyze_publishers(sample_df)
    assert publisher_stats['total_publishers'] == 2
    assert 'Reuters' in publisher_stats['publisher_counts']

    stock_stats = market_analysis.analyze_stocks(sample_df)
    assert stock_stats['total_stocks'] == 3
    assert 'TSLA' in stock_stats['stock_counts']
