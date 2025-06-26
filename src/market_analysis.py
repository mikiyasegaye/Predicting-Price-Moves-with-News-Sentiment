"""
Market analysis utilities for analyzing publishers and stocks.
"""
import pandas as pd
from typing import Dict, List, Any
from collections import Counter

def analyze_publishers(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze news publisher patterns.

    Args:
        df: Input dataframe with publisher information

    Returns:
        Dictionary containing publisher analysis results
    """
    # Publisher frequency
    publisher_counts = df['publisher'].value_counts()

    # Articles per day by publisher
    df['date'] = pd.to_datetime(df['date'])
    articles_per_day = df.groupby(['publisher', df['date'].dt.date]).size()
    avg_articles_per_day = articles_per_day.groupby('publisher').mean()

    # Most covered stocks by publisher
    publisher_stocks = df.groupby('publisher')['stock'].agg(list)
    publisher_top_stocks = {
        publisher: Counter(stocks).most_common(5)
        for publisher, stocks in publisher_stocks.items()
    }

    return {
        'total_publishers': len(publisher_counts),
        'publisher_counts': publisher_counts,
        'avg_articles_per_day': avg_articles_per_day,
        'publisher_top_stocks': publisher_top_stocks
    }

def analyze_stocks(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze stock coverage patterns.

    Args:
        df: Input dataframe with stock information

    Returns:
        Dictionary containing stock analysis results
    """
    # Stock frequency
    stock_counts = df['stock'].value_counts()

    # News volume over time
    df['date'] = pd.to_datetime(df['date'])
    daily_volume = df.groupby(['stock', df['date'].dt.date]).size()
    avg_daily_volume = daily_volume.groupby('stock').mean()

    # Publisher diversity
    stock_publishers = df.groupby('stock')['publisher'].agg(list)
    publisher_diversity = {
        stock: len(set(publishers))
        for stock, publishers in stock_publishers.items()
    }

    return {
        'total_stocks': len(stock_counts),
        'stock_counts': stock_counts,
        'avg_daily_volume': avg_daily_volume,
        'publisher_diversity': publisher_diversity
    }
