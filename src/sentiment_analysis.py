"""
Sentiment analysis utilities for news headlines.
"""
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, Any, List
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer."""
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')

        # Initialize analyzers
        self.sia = SentimentIntensityAnalyzer()

    def get_vader_sentiment(self, text: str) -> float:
        """
        Get VADER sentiment compound score.

        Args:
            text: Input text to analyze

        Returns:
            Compound sentiment score (-1 to 1)
        """
        return self.sia.polarity_scores(text)['compound']

    def get_textblob_sentiment(self, text: str) -> float:
        """
        Get TextBlob sentiment polarity.

        Args:
            text: Input text to analyze

        Returns:
            Sentiment polarity (-1 to 1)
        """
        return TextBlob(text).sentiment.polarity

    def analyze_headlines(self, df: pd.DataFrame, text_column: str = 'headline', date_column: str = 'date') -> pd.DataFrame:
        """
        Analyze sentiment for all headlines in a dataframe.

        Args:
            df: Input dataframe
            text_column: Name of the column containing text to analyze
            date_column: Name of the date column

        Returns:
            DataFrame with added sentiment columns
        """
        # Create a copy to avoid modifying the original
        df = df.copy()

        # Convert dates with flexible parsing
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], format='mixed', utc=True)
            df[date_column] = df[date_column].dt.tz_convert('America/New_York').dt.tz_localize(None)

        # Add sentiment scores
        df['vader_sentiment'] = df[text_column].apply(self.get_vader_sentiment)
        df['textblob_sentiment'] = df[text_column].apply(self.get_textblob_sentiment)

        # Add sentiment categories
        df['vader_category'] = pd.cut(
            df['vader_sentiment'],
            bins=[-1, -0.05, 0.05, 1],
            labels=['negative', 'neutral', 'positive']
        )
        df['textblob_category'] = pd.cut(
            df['textblob_sentiment'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['negative', 'neutral', 'positive']
        )

        return df

    def get_daily_sentiment(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Calculate daily average sentiment scores.

        Args:
            df: Input dataframe with sentiment scores
            date_column: Name of the date column

        Returns:
            DataFrame with daily sentiment aggregates
        """
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column], format='mixed', utc=True)
            df[date_column] = df[date_column].dt.tz_convert('America/New_York').dt.tz_localize(None)

        return df.groupby(df[date_column].dt.date).agg({
            'vader_sentiment': 'mean',
            'textblob_sentiment': 'mean',
            'vader_category': lambda x: x.mode()[0],
            'textblob_category': lambda x: x.mode()[0]
        }).reset_index()
