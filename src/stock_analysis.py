"""
Stock data analysis utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
from scipy.stats import pearsonr

class StockAnalyzer:
    def __init__(self):
        """Initialize the stock analyzer."""
        pass

    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol and date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with stock data or None if error
        """
        try:
            # Convert dates to datetime if they're strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).tz_localize('America/New_York')
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date).tz_localize('America/New_York')

            # Add one day to end_date to include it in the range
            end_date = end_date + timedelta(days=1)

            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)

            # Convert index to America/New_York timezone
            df.index = df.index.tz_convert('America/New_York')

            # Calculate returns
            df['Returns'] = df['Close'].pct_change()

            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate percentage returns from price series.

        Args:
            prices: Series of prices

        Returns:
            Series of returns
        """
        return prices.pct_change()

    def calculate_correlation(
        self,
        sentiment_data: pd.DataFrame,
        stock_data: pd.DataFrame,
        sentiment_column: str
    ) -> Tuple[float, float]:
        """
        Calculate correlation between sentiment and returns.

        Args:
            sentiment_data: DataFrame with sentiment scores
            stock_data: DataFrame with stock returns
            sentiment_column: Name of the sentiment column

        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        # Ensure dates are aligned
        merged = pd.merge(
            sentiment_data,
            stock_data['Returns'].to_frame(),
            left_index=True,
            right_index=True,
            how='inner'
        )

        # Calculate correlation
        if len(merged) > 1:
            return stats.pearsonr(
                merged[sentiment_column],
                merged['Returns']
            )
        return (np.nan, np.nan)

    def analyze_sentiment_impact(
        self,
        sentiment_data: pd.DataFrame,
        stock_data: pd.DataFrame,
        sentiment_column: str,
        lags: List[int] = [0, 1, 2, 3, 4, 5]
    ) -> pd.DataFrame:
        """
        Analyze impact of sentiment on returns with different lags.

        Args:
            sentiment_data: DataFrame with sentiment scores
            stock_data: DataFrame with stock returns
            sentiment_column: Name of the sentiment column
            lags: List of lag periods to test

        Returns:
            DataFrame with correlation results for each lag
        """
        results = []

        try:
            # Print initial shapes for debugging
            print("\nInitial data shapes:")
            print(f"Stock data shape: {stock_data.shape}")
            print(f"Sentiment data shape: {sentiment_data.shape}")

            # Convert stock data index to datetime and normalize to date only
            stock_data = stock_data.copy()
            stock_data.index = pd.to_datetime(stock_data.index).normalize()

            # Handle sentiment data based on its structure
            sentiment_data = sentiment_data.copy()

            # If sentiment_data has a MultiIndex, filter for the current stock
            if isinstance(sentiment_data.index, pd.MultiIndex):
                print("\nProcessing MultiIndex sentiment data...")
                if 'stock' in sentiment_data.index.names:
                    # Get the stock symbol
                    stock_symbol = sentiment_data.index.get_level_values('stock').unique()[0]
                    # Filter for the current stock
                    sentiment_data = sentiment_data.xs(stock_symbol, level='stock')
                else:
                    sentiment_data = sentiment_data.reset_index(level=0)

            # Ensure the index is datetime
            sentiment_data.index = pd.to_datetime(sentiment_data.index).normalize()

            print("\nProcessed data shapes:")
            print(f"Stock data shape: {stock_data.shape}")
            print(f"Sentiment data shape: {sentiment_data.shape}")

            for lag in lags:
                try:
                    # Shift returns by lag (negative lag means returns lead sentiment)
                    lagged_returns = stock_data['Returns'].shift(-lag)

                    # Create merged dataset with only the required columns
                    merged_data = pd.DataFrame({
                        'sentiment': sentiment_data[sentiment_column],
                        'returns': lagged_returns
                    })

                    # Drop any rows with NaN values
                    merged_data = merged_data.dropna()

                    print(f"\nLag {lag} - Merged data shape: {merged_data.shape}")

                    if len(merged_data) >= 2:
                        # Calculate correlation using numpy arrays
                        r, p = pearsonr(
                            merged_data['sentiment'].to_numpy(),
                            merged_data['returns'].to_numpy()
                        )
                        results.append({
                            'lag': lag,
                            'correlation': r,
                            'p_value': p,
                            'n_obs': len(merged_data)
                        })
                    else:
                        print(f"Insufficient data points for lag {lag}")
                        results.append({
                            'lag': lag,
                            'correlation': np.nan,
                            'p_value': np.nan,
                            'n_obs': len(merged_data)
                        })

                except Exception as e:
                    print(f"Error in lag {lag}: {str(e)}")
                    results.append({
                        'lag': lag,
                        'correlation': np.nan,
                        'p_value': np.nan,
                        'n_obs': 0
                    })

        except Exception as e:
            print(f"Error in analyze_sentiment_impact: {str(e)}")
            results = [{
                'lag': lag,
                'correlation': np.nan,
                'p_value': np.nan,
                'n_obs': 0
            } for lag in lags]

        return pd.DataFrame(results)
