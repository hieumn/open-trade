"""Module for computing technical indicator features from stock price data."""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator features to a stock price DataFrame.

    Computes the following features:
        - SMA_5, SMA_10, SMA_20: Simple moving averages over 5, 10, 20 days.
        - EMA_12, EMA_26: Exponential moving averages over 12 and 26 days.
        - RSI_14: Relative Strength Index over 14 days.
        - MACD: Moving Average Convergence Divergence (EMA_12 - EMA_26).
        - BB_upper, BB_lower: Bollinger Bands (SMA_20 ± 2 * std_20).
        - Returns_1d: 1-day percentage return.
        - Volatility_5d: 5-day rolling standard deviation of daily returns.

    Args:
        df: DataFrame with at least a 'Close' column.

    Returns:
        DataFrame with added feature columns; rows with NaN values are dropped.
    """
    df = df.copy()
    close = df["Close"]

    df["SMA_5"] = close.rolling(5).mean()
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()

    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    std_20 = close.rolling(20).std()
    df["BB_upper"] = df["SMA_20"] + 2 * std_20
    df["BB_lower"] = df["SMA_20"] - 2 * std_20

    df["Returns_1d"] = close.pct_change()
    df["Volatility_5d"] = df["Returns_1d"].rolling(5).std()

    df.dropna(inplace=True)
    return df
