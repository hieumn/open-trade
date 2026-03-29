"""Module for fetching and preprocessing stock price data."""

import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        period: Data period to download. Valid periods: 1d, 5d, 1mo, 3mo,
                6mo, 1y, 2y, 5y, 10y, ytd, max.
        interval: Data interval. Valid intervals: 1m, 2m, 5m, 15m, 30m,
                  60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume.

    Raises:
        ValueError: If no data is returned for the given ticker or period.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}' with period '{period}'.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df
