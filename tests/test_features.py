"""Unit tests for src/features.py."""

import numpy as np
import pandas as pd
import pytest

from src.features import add_features


def _make_df(n: int = 60) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame with *n* rows."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close * rng.uniform(0.99, 1.01, n)
    high = close * rng.uniform(1.00, 1.02, n)
    low = close * rng.uniform(0.98, 1.00, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def test_add_features_returns_dataframe():
    df = add_features(_make_df())
    assert isinstance(df, pd.DataFrame)


def test_add_features_expected_columns():
    df = add_features(_make_df())
    expected = {
        "SMA_5", "SMA_10", "SMA_20",
        "EMA_12", "EMA_26", "MACD",
        "RSI_14", "BB_upper", "BB_lower",
        "Returns_1d", "Volatility_5d",
    }
    assert expected.issubset(set(df.columns))


def test_add_features_drops_nan_rows():
    df = add_features(_make_df())
    assert not df.isnull().any().any()


def test_add_features_fewer_rows_than_input():
    raw = _make_df(60)
    df = add_features(raw)
    assert len(df) < len(raw)


def test_add_features_rsi_bounds():
    df = add_features(_make_df(100))
    assert (df["RSI_14"] >= 0).all() and (df["RSI_14"] <= 100).all()


def test_add_features_bollinger_band_order():
    df = add_features(_make_df())
    assert (df["BB_upper"] >= df["SMA_20"]).all()
    assert (df["BB_lower"] <= df["SMA_20"]).all()


def test_add_features_does_not_mutate_input():
    raw = _make_df()
    original_cols = list(raw.columns)
    add_features(raw)
    assert list(raw.columns) == original_cols
