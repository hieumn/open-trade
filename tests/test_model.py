"""Unit tests for src/model.py."""

import numpy as np
import pandas as pd
import pytest

from src.features import add_features
from src.model import FEATURE_COLS, StockPricePredictor


def _make_featured_df(n: int = 80) -> pd.DataFrame:
    """Create a DataFrame with feature columns already computed."""
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close * rng.uniform(0.99, 1.01, n)
    high = close * rng.uniform(1.00, 1.02, n)
    low = close * rng.uniform(0.98, 1.00, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    raw = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )
    return add_features(raw)


def test_predictor_fit_returns_self():
    df = _make_featured_df()
    predictor = StockPricePredictor()
    result = predictor.fit(df)
    assert result is predictor


def test_predictor_predict_next_returns_float():
    df = _make_featured_df()
    predictor = StockPricePredictor()
    predictor.fit(df)
    prediction = predictor.predict_next(df)
    assert isinstance(prediction, float)


def test_predictor_prediction_is_positive():
    df = _make_featured_df()
    predictor = StockPricePredictor()
    predictor.fit(df)
    assert predictor.predict_next(df) > 0


def test_predictor_raises_before_fit():
    df = _make_featured_df()
    predictor = StockPricePredictor()
    with pytest.raises(RuntimeError, match="not been fitted"):
        predictor.predict_next(df)


def test_predictor_raises_on_missing_columns():
    df = _make_featured_df().drop(columns=["RSI_14"])
    predictor = StockPricePredictor()
    with pytest.raises(ValueError, match="missing required columns"):
        predictor.fit(df)


def test_predictor_different_alpha_gives_different_result():
    df = _make_featured_df()
    p1 = StockPricePredictor(alpha=0.01).fit(df).predict_next(df)
    p2 = StockPricePredictor(alpha=1000.0).fit(df).predict_next(df)
    assert p1 != p2
