"""Unit tests for predict.py CLI."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from predict import parse_args, run
from src.features import add_features


def _make_featured_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
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
    return raw


def test_parse_args_defaults():
    args = parse_args(["AAPL"])
    assert args.ticker == "AAPL"
    assert args.period == "2y"
    assert args.alpha == 1.0


def test_parse_args_custom():
    args = parse_args(["TSLA", "--period", "1y", "--alpha", "0.5"])
    assert args.ticker == "TSLA"
    assert args.period == "1y"
    assert args.alpha == 0.5


def test_parse_args_missing_ticker(capsys):
    with pytest.raises(SystemExit):
        parse_args([])


def test_run_prints_prediction(capsys):
    """Integration test: run() should print a predicted price without network calls."""
    raw = _make_featured_df()

    with patch("predict.fetch_stock_data", return_value=raw):
        run(["AAPL", "--period", "1y"])

    captured = capsys.readouterr().out
    assert "Predicted close:" in captured
    assert "AAPL" in captured
    assert "Last close:" in captured
