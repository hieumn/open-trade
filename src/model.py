"""Module for training and using a stock price prediction model."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "Open",
    "High",
    "Low",
    "Volume",
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "EMA_12",
    "EMA_26",
    "MACD",
    "RSI_14",
    "BB_upper",
    "BB_lower",
    "Returns_1d",
    "Volatility_5d",
]


class StockPricePredictor:
    """Predicts the next-day closing price of a stock.

    Uses a Ridge regression model trained on technical indicators derived
    from historical OHLCV data.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialise the predictor.

        Args:
            alpha: Regularisation strength for Ridge regression.
        """
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Raise ValueError if any required feature columns are missing.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If any FEATURE_COLS are absent from *df*.
        """
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

    def _prepare(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix X and target vector y from *df*.

        The target for each row is the *next* day's closing price.

        Args:
            df: DataFrame that must contain all FEATURE_COLS and 'Close'.

        Returns:
            Tuple of (X, y) arrays, aligned so that X[i] predicts y[i].
        """
        self._validate_columns(df)

        X = df[FEATURE_COLS].values[:-1]
        y = df["Close"].values[1:]
        return X, y

    def fit(self, df: pd.DataFrame) -> "StockPricePredictor":
        """Train the model on historical data.

        Args:
            df: DataFrame with feature columns and 'Close'.

        Returns:
            self (to allow method chaining).
        """
        X, y = self._prepare(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict_next(self, df: pd.DataFrame) -> float:
        """Predict the next-day closing price using the most recent row.

        Args:
            df: DataFrame with feature columns (most recent row is used).

        Returns:
            Predicted next-day closing price.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        self._validate_columns(df)

        X_latest = df[FEATURE_COLS].values[-1].reshape(1, -1)
        X_scaled = self.scaler.transform(X_latest)
        return float(self.model.predict(X_scaled)[0])
