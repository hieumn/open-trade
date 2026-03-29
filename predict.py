#!/usr/bin/env python3
"""CLI entry point for stock price prediction."""

import argparse
import sys

from src.data import fetch_stock_data
from src.features import add_features
from src.model import StockPricePredictor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict the next-day closing price of a stock.",
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol (e.g. AAPL, TSLA, MSFT).",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Historical data period used for training (default: 2y).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regression regularisation strength (default: 1.0).",
    )
    return parser.parse_args(argv)


def run(argv=None):
    args = parse_args(argv)

    print(f"Fetching data for {args.ticker.upper()} ({args.period})...")
    df = fetch_stock_data(args.ticker, period=args.period)

    print("Computing technical indicators...")
    df = add_features(df)

    print("Training prediction model...")
    predictor = StockPricePredictor(alpha=args.alpha)
    predictor.fit(df)

    prediction = predictor.predict_next(df)
    last_close = df["Close"].iloc[-1]
    change = prediction - last_close
    pct_change = (change / last_close) * 100

    print(f"\nResults for {args.ticker.upper()}:")
    print(f"  Last close:       ${last_close:.2f}")
    print(f"  Predicted close:  ${prediction:.2f}")
    print(f"  Expected change:  {change:+.2f} ({pct_change:+.2f}%)")


if __name__ == "__main__":
    run(sys.argv[1:])
