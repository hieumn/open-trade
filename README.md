# open-trade

A Python tool to predict the **next-day closing price** of any stock ticker for daily trading decisions.

## Features

- Fetches historical OHLCV data via [yfinance](https://github.com/ranaroussi/yfinance)
- Computes technical indicators: SMA, EMA, MACD, RSI, Bollinger Bands, returns & volatility
- Trains a Ridge regression model on those indicators
- Predicts the next-day closing price from the CLI

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python predict.py AAPL
python predict.py TSLA --period 1y
python predict.py MSFT --period 2y --alpha 0.5
```

Example output:

```
Fetching data for AAPL (2y)...
Computing technical indicators...
Training prediction model...

Results for AAPL:
  Last close:       $189.30
  Predicted close:  $191.05
  Expected change:  +1.75 (+0.92%)
```

### Arguments

| Argument   | Description                                         | Default |
|------------|-----------------------------------------------------|---------|
| `ticker`   | Stock ticker symbol (e.g. `AAPL`, `TSLA`, `MSFT`)  | —       |
| `--period` | Historical data period for training (`1y`, `2y` …) | `2y`    |
| `--alpha`  | Ridge regression regularisation strength            | `1.0`   |

## Project Structure

```
open-trade/
├── predict.py          # CLI entry point
├── requirements.txt
├── src/
│   ├── data.py         # Data fetching (yfinance)
│   ├── features.py     # Technical indicator computation
│   └── model.py        # Ridge regression predictor
└── tests/
    ├── test_features.py
    ├── test_model.py
    └── test_predict.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

