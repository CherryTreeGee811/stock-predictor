"""
Entry point: accept a ticker symbol and run the full prediction pipeline.

Usage:
    python app/main.py AAPL
    python app/main.py          (will prompt for ticker)

Pipeline:
    1. Fetch last 300 days OHLCV + VIX
    2. Compute all technical indicators
    3. Fetch last 7 days of news headlines
    4. Compute daily sentiment via FinBERT
    5. Build the final feature vector for the most recent trading day
    6. Run XGBoost prediction (direction + price)
    7. Run LSTM prediction (direction, if model exists)
    8. Print clean CLI output
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths — make all sibling packages importable
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.fetch_price import fetch_price_data
from data.fetch_news import fetch_news_headlines, group_headlines_by_date
from features.technical_indicators import compute_indicators
from features.sentiment import compute_daily_sentiment
from models.predict import (
    predict_xgboost,
    get_feature_columns,
    load_lstm_and_predict,
)

_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Sentiment label helper
# ---------------------------------------------------------------------------

def _sentiment_label(score: float) -> str:
    """Return a plain-English label for a sentiment score."""
    if score >= 0.25:
        return "Bullish"
    elif score <= -0.25:
        return "Bearish"
    else:
        return "Neutral"


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

def run_prediction(ticker: str) -> None:
    """Execute the full prediction pipeline for *ticker* and print results."""

    cfg = _load_config()
    ticker = ticker.upper().strip()
    today_str = datetime.today().strftime("%Y-%m-%d")

    print()
    print("=" * 62)
    print(f"  STOCK PRICE PREDICTION ENGINE")
    print(f"  Ticker : {ticker}")
    print(f"  Date   : {today_str}")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Step 1 — Fetch price data (last 300 trading days + VIX)
    # ------------------------------------------------------------------
    print("\n[1/6] Fetching price data…")
    price_df = fetch_price_data(ticker, mode="prediction")
    if price_df.empty:
        print(f"\n  ERROR: Could not fetch price data for '{ticker}'.")
        print("  Please check the ticker symbol and try again.")
        return

    current_close = round(float(price_df["Close"].iloc[-1]), 2)

    # ------------------------------------------------------------------
    # Step 2 — Compute technical indicators
    # ------------------------------------------------------------------
    print("\n[2/6] Computing technical indicators…")
    feat_df = compute_indicators(price_df, drop_na=True)
    if feat_df.empty:
        print("  ERROR: Not enough data to compute indicators.")
        return

    # ------------------------------------------------------------------
    # Step 3 — Fetch news headlines (last 7 days)
    # ------------------------------------------------------------------
    print("\n[3/6] Fetching news headlines…")
    news_lookback = cfg["data"]["news_lookback_days"]
    headlines_list = fetch_news_headlines(ticker, lookback_days=news_lookback)
    by_date = group_headlines_by_date(headlines_list)

    # ------------------------------------------------------------------
    # Step 4 — Compute sentiment
    # ------------------------------------------------------------------
    print("\n[4/6] Running FinBERT sentiment analysis…")
    default_sentiment = cfg["sentiment"]["default_score"]

    # Build sentiment for the dates we have data for
    sentiment_map: dict[str, float] = {}
    for date_str, titles in by_date.items():
        sentiment_map[date_str] = compute_daily_sentiment(titles)

    # Assign sentiment to each row of feat_df (default 0.0 for missing days)
    sentiment_col = pd.Series(default_sentiment, index=feat_df.index,
                              name="Daily_Sentiment")
    for date_str, score in sentiment_map.items():
        try:
            dt = pd.Timestamp(date_str)
            if dt in sentiment_col.index:
                sentiment_col.loc[dt] = score
        except Exception:
            continue
    feat_df["Daily_Sentiment"] = sentiment_col.values

    # Today's sentiment (most recent trading day)
    last_date_str = feat_df.index[-1].strftime("%Y-%m-%d")
    today_sentiment = float(feat_df["Daily_Sentiment"].iloc[-1])

    # ------------------------------------------------------------------
    # Step 5 — Build feature vector
    # ------------------------------------------------------------------
    print("\n[5/6] Building feature vector…")
    feature_cols = get_feature_columns()

    # Make sure all expected columns exist in feat_df
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        print(f"  WARNING: Missing feature columns: {missing}")
        for c in missing:
            feat_df[c] = 0.0

    # The most recent row is the prediction input
    latest_row = feat_df[feature_cols].iloc[[-1]].values  # shape (1, n_features)

    # Full feature matrix for LSTM (needs seq_len rows)
    full_matrix = feat_df[feature_cols].values  # shape (n_days, n_features)

    # ------------------------------------------------------------------
    # Step 6 — XGBoost prediction
    # ------------------------------------------------------------------
    print("\n[6/6] Running predictions…")
    try:
        xgb_result = predict_xgboost(latest_row, current_close=current_close)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    # ------------------------------------------------------------------
    # Step 7 — LSTM prediction (optional)
    # ------------------------------------------------------------------
    lstm_result = load_lstm_and_predict(full_matrix)

    # ------------------------------------------------------------------
    # Step 8 — Print results
    # ------------------------------------------------------------------
    _print_results(
        ticker=ticker,
        today=today_str,
        current_close=current_close,
        xgb=xgb_result,
        lstm=lstm_result,
        sentiment_score=today_sentiment,
        headlines_count=len(headlines_list),
    )


# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------

def _print_results(ticker, today, current_close, xgb, lstm,
                   sentiment_score, headlines_count):
    """Print a clean, readable CLI summary."""

    sent_label = _sentiment_label(sentiment_score)
    arrow_xgb = "▲" if xgb["direction"] == "UP" else "▼"

    print()
    print("=" * 62)
    print(f"  PREDICTION RESULTS — {ticker}")
    print(f"  Generated: {today}")
    print("=" * 62)

    print(f"""
  Current Close Price    : ${current_close:,.2f}

  ── XGBoost Prediction ────────────────────────────────
     Direction            : {arrow_xgb} {xgb['direction']}
     Confidence           : {xgb['confidence']:.1f}%
     Predicted Next Close : ${xgb['price']:,.2f}""")

    if lstm:
        arrow_lstm = "▲" if lstm["direction"] == "UP" else "▼"
        print(f"""
  ── LSTM Prediction ───────────────────────────────────
     Direction            : {arrow_lstm} {lstm['direction']}
     Confidence           : {lstm['confidence']:.1f}%""")
    else:
        print("""
  ── LSTM Prediction ───────────────────────────────────
     (LSTM model not available — skipped)""")

    print(f"""
  ── Sentiment Analysis ────────────────────────────────
     Today's Score        : {sentiment_score:+.4f}
     Interpretation       : {sent_label}
     Headlines Analysed   : {headlines_count}""")

    print(f"""
  ── Top Features Driving Prediction ───────────────────""")
    for rank, (name, imp) in enumerate(xgb["top_features"], 1):
        print(f"     {rank}. {name:<25} (importance: {imp:.4f})")

    print(f"""
  ── Disclaimer ────────────────────────────────────────
     This is NOT financial advice. Predictions are based
     on historical patterns and may not reflect future
     market behaviour. Do not use for real trading.
""")
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = input("Enter stock ticker symbol (e.g. AAPL): ").strip()
        if not ticker:
            print("No ticker provided. Exiting.")
            return

    run_prediction(ticker)


if __name__ == "__main__":
    main()
