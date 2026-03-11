"""
Merge price features and sentiment into a single training DataFrame.

Orchestrates the full dataset build:
  1. Loop through all training_tickers from config.yaml
  2. For each ticker: fetch price data (full history), compute indicators
  3. Fetch available news (last ~30 days due to free-tier limit),
     compute sentiment for those days; backfill everything else with 0.0
  4. Add Day_of_Week, Month columns (already from indicators)
  5. Create target columns: Direction and Next_Close
  6. Drop the last row per ticker (no target available)
  7. Concatenate all tickers into one DataFrame
  8. Save to data/cache/training_dataset.csv
"""

import os
import sys

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "cache")

# Make sure project root is on sys.path so sibling packages import cleanly
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.fetch_price import fetch_price_data
from data.fetch_news import fetch_news_headlines, group_headlines_by_date
from features.technical_indicators import compute_indicators
from features.sentiment import compute_daily_sentiment


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sentiment_column(df: pd.DataFrame, ticker: str,
                            news_lookback: int,
                            default_score: float) -> pd.Series:
    """
    Build a Daily_Sentiment Series aligned to *df*'s DatetimeIndex.

    Only the most recent *news_lookback* days can have real scores
    (NewsAPI free-tier limit). All older dates get *default_score*.
    """
    # Fetch whatever headlines NewsAPI can provide
    headlines_list = fetch_news_headlines(ticker, lookback_days=news_lookback)
    by_date = group_headlines_by_date(headlines_list)  # {date_str: [headlines]}

    # Score each day that has headlines
    daily_scores: dict[str, float] = {}
    for date_str, titles in by_date.items():
        daily_scores[date_str] = compute_daily_sentiment(titles)

    # Map scores onto the DataFrame index
    sentiment = pd.Series(default_score, index=df.index, name="Daily_Sentiment")
    for date_str, score in daily_scores.items():
        try:
            dt = pd.Timestamp(date_str)
            if dt in sentiment.index:
                sentiment.loc[dt] = score
        except Exception:
            continue

    return sentiment


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add target columns:
      - Direction: 1 if next day Close > today Close, else 0
      - Next_Close: actual next-day closing price
    Then drop the last row (no target available).
    """
    df = df.copy()
    df["Next_Close"] = df["Close"].shift(-1)
    df["Direction"] = (df["Next_Close"] > df["Close"]).astype(int)

    # Percentage change target — scale-independent across tickers
    df["Pct_Change"] = (df["Next_Close"] - df["Close"]) / df["Close"] * 100

    # Drop the final row where targets are NaN
    df = df.iloc[:-1]
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_training_dataset(save: bool = True) -> pd.DataFrame:
    """
    Build the combined training dataset for all tickers in config.

    Parameters
    ----------
    save : bool
        If True, save the result to data/cache/training_dataset.csv.

    Returns
    -------
    pd.DataFrame
        Columns: all features + Daily_Sentiment + Direction + Next_Close +
                 a "Ticker" column identifying the source ticker.
    """
    cfg = _load_config()
    tickers = cfg["data"]["training_tickers"]
    default_sentiment = cfg["sentiment"]["default_score"]
    # NewsAPI free tier: ~30 days max history
    news_lookback = 30

    all_frames: list[pd.DataFrame] = []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'='*60}")
        print(f"[build_dataset] Processing {ticker}  ({i}/{len(tickers)})")
        print(f"{'='*60}")

        # 1. Fetch full-history price data + VIX
        price_df = fetch_price_data(ticker, mode="training")
        if price_df.empty:
            print(f"[build_dataset] Skipping {ticker} — no price data.")
            continue

        # 2. Compute all technical indicators
        feat_df = compute_indicators(price_df, drop_na=True)
        if feat_df.empty:
            print(f"[build_dataset] Skipping {ticker} — no rows after "
                  "indicator warm-up.")
            continue

        # 3. Sentiment column
        print(f"[build_dataset] Building sentiment for {ticker}…")
        sentiment = _build_sentiment_column(
            feat_df, ticker,
            news_lookback=news_lookback,
            default_score=default_sentiment,
        )
        feat_df["Daily_Sentiment"] = sentiment.values

        # 4. Target columns + drop last row
        feat_df = _add_targets(feat_df)

        # 5. Tag with ticker symbol
        feat_df["Ticker"] = ticker

        print(f"[build_dataset] {ticker}: {len(feat_df)} rows ready.")
        all_frames.append(feat_df)

    if not all_frames:
        print("[build_dataset] ERROR: No data collected for any ticker.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, axis=0)
    combined.sort_index(inplace=True)

    print(f"\n{'='*60}")
    print(f"[build_dataset] Combined dataset: {combined.shape[0]} rows, "
          f"{combined.shape[1]} columns")
    print(f"  Tickers: {combined['Ticker'].unique().tolist()}")
    print(f"  Date range: {combined.index.min()} → {combined.index.max()}")
    print(f"  Direction balance: "
          f"{combined['Direction'].value_counts().to_dict()}")
    print(f"{'='*60}")

    # Save
    if save:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        out_path = os.path.join(_CACHE_DIR, "training_dataset.csv")
        combined.to_csv(out_path)
        print(f"[build_dataset] Saved → {out_path}")

    return combined


# ---------------------------------------------------------------------------
# Quick CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = build_training_dataset(save=True)
    if not dataset.empty:
        print(f"\nFinal shape : {dataset.shape}")
        print(f"Columns     : {list(dataset.columns)}")
        print(f"\nSample (last 3 rows):\n{dataset.tail(3)}")
    else:
        print("Dataset is empty.")
