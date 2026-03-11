"""
Fetch financial news headlines from NewsAPI.

Uses the /v2/everything endpoint with query "{TICKER} stock".
Returns headlines grouped by date as a list of (date_str, headline) tuples.
"""

import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import requests
import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths & env
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")

# Load .env so NEWS_API_KEY is available via os.environ
load_dotenv(_ENV_PATH)

_NEWSAPI_BASE = "https://newsapi.org/v2/everything"


def _load_config() -> dict:
    """Load config.yaml and return the full dict."""
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news_headlines(
    ticker: str,
    lookback_days: int | None = None,
    max_headlines: int | None = None,
) -> list[tuple[str, str]]:
    """
    Fetch recent news headlines for a stock ticker from NewsAPI.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    lookback_days : int or None
        Number of past days to search. Defaults to config value (7).
    max_headlines : int or None
        Cap on total headlines returned. Defaults to None (return all).

    Returns
    -------
    list[tuple[str, str]]
        Each element is (date_str "YYYY-MM-DD", headline_text).
        Sorted by date descending (most recent first).
    """
    cfg = _load_config()
    api_key = os.environ.get("NEWS_API_KEY", "")

    if not api_key or api_key == "your_newsapi_key_here":
        print("[fetch_news] WARNING: NEWS_API_KEY is not set in .env — "
              "returning empty headlines.")
        return []

    if lookback_days is None:
        lookback_days = cfg["data"]["news_lookback_days"]  # default 7

    # Date range
    to_date = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    query = f"{ticker} stock"

    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,        # max allowed per page on free tier
        "apiKey": api_key,
    }

    # --- Make the request ---
    try:
        print(f"[fetch_news] Fetching headlines for \"{query}\" "
              f"({from_date} → {to_date})...")
        resp = requests.get(_NEWSAPI_BASE, params=params, timeout=15)
    except requests.RequestException as exc:
        print(f"[fetch_news] ERROR: Request failed — {exc}")
        return []

    # --- Handle HTTP / API errors ---
    if resp.status_code == 401:
        print("[fetch_news] ERROR: Invalid API key (401 Unauthorized).")
        return []
    if resp.status_code == 429:
        print("[fetch_news] ERROR: Rate limit exceeded (429). "
              "Free tier allows 100 requests/day.")
        return []
    if resp.status_code != 200:
        print(f"[fetch_news] ERROR: NewsAPI returned status {resp.status_code}.")
        try:
            body = resp.json()
            print(f"  Message: {body.get('message', 'N/A')}")
        except Exception:
            pass
        return []

    data = resp.json()

    if data.get("status") != "ok":
        print(f"[fetch_news] ERROR: API status = {data.get('status')}; "
              f"message = {data.get('message', 'N/A')}")
        return []

    articles = data.get("articles", [])
    if not articles:
        print(f"[fetch_news] No articles found for \"{query}\".")
        return []

    # --- Parse into (date, headline) tuples ---
    results: list[tuple[str, str]] = []
    for article in articles:
        title = (article.get("title") or "").strip()
        published = article.get("publishedAt", "")

        # Skip removed / empty headlines
        if not title or title.lower() == "[removed]":
            continue

        # Extract date portion (YYYY-MM-DD) from ISO timestamp
        try:
            date_str = published[:10]
            # Validate it parses correctly
            datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            continue

        results.append((date_str, title))

    # Sort by date descending
    results.sort(key=lambda x: x[0], reverse=True)

    # Optional cap
    if max_headlines and len(results) > max_headlines:
        results = results[:max_headlines]

    print(f"[fetch_news] Retrieved {len(results)} headlines across "
          f"{len(set(d for d, _ in results))} days.")
    return results


def group_headlines_by_date(
    headlines: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """
    Utility: group a flat list of (date, headline) into a dict keyed by date.

    Returns
    -------
    dict[str, list[str]]
        {date_str: [headline1, headline2, ...]}  sorted by date descending.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for date_str, headline in headlines:
        grouped[date_str].append(headline)

    # Return as a regular dict sorted by date descending
    return dict(sorted(grouped.items(), reverse=True))


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Usage: python fetch_news.py AAPL
    #        python fetch_news.py NVDA 14        (override lookback days)
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    days_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    headlines = fetch_news_headlines(ticker_arg, lookback_days=days_arg)

    if headlines:
        by_date = group_headlines_by_date(headlines)
        for date, titles in by_date.items():
            print(f"\n=== {date} ({len(titles)} headlines) ===")
            for t in titles[:5]:  # show up to 5 per day
                print(f"  • {t}")
            if len(titles) > 5:
                print(f"  ... and {len(titles) - 5} more")
    else:
        print("No headlines returned.")
