"""
Run FinBERT sentiment analysis on news headlines.

Model: ProsusAI/finbert (loaded once, runs locally).
Input:  list of headline strings for a single day.
Output: one aggregated daily sentiment score in [-1, 1].

Formula:
    raw = sum(positive_scores) - sum(negative_scores)
    daily_sentiment = raw / max(abs(raw), 1)   # normalise to [-1, 1]

If no headlines are provided, returns the default score (0.0 = neutral).
"""

import os

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Lazy-loaded singleton for the FinBERT model + tokenizer
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None


def _ensure_model_loaded() -> None:
    """Download / load FinBERT once and cache in module-level globals."""
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    cfg = _load_config()
    model_name = cfg["sentiment"]["model_name"]  # "ProsusAI/finbert"

    print(f"[sentiment] Loading FinBERT model ({model_name})…")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSequenceClassification.from_pretrained(model_name)
    _model.eval()  # inference mode
    print("[sentiment] FinBERT ready.")


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

# FinBERT label mapping (model output indices → label names)
_LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}


def _score_headlines(headlines: list[str]) -> list[dict]:
    """
    Run FinBERT on a batch of headlines.

    Returns a list of dicts, one per headline:
        {"label": "positive"|"negative"|"neutral",
         "score": float}
    """
    _ensure_model_loaded()

    inputs = _tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    results = []
    for row in probs:
        idx = torch.argmax(row).item()
        results.append({
            "label": _LABEL_MAP[idx],
            "score": row[idx].item(),
        })
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_daily_sentiment(headlines: list[str]) -> float:
    """
    Compute one aggregated sentiment score for a list of headlines.

    Parameters
    ----------
    headlines : list[str]
        Raw headline strings for a single day.

    Returns
    -------
    float
        Sentiment score in the range [-1.0, 1.0].
        0.0 is returned when *headlines* is empty.
    """
    cfg = _load_config()
    default = cfg["sentiment"]["default_score"]       # 0.0
    max_per_day = cfg["sentiment"]["max_headlines_per_day"]  # 20

    if not headlines:
        return default

    # Cap headlines to avoid excessive compute
    if len(headlines) > max_per_day:
        headlines = headlines[:max_per_day]

    scored = _score_headlines(headlines)

    pos_sum = sum(s["score"] for s in scored if s["label"] == "positive")
    neg_sum = sum(s["score"] for s in scored if s["label"] == "negative")

    raw = pos_sum - neg_sum

    # Normalise to [-1, 1]
    # The theoretical max of |raw| equals len(headlines) (every headline
    # scores 1.0 for one class).  Dividing by the count would give a
    # "per-headline average", which naturally falls in [-1, 1].
    n = len(scored)
    daily_sentiment = raw / n if n > 0 else 0.0

    # Clamp just in case of floating-point edge cases
    daily_sentiment = max(-1.0, min(1.0, daily_sentiment))

    return round(daily_sentiment, 6)


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_headlines = [
        "Apple stock surges to all-time high after record earnings",
        "Investors worry about Apple's slowing growth in China",
        "Apple announces new AI features for iPhone lineup",
        "Tech stocks fall amid rising interest rate fears",
        "Apple beats revenue expectations for Q4 2025",
    ]

    print("Testing FinBERT sentiment on sample headlines:\n")
    for h in test_headlines:
        score = compute_daily_sentiment([h])
        print(f"  [{score:+.4f}]  {h}")

    combined = compute_daily_sentiment(test_headlines)
    print(f"\nAggregated daily score: {combined:+.4f}")
