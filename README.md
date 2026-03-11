# Stock Price Prediction Engine

A machine learning system that predicts whether a stock's price will move **UP or DOWN** the next trading day, along with an approximate closing price. It combines technical indicators from historical OHLCV data, VIX volatility index, and real-time news sentiment analysis powered by FinBERT — all running locally with zero paid API costs.

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │           DATA SOURCES                   │
                        └──────────────────────────────────────────┘

    yfinance (OHLCV)          NewsAPI (Headlines)         yfinance (^VIX)
         │                          │                          │
         ▼                          ▼                          │
  ┌─────────────┐          ┌───────────────┐                   │
  │ Technical   │          │   FinBERT     │                   │
  │ Indicators  │          │  (Sentiment)  │                   │
  │ SMA, EMA,   │          │  ProsusAI/    │                   │
  │ RSI, MACD,  │          │  finbert      │                   │
  │ BB, ATR,    │          │  (local)      │                   │
  │ OBV, Lags   │          └──────┬────────┘                   │
  └──────┬──────┘                 │                            │
         │                        │                            │
         ▼                        ▼                            ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                    COMBINED FEATURE VECTOR                     │
  │   30+ features per trading day including sentiment + VIX       │
  └────────────────────────────┬───────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │   Logistic   │ │   XGBoost    │ │    LSTM      │
     │  Regression  │ │  Classifier  │ │   Neural     │
     │  (baseline)  │ │  + Regressor │ │   Network    │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            ▼                ▼                ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                       PREDICTIONS                              │
  │   Direction: UP / DOWN    |    Next-Day Price    |   Confidence │
  └────────────────────────────────────────────────────────────────┘
```

---

## Models

| Model | Type | Purpose |
|---|---|---|
| **Logistic Regression** | Classification | Baseline — every other model must beat this |
| **XGBoost** | Classification + Regression | Primary model — direction prediction + price estimate |
| **LSTM** | Classification | Deep learning — learns from 60-day sequences of features |

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** FinBERT (`ProsusAI/finbert`) downloads automatically from HuggingFace on first run (~400 MB). After that it runs fully offline.

### 2. Add your API key

Sign up for a free key at [https://newsapi.org](https://newsapi.org), then add it to the `.env` file:

```
NEWS_API_KEY=your_actual_key_here
```

### 3. Build the training dataset

```bash
python features/build_dataset.py
```

This fetches historical data for 10 tickers, computes all indicators and sentiment, and saves `data/cache/training_dataset.csv`.

### 4. Train all models

```bash
python models/train.py
```

Trains Logistic Regression, XGBoost (with GridSearchCV + TimeSeriesSplit), and LSTM. All models and scalers are saved to `models/saved/`.

### 5. Evaluate models

```bash
python models/evaluate.py
```

Prints accuracy, precision, recall, F1, directional accuracy, MAE, RMSE, MAPE, and generates three charts in `models/saved/`:
- `predicted_vs_actual.png` — XGBoost predicted vs actual closing prices
- `feature_importance.png` — top 20 features driving XGBoost decisions
- `model_comparison.png` — side-by-side metric comparison table

### 6. Run a prediction

```bash
python app/main.py AAPL
```

Or simply run `python app/main.py` and enter the ticker when prompted.

### One-command training pipeline

```bash
python run_pipeline.py
```

Runs steps 3 → 4 → 5 in sequence.

---

## Project Structure

```
stock-predictor/
├── .env                         ← API key (user fills this)
├── .env.example                 ← template
├── .gitignore                   ← excludes generated data/models
├── config.yaml                  ← all tunable parameters
├── requirements.txt             ← pinned Python dependencies
├── run_pipeline.py              ← one-command: build → train → evaluate
├── README.md                    ← this file
│
├── data/
│   ├── fetch_price.py           ← OHLCV + VIX from yfinance
│   ├── fetch_news.py            ← headlines from NewsAPI
│   └── cache/                   ← cached CSVs (git-ignored)
│
├── features/
│   ├── technical_indicators.py  ← SMA, EMA, RSI, MACD, BB, ATR, OBV, lags
│   ├── sentiment.py             ← FinBERT sentiment scoring
│   └── build_dataset.py         ← merges everything into training CSV
│
├── models/
│   ├── train.py                 ← trains all 3 models
│   ├── evaluate.py              ← metrics + charts
│   ├── predict.py               ← loads models, returns predictions
│   └── saved/                   ← .pkl, .h5 model files (git-ignored)
│
├── notebooks/
│   └── exploration.ipynb        ← optional data exploration
│
└── app/
    └── main.py                  ← CLI entry point
```

---

## Known Limitations

1. **Stock prices are partially random.** This model targets directional accuracy above 50%, not perfect price prediction.

2. **News sentiment history is limited by the free NewsAPI tier.** Training data uses neutral sentiment (0.0) as a proxy for missing historical news. Only the most recent ~30 days have real sentiment scores.

3. **The model cannot react to brand-new event types it has never seen in training.** VIX and sentiment scores serve as indirect proxies for geopolitical and macro shocks.

4. **Look-ahead bias is deliberately avoided throughout.** All features for day T use only data from day T and earlier (Close lags use T−1 through T−10).

5. **This is not financial advice and should not be used for real trading decisions.**

---
