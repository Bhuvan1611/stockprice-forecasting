# The Ledger — Stock Price Predictor

A small multi-page Flask site that downloads historical price data with
`yfinance`, backtests an LSTM model, forecasts future prices, and shows
interactive Plotly charts plus RSI/MACD/Bollinger Band indicators — now
with a real navigation structure instead of a single form-and-results page.

## What's new in this version

- **Multi-page site**: Predict, Compare, Watchlist, and About, sharing one
  layout and navigation instead of a single page.
- **Compare tool** (`/compare`): run the backtest + forecast for two tickers
  at once and see their relative performance on one normalized chart, plus
  metrics side by side.
- **Watchlist** (`/watchlist`): save tickers in your session (no account
  needed), see a live-ish quote next to each, jump straight to a prediction,
  or remove one.
- **News panel**: latest headlines for the ticker you just predicted, pulled
  from Yahoo Finance.
- **Ticker tape**: a scrolling strip of a few major indices/stocks at the
  top of every page, refreshed via a small `/api/ticker-tape` endpoint so it
  never blocks page load.
- **Dark / light mode**, remembered per browser via `localStorage`.
- **Popular ticker chips** on the predictor so you don't have to remember
  exact Yahoo Finance symbols.
- **Loading overlay** on submit so long-running predictions (yfinance
  download + model inference) don't look like a frozen page.
- Redesigned visual language: a "ledger / exchange board" look (serif
  headings, monospace figures, hairline rules) instead of default
  Bootstrap-style cards.

Everything from the previous fixed version is still in place: a real string
`SECRET_KEY`, flattened `yfinance` MultiIndex columns, a scaler fit once on
training data only, sanitized filenames, per-request unique export files,
and a `today`-based end date.

## 1. Project layout

```
stock_predictor/
├── app.py                    # Flask app — routes, model pipeline, chart builders
├── stock_dl_model.h5         # your trained LSTM model
├── requirements.txt
├── Dockerfile
├── templates/
│   ├── base.html              # shared shell: nav, ticker tape, theme toggle
│   ├── index.html              # predictor form + results
│   ├── compare.html            # two-ticker comparison
│   ├── watchlist.html          # saved tickers
│   └── about.html              # methodology / how it works
├── static/
│   ├── css/style.css           # design system (light + dark themes)
│   ├── js/app.js               # theme toggle, ticker tape, watchlist AJAX
│   └── exports/                # generated CSVs (created automatically)
└── cache/                      # cached price downloads (created automatically)
```

## 2. Local setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure stock_dl_model.h5 is in the project root (same folder as app.py)

# 4. Run the app
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

To enable Flask's debug/reload mode during development:

```bash
export FLASK_DEBUG=1      # Windows (PowerShell): $env:FLASK_DEBUG="1"
python app.py
```

## 3. Using the site

**Predict** (`/`)
1. Enter a ticker using the Yahoo Finance format, or click a popular-ticker chip:
   - Indian NSE stocks: `POWERGRID.NS`, `RELIANCE.NS`
   - US stocks: `AAPL`, `MSFT`
   - Indices: `^NSEI` (Nifty 50), `^GSPC` (S&P 500)
2. Set how many future trading days to forecast (1–90).
3. Click **Run prediction**. You'll get RMSE/MAE/MAPE/R² on a held-out
   backtest, candlestick / EMA / RSI / MACD / Bollinger charts, a
   prediction-vs-actual-plus-forecast chart, a downloadable forecast and
   dataset CSV, and recent headlines for that ticker.
4. Click **+ Watchlist** to save the ticker.

**Compare** (`/compare`) — run the same pipeline on two tickers and see
their normalized % change on one chart, with metrics side by side.

**Watchlist** (`/watchlist`) — see saved tickers with a quick quote, jump to
a full prediction, or remove one. Stored in your session only; there's no
account system.

**About** (`/about`) — a plain-language explanation of the model, the
accuracy metrics, and each technical indicator.

## 4. Retraining the model

The original notebook (`Stock_Price_Prediction_.ipynb`) still works for
retraining. Two things worth fixing there to match `app.py`:

```python
# Fit the scaler on the training split only, then just transform the rest:
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_training)
input_data = scaler.transform(final_df)   # not fit_transform

# Use scaler.inverse_transform(...) to get back to real prices instead of a
# hardcoded "scaler_factor = 1 / 0.0035166" (that number only applies to the
# exact dataset it was computed from, and silently gives wrong numbers on
# any other ticker or date range).
```

After retraining, overwrite `stock_dl_model.h5` in the project root.

## 5. Deploying with Docker

```bash
docker build -t stock-predictor .
docker run -p 5000:5000 stock-predictor
```

This runs the app behind `gunicorn` (2 workers) instead of Flask's dev
server. For a real deployment, put nginx (or another reverse proxy) in front
of it and set `FLASK_SECRET_KEY` to a real secret via environment variable:

```bash
docker run -p 5000:5000 -e FLASK_SECRET_KEY="a-long-random-string" stock-predictor
```

## 6. Notes / limitations

- The model is univariate (trained on `Close` only) and forecasts are
  generated by iteratively feeding predictions back in — accuracy degrades
  the further out you forecast. Treat long forecasts as illustrative, not
  reliable.
- Cached price data expires after 6 hours (`CACHE_TTL_SECONDS` in `app.py`);
  quick quotes (ticker tape, watchlist) expire after 5 minutes
  (`QUOTE_TTL_SECONDS`).
- The watchlist lives in the Flask session (a signed cookie plus
  server-side filesystem store), not a database — clearing cookies clears it.
- This is an educational project; nothing it outputs is financial advice.
