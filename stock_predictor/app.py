"""
The Ledger — Stock Price Predictor
Flask app that downloads historical price data with yfinance, backtests an
LSTM model, forecasts future prices, and renders an interactive multi-page
website: a predictor, a two-ticker comparison tool, a personal watchlist,
and a live-ish ticker tape.

See README.md for the full feature list and setup instructions.
"""

import os
import time
import uuid
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from werkzeug.utils import secure_filename

import yfinance as yf
from flask import Flask, render_template, request, send_from_directory, session, jsonify
from flask_session import Session
from keras.models import load_model  # type: ignore

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
EXPORT_DIR = os.path.join(STATIC_DIR, "exports")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CACHE_TTL_SECONDS = 6 * 60 * 60  # re-download OHLCV after 6 hours
QUOTE_TTL_SECONDS = 5 * 60  # ticker-tape / quick-quote cache
MODEL_PATH = os.path.join(BASE_DIR, "stock_dl_model.h5")
WINDOW = 100  # lookback window the LSTM was trained on
MAX_WATCHLIST = 20

POPULAR_TICKERS = [
    {"symbol": "^NSEI", "label": "Nifty 50"},
    {"symbol": "^BSESN", "label": "Sensex"},
    {"symbol": "RELIANCE.NS", "label": "Reliance"},
    {"symbol": "TCS.NS", "label": "TCS"},
    {"symbol": "AAPL", "label": "Apple"},
    {"symbol": "MSFT", "label": "Microsoft"},
    {"symbol": "^GSPC", "label": "S&P 500"},
]

TAPE_TICKERS = ["^NSEI", "^BSESN", "^GSPC", "^DJI", "AAPL", "MSFT", "RELIANCE.NS"]

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "final_project_g3")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(BASE_DIR, "flask_session")
Session(app)

# Load the model once at startup. If this fails we want a clear error, not a
# crash the first time someone submits the form.
try:
    model = load_model(MODEL_PATH)
    MODEL_LOAD_ERROR = None
except Exception as exc:  # noqa: BLE001
    model = None
    MODEL_LOAD_ERROR = str(exc)

_quote_cache: dict[str, tuple[float, dict]] = {}
_news_cache: dict[str, tuple[float, list]] = {}


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def get_stock_data(ticker: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Download (or load from cache) OHLCV data for a ticker, with flattened columns."""
    safe_ticker = secure_filename(ticker) or "unknown"
    cache_path = os.path.join(CACHE_DIR, f"{safe_ticker}.csv")

    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < CACHE_TTL_SECONDS:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty:
                return df

    df = yf.download(ticker, start=start, end=end, progress=False)

    # yfinance can return MultiIndex columns (e.g. ('Close', 'POWERGRID.NS'))
    # even for a single ticker on newer versions - flatten to plain names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not df.empty:
        df.to_csv(cache_path)

    return df


def get_quick_quote(ticker: str) -> dict | None:
    """Small, short-lived cache of last price + % change, for the ticker tape
    and watchlist snapshot rows. Cheap enough to call for a handful of symbols
    on every page load without hammering yfinance."""
    now = time.time()
    cached = _quote_cache.get(ticker)
    if cached and now - cached[0] < QUOTE_TTL_SECONDS:
        return cached[1]

    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if hist.empty or len(hist) < 2:
            return None
        last_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])
        change_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0.0
        quote = {
            "symbol": ticker,
            "price": round(last_close, 2),
            "change_pct": round(change_pct, 2),
            "up": change_pct >= 0,
        }
        _quote_cache[ticker] = (now, quote)
        return quote
    except Exception:  # noqa: BLE001
        return None


def get_news(ticker: str, limit: int = 6) -> list:
    """Latest headlines for a ticker via yfinance, short-lived in-memory cache."""
    now = time.time()
    cached = _news_cache.get(ticker)
    if cached and now - cached[0] < QUOTE_TTL_SECONDS:
        return cached[1]

    items = []
    try:
        raw = yf.Ticker(ticker).news or []
        for entry in raw[:limit]:
            content = entry.get("content", entry)  # yfinance has changed this shape across versions
            title = content.get("title") or entry.get("title")
            link = (
                (content.get("canonicalUrl") or {}).get("url")
                or (content.get("clickThroughUrl") or {}).get("url")
                or entry.get("link")
            )
            publisher = (content.get("provider") or {}).get("displayName") or entry.get("publisher")
            if title and link:
                items.append({"title": title, "link": link, "publisher": publisher or "Source"})
    except Exception:  # noqa: BLE001
        items = []

    _news_cache[ticker] = (now, items)
    return items


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, period: int = 20, num_std: int = 2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


# --------------------------------------------------------------------------- #
# Modeling helpers
# --------------------------------------------------------------------------- #
def run_backtest(df: pd.DataFrame):
    """Split into train/test, scale correctly (fit on train only), and get
    the model's predictions on the held-out test portion, in real price units."""
    close = df[["Close"]].dropna()

    split_idx = int(len(close) * 0.70)
    data_training = close.iloc[:split_idx]
    data_testing = close.iloc[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_training)  # fit ONCE, on training data only

    past_100_days = data_training.tail(WINDOW)
    final_df = pd.concat([past_100_days, data_testing])
    input_data = scaler.transform(final_df)  # transform only - no re-fit

    x_test, y_test = [], []
    for i in range(WINDOW, input_data.shape[0]):
        x_test.append(input_data[i - WINDOW:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_pred_scaled = model.predict(x_test, verbose=0)

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred_scaled).flatten()

    test_dates = data_testing.index[: len(y_test_actual)]

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))),
        "mae": float(mean_absolute_error(y_test_actual, y_pred_actual)),
        "mape": float(
            np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
        ),
        "r2": float(r2_score(y_test_actual, y_pred_actual)),
    }

    return test_dates, y_test_actual, y_pred_actual, scaler, metrics


def forecast_future(df: pd.DataFrame, scaler: MinMaxScaler, days: int):
    """Iteratively predict the next `days` trading days using the last WINDOW
    days of actual closes, scaled with the SAME scaler used for training."""
    close = df[["Close"]].dropna()
    last_window = close.tail(WINDOW).values  # shape (WINDOW, 1)
    scaled_window = scaler.transform(last_window)

    preds_scaled = []
    window = scaled_window.copy()
    for _ in range(days):
        x = window[-WINDOW:].reshape(1, WINDOW, 1)
        next_scaled = model.predict(x, verbose=0)[0]  # shape (1,)
        preds_scaled.append(next_scaled[0])
        window = np.vstack([window, next_scaled.reshape(1, 1)])

    preds_actual = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    last_date = close.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days)

    return pd.DataFrame({"Date": future_dates, "Predicted Close": preds_actual})


def run_full_pipeline(stock: str, forecast_days: int):
    """Download data, run the backtest, and forecast forward. Returns a dict
    ready to drop into a template context, or an 'error' key on failure."""
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime.now()

    df = get_stock_data(stock, start, end)

    if df.empty or "Close" not in df.columns:
        return {"error": f"No data found for ticker '{stock}'. Check the symbol and try again."}

    if len(df) < WINDOW + 20:
        return {
            "error": (
                f"Not enough history for '{stock}' to run the model "
                f"(need at least {WINDOW + 20} trading days)."
            )
        }

    try:
        test_dates, y_test_actual, y_pred_actual, scaler, metrics = run_backtest(df)
        forecast_df = forecast_future(df, scaler, forecast_days)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Model inference failed: {exc}"}

    return {
        "df": df,
        "test_dates": test_dates,
        "y_test_actual": y_test_actual,
        "y_pred_actual": y_pred_actual,
        "forecast_df": forecast_df,
        "metrics": metrics,
        "error": None,
    }


# --------------------------------------------------------------------------- #
# Chart builders (Plotly -> HTML div strings)
# --------------------------------------------------------------------------- #
PLOTLY_TEMPLATE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#8a95a1", size=12),
    margin=dict(l=40, r=20, t=90, b=32),
    title=dict(y=0.97, yanchor="top"),
    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
)


def plot_candlestick(df, ticker):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                increasing_line_color="#1f6f5c", decreasing_line_color="#b5473a",
            )
        ]
    )
    fig.update_layout(**PLOTLY_TEMPLATE_LAYOUT)
    fig.update_layout(title=f"{ticker} — daily candles", xaxis_rangeslider_visible=False, height=440)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_ema(df, ticker):
    ema20 = df.Close.ewm(span=20, adjust=False).mean()
    ema50 = df.Close.ewm(span=50, adjust=False).mean()
    ema100 = df.Close.ewm(span=100, adjust=False).mean()
    ema200 = df.Close.ewm(span=200, adjust=False).mean()

    fig = make_subplots(rows=1, cols=2, subplot_titles=("EMA 20 / 50", "EMA 100 / 200"))
    fig.add_trace(go.Scatter(x=df.index, y=df.Close, name="Close", line=dict(color="#c89b3c")), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA 20", line=dict(color="#1f6f5c")), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA 50", line=dict(color="#b5473a")), 1, 1)

    fig.add_trace(go.Scatter(x=df.index, y=df.Close, name="Close", line=dict(color="#c89b3c"), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=df.index, y=ema100, name="EMA 100", line=dict(color="#1f6f5c")), 1, 2)
    fig.add_trace(go.Scatter(x=df.index, y=ema200, name="EMA 200", line=dict(color="#b5473a")), 1, 2)

    fig.update_layout(**PLOTLY_TEMPLATE_LAYOUT)
    fig.update_layout(
        height=440, title=f"{ticker} — exponential moving averages",
        margin=dict(l=40, r=20, t=130, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=1.14, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_indicators(df, ticker):
    rsi = compute_rsi(df.Close)
    macd_line, signal_line, hist = compute_macd(df.Close)
    upper, sma, lower = compute_bollinger(df.Close)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Bollinger bands", "RSI (14)", "MACD"),
        row_heights=[0.4, 0.3, 0.3],
    )

    fig.add_trace(go.Scatter(x=df.index, y=df.Close, name="Close", line=dict(color="#c89b3c")), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=upper, name="Upper band", line=dict(color="#8a95a1", dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=sma, name="SMA 20", line=dict(color="#3d6ea5")), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=lower, name="Lower band", line=dict(color="#8a95a1", dash="dot")), 1, 1)

    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color="#7b5ea7")), 2, 1)
    fig.add_hline(y=70, line_dash="dash", line_color="#b5473a", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#1f6f5c", row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", line=dict(color="#3d6ea5")), 3, 1)
    fig.add_trace(go.Scatter(x=df.index, y=signal_line, name="Signal", line=dict(color="#c89b3c")), 3, 1)
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram", marker_color="#8a95a1"), 3, 1)

    fig.update_layout(**PLOTLY_TEMPLATE_LAYOUT)
    fig.update_layout(
        height=800, title=f"{ticker} — technical indicators",
        margin=dict(l=40, r=20, t=110, b=32),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_prediction_and_forecast(test_dates, y_test_actual, y_pred_actual, forecast_df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_actual, name="Actual (test)", line=dict(color="#1f6f5c")))
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred_actual, name="Predicted (test)", line=dict(color="#b5473a")))
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Predicted Close"],
            name="Forecast (future)", line=dict(color="#c89b3c", dash="dash"),
        )
    )
    fig.update_layout(**PLOTLY_TEMPLATE_LAYOUT)
    fig.update_layout(title=f"{ticker} — prediction vs. actual, plus forecast", height=440)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_comparison(df_a, ticker_a, df_b, ticker_b):
    """Normalize both series to % change from the start of the shared window
    so two very differently priced tickers can be read on one axis."""
    joined = pd.DataFrame({ticker_a: df_a["Close"], ticker_b: df_b["Close"]}).dropna()
    normalized = (joined / joined.iloc[0] - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker_a], name=ticker_a, line=dict(color="#1f6f5c")))
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[ticker_b], name=ticker_b, line=dict(color="#c89b3c")))
    fig.add_hline(y=0, line_color="#8a95a1", line_width=1)
    fig.update_layout(**PLOTLY_TEMPLATE_LAYOUT)
    fig.update_layout(title="Relative performance (% change, common period)", height=440,
                       yaxis_title="% change")
    return fig.to_html(full_html=False, include_plotlyjs=False)


# --------------------------------------------------------------------------- #
# Watchlist helpers (session-based, no account system)
# --------------------------------------------------------------------------- #
def get_watchlist() -> list:
    return session.get("watchlist", [])


def add_to_watchlist(ticker: str):
    wl = get_watchlist()
    if ticker not in wl:
        wl.append(ticker)
        wl = wl[-MAX_WATCHLIST:]
        session["watchlist"] = wl


def remove_from_watchlist(ticker: str):
    wl = [t for t in get_watchlist() if t != ticker]
    session["watchlist"] = wl


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.context_processor
def inject_globals():
    return {"popular_tickers": POPULAR_TICKERS, "watchlist": get_watchlist()}


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "stock": session.get("stock", ""),
        "error": None,
        "data_desc": None,
        "metrics": None,
        "candlestick_chart": None,
        "ema_chart": None,
        "indicator_chart": None,
        "prediction_chart": None,
        "forecast_table": None,
        "dataset_link": None,
        "forecast_link": None,
        "forecast_days": 30,
        "news": None,
    }

    if MODEL_LOAD_ERROR:
        context["error"] = f"Model failed to load: {MODEL_LOAD_ERROR}"
        return render_template("index.html", **context)

    prefill = request.args.get("stock")
    if prefill and request.method == "GET":
        context["stock"] = prefill.strip().upper()

    if request.method == "POST":
        stock = (request.form.get("stock") or "").strip().upper() or "^NSEI"
        try:
            forecast_days = int(request.form.get("forecast_days", 30))
        except ValueError:
            forecast_days = 30
        forecast_days = max(1, min(forecast_days, 90))  # clamp to a sane range

        session["stock"] = stock
        context["stock"] = stock
        context["forecast_days"] = forecast_days

        result = run_full_pipeline(stock, forecast_days)
        if result.get("error"):
            context["error"] = result["error"]
            return render_template("index.html", **context)

        df = result["df"]
        context["metrics"] = result["metrics"]
        context["data_desc"] = df.describe().to_html(classes="table")

        context["candlestick_chart"] = plot_candlestick(df, stock)
        context["ema_chart"] = plot_ema(df, stock)
        context["indicator_chart"] = plot_indicators(df, stock)
        context["prediction_chart"] = plot_prediction_and_forecast(
            result["test_dates"], result["y_test_actual"], result["y_pred_actual"],
            result["forecast_df"], stock,
        )
        context["forecast_table"] = result["forecast_df"].to_html(index=False, classes="table")
        context["news"] = get_news(stock)

        # Unique, sanitized export filenames so concurrent users don't collide.
        safe_stock = secure_filename(stock) or "stock"
        uid = uuid.uuid4().hex[:8]
        dataset_filename = f"{safe_stock}_dataset_{uid}.csv"
        forecast_filename = f"{safe_stock}_forecast_{uid}.csv"

        df.to_csv(os.path.join(EXPORT_DIR, dataset_filename))
        result["forecast_df"].to_csv(os.path.join(EXPORT_DIR, forecast_filename), index=False)

        context["dataset_link"] = dataset_filename
        context["forecast_link"] = forecast_filename

    return render_template("index.html", **context)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    context = {
        "stock_a": "AAPL",
        "stock_b": "MSFT",
        "forecast_days": 30,
        "error": None,
        "comparison_chart": None,
        "results": None,
    }

    if MODEL_LOAD_ERROR:
        context["error"] = f"Model failed to load: {MODEL_LOAD_ERROR}"
        return render_template("compare.html", **context)

    if request.method == "POST":
        stock_a = (request.form.get("stock_a") or "").strip().upper()
        stock_b = (request.form.get("stock_b") or "").strip().upper()
        try:
            forecast_days = int(request.form.get("forecast_days", 30))
        except ValueError:
            forecast_days = 30
        forecast_days = max(1, min(forecast_days, 90))

        context.update(stock_a=stock_a, stock_b=stock_b, forecast_days=forecast_days)

        if not stock_a or not stock_b:
            context["error"] = "Enter two ticker symbols to compare."
            return render_template("compare.html", **context)
        if stock_a == stock_b:
            context["error"] = "Pick two different tickers to compare."
            return render_template("compare.html", **context)

        result_a = run_full_pipeline(stock_a, forecast_days)
        if result_a.get("error"):
            context["error"] = f"{stock_a}: {result_a['error']}"
            return render_template("compare.html", **context)

        result_b = run_full_pipeline(stock_b, forecast_days)
        if result_b.get("error"):
            context["error"] = f"{stock_b}: {result_b['error']}"
            return render_template("compare.html", **context)

        context["comparison_chart"] = plot_comparison(
            result_a["df"], stock_a, result_b["df"], stock_b
        )
        context["results"] = [
            {"ticker": stock_a, "metrics": result_a["metrics"],
             "last_close": float(result_a["df"]["Close"].iloc[-1]),
             "forecast_end": float(result_a["forecast_df"]["Predicted Close"].iloc[-1])},
            {"ticker": stock_b, "metrics": result_b["metrics"],
             "last_close": float(result_b["df"]["Close"].iloc[-1]),
             "forecast_end": float(result_b["forecast_df"]["Predicted Close"].iloc[-1])},
        ]

    return render_template("compare.html", **context)


@app.route("/watchlist", methods=["GET", "POST"])
def watchlist():
    if request.method == "POST":
        ticker = (request.form.get("stock") or "").strip().upper()
        if ticker:
            add_to_watchlist(ticker)
    quotes = [get_quick_quote(t) or {"symbol": t, "price": None, "change_pct": None, "up": True}
              for t in get_watchlist()]
    return render_template("watchlist.html", quotes=quotes)


@app.route("/watchlist/add", methods=["POST"])
def watchlist_add():
    ticker = (request.form.get("stock") or "").strip().upper()
    if ticker:
        add_to_watchlist(ticker)
    return jsonify({"ok": True, "watchlist": get_watchlist()})


@app.route("/watchlist/remove/<ticker>", methods=["POST"])
def watchlist_remove(ticker):
    remove_from_watchlist(ticker.strip().upper())
    return jsonify({"ok": True, "watchlist": get_watchlist()})


@app.route("/api/ticker-tape")
def api_ticker_tape():
    quotes = [q for q in (get_quick_quote(t) for t in TAPE_TICKERS) if q]
    return jsonify(quotes)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/download/<path:filename>")
def download_file(filename):
    safe_name = secure_filename(filename)
    return send_from_directory(EXPORT_DIR, safe_name, as_attachment=True)


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)
