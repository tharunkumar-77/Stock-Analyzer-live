from flask import Flask, request, render_template, jsonify
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import time
import os
import random
import requests
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "stocks.csv")

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP session for yfinance
# Using a real browser User-Agent significantly reduces the odds of Yahoo
# Finance soft-blocking requests coming from a shared/free-tier hosting IP.
# ─────────────────────────────────────────────────────────────────────────────
_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
})


def make_ticker(symbol):
    """Create a yfinance Ticker using our shared, browser-like session."""
    return yf.Ticker(symbol, session=_session)


# ─────────────────────────────────────────────────────────────────────────────
# Very small in-memory cache to avoid hammering Yahoo Finance with repeat
# requests for the same symbol (helps a LOT with free-tier rate limits).
# ─────────────────────────────────────────────────────────────────────────────
_cache = {}
INFO_CACHE_TTL = 15 * 60       # 15 minutes — info/price rarely needs to be fresher
HISTORY_CACHE_TTL = 10 * 60    # 10 minutes


def cache_get(key):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < entry[2]:
        return entry[1]
    return None


def cache_set(key, value, ttl):
    _cache[key] = (time.time(), value, ttl)


# Load CSV once at startup instead of on every request
try:
    _stocks_df = pd.read_csv(CSV_PATH)
    _stocks_df["name"] = _stocks_df["name"].astype(str)
    _stocks_df["ticker"] = _stocks_df["ticker"].astype(str)
    if "description" not in _stocks_df.columns:
        _stocks_df["description"] = ""
    _stocks_df["description"] = _stocks_df["description"].fillna("").astype(str)
except Exception as e:
    print(f"[WARN] Could not load stocks.csv from {CSV_PATH}: {e}")
    _stocks_df = pd.DataFrame(columns=["name", "ticker", "category", "description"])


def get_suggestions():
    return _stocks_df["name"].tolist()


def resolve_ticker(stock):
    """
    Resolve user input to a known Indian ticker using ONLY stocks.csv.
    Tries exact name match, exact ticker match, then partial name match.
    Returns the ticker string, or None if nothing matches.
    """
    if not stock:
        return None
    stock_clean = stock.strip().lower()

    exact = _stocks_df[
        (_stocks_df["name"].str.lower() == stock_clean) |
        (_stocks_df["ticker"].str.lower() == stock_clean)
    ]
    if not exact.empty:
        return exact.iloc[0]["ticker"]

    partial = _stocks_df[_stocks_df["name"].str.lower().str.contains(stock_clean, na=False, regex=False)]
    if not partial.empty:
        return partial.iloc[0]["ticker"]

    return None


def lookup_csv_name(ticker_symbol):
    """Best-effort fallback name lookup from stocks.csv when Yahoo info fails."""
    match = _stocks_df[_stocks_df["ticker"].str.lower() == str(ticker_symbol).lower()]
    if not match.empty:
        return match.iloc[0]["name"]
    return None


def lookup_csv_description(ticker_symbol):
    """Best-effort fallback business/index description from stocks.csv.
    Used when Yahoo's longBusinessSummary is empty (always the case for
    indices, and often the case for ETFs/mutual funds)."""
    match = _stocks_df[_stocks_df["ticker"].str.lower() == str(ticker_symbol).lower()]
    if not match.empty:
        desc = match.iloc[0].get("description", "")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
    return None


def get_sentiment(symbol):
    try:
        if not HF_API_KEY:
            return None, None
        ticker = make_ticker(symbol)
        news = ticker.news or []
        headlines = [
            (item.get("content", {}).get("title") or item.get("title", ""))
            for item in news[:8]
        ]
        headlines = [h for h in headlines if h]
        if not headlines:
            return None, None

        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": headlines},
            timeout=15
        )
        if response.status_code != 200:
            return None, None

        results = response.json()
        # HF returns list of lists when inputs is a list
        if not results or not isinstance(results, list):
            return None, None

        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        scores = []
        for item in results:
            # each item is a list of dicts [{"label":..,"score":..}, ...]
            if isinstance(item, list):
                top = max(item, key=lambda x: x["score"])
            elif isinstance(item, dict):
                top = item
            else:
                continue
            label = top.get("label", "neutral").lower()
            scores.append(score_map.get(label, 0) * top["score"])

        if not scores:
            return None, None

        avg = sum(scores) / len(scores)
        label = "Positive" if avg > 0.15 else "Negative" if avg < -0.15 else "Neutral"
        return label, round(avg, 2)
    except Exception:
        return None, None


def calculate_return(history):
    if history is None or history.empty or len(history) < 2:
        return 0.0
    first = history["Close"].iloc[0]
    if first == 0:
        return 0.0
    return ((history["Close"].iloc[-1] - first) / first) * 100


def calculate_projection(amount, return_5y, sentiment_score, horizon_days):
    try:
        annual_rate = (1 + float(return_5y) / 100) ** (1 / 5) - 1
    except Exception:
        annual_rate = 0.08
    years = horizon_days / 365.25
    sentiment_weight = max(0.0, 1.0 - (horizon_days / 365))
    sentiment_nudge = (sentiment_score or 0) * sentiment_weight * 0.05
    adjusted_rate = max(annual_rate + sentiment_nudge, -0.99)
    projected = round(amount * (1 + adjusted_rate) ** years, 2)
    uncertainty = min(0.5, 0.08 + 0.1 * years)
    return projected, round(projected * (1 - uncertainty), 2), round(projected * (1 + uncertainty), 2)


def _encode_figure(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _normalize_index(history):
    if history is None or history.empty:
        return history
    if history.index.tz is not None:
        history.index = history.index.tz_convert(None)
    return history


def _fetch_with_retry(fn, retries=4, base_delay=1.5):
    """
    Retry helper with exponential backoff + jitter.
    Never lets an exception escape on the final attempt — returns None instead,
    so callers can degrade gracefully rather than crashing with a raw traceback.
    """
    last_result = None
    for i in range(retries):
        try:
            result = fn()
            if result is not None and (not hasattr(result, 'empty') or not result.empty):
                return result
            last_result = result
        except Exception:
            last_result = None
        if i < retries - 1:
            sleep_time = base_delay * (2 ** i) + random.uniform(0, 0.5)
            time.sleep(sleep_time)
    return last_result


def safe_get_info(ticker):
    """Fetch ticker.info defensively. Always returns a dict (never None)."""
    result = _fetch_with_retry(lambda: ticker.info, retries=4, base_delay=1.5)
    if not isinstance(result, dict):
        return {}
    return result


def make_chart(history, color):
    if history is None or history.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=100)
    ax.plot(history.index, history["Close"], color=color, linewidth=1.5)
    ax.fill_between(history.index, history["Close"], alpha=0.08, color=color)
    ax.set_xlim(history.index[0], history.index[-1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=1.0)
    return _encode_figure(fig)


def generate_growth_chart(history, amount):
    if history is None or history.empty:
        return None
    first_price = history["Close"].iloc[0]
    if first_price == 0:
        return None
    shares = amount / first_price
    portfolio = history["Close"] * shares
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=100)
    ax.plot(history.index, portfolio, color="#198754", linewidth=1.5, label="Portfolio value")
    ax.axhline(y=amount, color="#dc3545", linestyle="--", linewidth=1.2, label=f"Invested ₹{amount:,.0f}")
    ax.fill_between(history.index, portfolio, amount,
                    where=(portfolio >= amount), alpha=0.1, color="#198754")
    ax.fill_between(history.index, portfolio, amount,
                    where=(portfolio < amount), alpha=0.1, color="#dc3545")
    ax.set_xlim(history.index[0], history.index[-1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=1.0)
    return _encode_figure(fig)


def generate_growth_chart_monthly(monthly_history, amount):
    if monthly_history is None or monthly_history.empty:
        return None
    total_units, running_invested = 0, 0
    portfolio_values, total_invested_values = [], []
    valid_index = []
    for idx, row in monthly_history.iterrows():
        if row["Close"] == 0:
            continue
        total_units += amount / row["Close"]
        running_invested += amount
        portfolio_values.append(total_units * row["Close"])
        total_invested_values.append(running_invested)
        valid_index.append(idx)
    if not portfolio_values:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=100)
    ax.plot(valid_index, portfolio_values, color="#198754", linewidth=1.5, label="Portfolio value")
    ax.plot(valid_index, total_invested_values, color="#dc3545", linestyle="--", linewidth=1.2, label="Total invested")
    ax.fill_between(valid_index, portfolio_values, total_invested_values,
                    where=[p >= t for p, t in zip(portfolio_values, total_invested_values)],
                    alpha=0.1, color="#198754")
    ax.set_xlim(valid_index[0], valid_index[-1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=1.0)
    return _encode_figure(fig)


def generate_comparison_chart(h1, h2, label1, label2):
    if h1 is None or h2 is None or h1.empty or h2.empty:
        return None
    if h1["Close"].iloc[0] == 0 or h2["Close"].iloc[0] == 0:
        return None
    n1 = h1["Close"] / h1["Close"].iloc[0] * 100
    n2 = h2["Close"] / h2["Close"].iloc[0] * 100
    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
    ax.plot(h1.index, n1, label=label1, color="#0d6efd", linewidth=1.5)
    ax.plot(h2.index, n2, label=label2, color="#fd7e14", linewidth=1.5)
    ax.axhline(y=100, color="#adb5bd", linestyle="--", linewidth=0.8)
    ax.set_xlim(min(h1.index[0], h2.index[0]), max(h1.index[-1], h2.index[-1]))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=1.0)
    return _encode_figure(fig)


def fetch_history_cached(symbol, period=None, start=None, end=None):
    """Fetch history via yfinance with caching + retry, keyed by params."""
    key = f"hist:{symbol}:{period}:{start}:{end}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    ticker = make_ticker(symbol)

    def _do_fetch():
        if period:
            return ticker.history(period=period)
        return ticker.history(start=start, end=end)

    history = _fetch_with_retry(_do_fetch, retries=4, base_delay=1.5)
    if history is None:
        history = pd.DataFrame()
    history = _normalize_index(history)
    cache_set(key, history, HISTORY_CACHE_TTL)
    return history


def fetch_info_cached(symbol):
    key = f"info:{symbol}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    ticker = make_ticker(symbol)
    info = safe_get_info(ticker)
    cache_set(key, info, INFO_CACHE_TTL)
    return info


def get_detail(stock):
    """Look up an Indian stock/ETF/index purely from stocks.csv, then pull data from Yahoo Finance."""
    stock = (stock or "").strip()
    if not stock:
        return {"error": "No stock provided"}

    resolved = resolve_ticker(stock)
    if not resolved:
        return {"error": f"'{stock}' was not found in our list of Indian stocks/ETFs."}

    try:
        # Sequential-ish fetch with only 2 workers max — fewer simultaneous
        # requests to Yahoo Finance means fewer rate-limit hits on free hosting.
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_info = ex.submit(fetch_info_cached, resolved)
            f_1y = ex.submit(fetch_history_cached, resolved, "1y")
            info = f_info.result()
            history_1y = f_1y.result()

        history_3y = fetch_history_cached(resolved, "3y")
        history_5y = fetch_history_cached(resolved, "5y")

        csv_name = lookup_csv_name(resolved)
        name = info.get("longName") or info.get("shortName") or csv_name or "Not Available"
        symbol = info.get("symbol") or resolved
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None and not history_1y.empty:
            price = round(history_1y["Close"].iloc[-1], 2)
        if price is None:
            price = "Not Available"
        asset_type = info.get("quoteType") or "Not Available"
        raw_expense = info.get("annualReportExpenseRatio") or info.get("netExpenseRatio")
        expense_ratio = round(raw_expense * 100, 2) if raw_expense else "Not Available"
        raw_exp = info.get("longBusinessSummary", "") or ""
        if raw_exp:
            explanation = (raw_exp[:510] + "...") if len(raw_exp) > 510 else raw_exp
        else:
            # Yahoo doesn't provide longBusinessSummary for indices, and often
            # not for ETFs/mutual funds either — fall back to stocks.csv.
            csv_description = lookup_csv_description(resolved)
            explanation = csv_description or "Not Available"

        if not history_1y.empty or not history_3y.empty or not history_5y.empty:
            return {
                "name":          name,
                "symbol":        symbol,
                "price":         price,
                "asset_type":    asset_type,
                "explanation":   explanation,
                "expense_ratio": expense_ratio,
                "return_1y":     round(calculate_return(history_1y), 2),
                "return_3y":     round(calculate_return(history_3y), 2),
                "return_5y":     round(calculate_return(history_5y), 2),
                "chart_1y":      make_chart(history_1y, "#0d6efd"),
                "chart_3y":      make_chart(history_3y, "#6610f2"),
                "chart_5y":      make_chart(history_5y, "#fd7e14"),
            }
        else:
            return {"error": "Data provider is currently rate-limiting requests. Please wait a minute and try again."}
    except Exception as e:
        return {"error": f"Could not fetch data: {e}"}


@app.route("/")
def home():
    suggestions = get_suggestions()
    return render_template("home.html", suggestions=suggestions)


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(silent=True) or {}
    stock = data.get("stock", "").strip()
    if not stock:
        return jsonify({"error": "No stock provided"})
    details = get_detail(stock)
    if "error" in details:
        return jsonify(details)
    sentiment_label, sentiment_score = get_sentiment(details.get("symbol", stock))
    details["sentiment_label"] = sentiment_label
    details["sentiment_score"] = sentiment_score
    return jsonify(details)


@app.route("/api/historical", methods=["POST"])
def api_historical():
    data       = request.get_json(silent=True) or {}
    stock      = data.get("stock", "").strip()
    start_date = data.get("start_date", "")
    end_date   = data.get("end_date", "")
    monthly    = data.get("monthly", False)

    if not stock:
        return jsonify({"error": "No stock provided"})

    resolved = resolve_ticker(stock) or stock  # currentSymbol from the frontend is already resolved

    try:
        amount = float(data.get("amount", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid amount"})

    if amount <= 0:
        return jsonify({"error": "Amount must be greater than zero"})

    try:
        history = fetch_history_cached(resolved, start=start_date, end=end_date)

        if history is None or history.empty:
            return jsonify({"error": "No data found for the selected date range, or the data provider is temporarily rate-limiting requests. Please try again shortly."})

        if history["Close"].iloc[0] == 0:
            return jsonify({"error": "Invalid price data for this stock"})

        if monthly:
            monthly_history = history.resample("MS").first()
            total_units, total_invested = 0, 0
            for _, row in monthly_history.iterrows():
                if row["Close"] == 0:
                    continue
                total_units    += amount / row["Close"]
                total_invested += amount
            final_value  = round(total_units * history["Close"].iloc[-1], 2)
            profit       = round(final_value - total_invested, 2)
            buy_price    = round(history["Close"].iloc[0], 2)
            sell_price   = round(history["Close"].iloc[-1], 2)
            growth_chart = generate_growth_chart_monthly(monthly_history, amount)
        else:
            buy_price    = round(history["Close"].iloc[0], 2)
            sell_price   = round(history["Close"].iloc[-1], 2)
            shares       = amount / buy_price
            final_value  = round(shares * sell_price, 2)
            profit       = round(final_value - amount, 2)
            total_invested = amount
            growth_chart = generate_growth_chart(history, amount)

        return jsonify({
            "buy_price":      buy_price,
            "sell_price":     sell_price,
            "returns":        profit,
            "total_invested": round(total_invested, 2),
            "growth_chart":   growth_chart,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/future", methods=["POST"])
def api_future():
    data  = request.get_json(silent=True) or {}
    stock = data.get("stock", "").strip()

    if not stock:
        return jsonify({"error": "No stock provided"})

    try:
        amount  = float(data.get("amount", 0))
        horizon = int(data.get("horizon", 30))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid amount or horizon"})

    if amount <= 0:
        return jsonify({"error": "Amount must be greater than zero"})

    if horizon not in [7, 14, 30, 90, 180, 365]:
        return jsonify({"error": "Invalid horizon value"})

    resolved_stock = resolve_ticker(stock) or stock  # currentSymbol from frontend is already resolved

    try:
        history5y = fetch_history_cached(resolved_stock, "5y")
        info = fetch_info_cached(resolved_stock)
        symbol = info.get("symbol") or resolved_stock.upper()
        return_5y = round(calculate_return(history5y), 2)
    except Exception as e:
        return jsonify({"error": f"Could not fetch data: {e}"})

    sentiment_label, sentiment_score = get_sentiment(symbol)
    projected, low, high = calculate_projection(amount, return_5y, sentiment_score, horizon)

    horizon_labels = {7: "1 Week", 14: "2 Weeks", 30: "1 Month", 90: "3 Months", 180: "6 Months", 365: "1 Year"}

    return jsonify({
        "projected":       projected,
        "low":             low,
        "high":            high,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "horizon_label":   horizon_labels.get(horizon, f"{horizon} days"),
    })


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data   = request.get_json(silent=True) or {}
    stock1 = data.get("stock1", "").strip()
    stock2 = data.get("stock2", "").strip()

    if not stock1 or not stock2:
        return jsonify({"error": "Both stock symbols are required"})

    details1 = get_detail(stock1)
    if "error" in details1:
        return jsonify({"error": details1["error"]})

    details2 = get_detail(stock2)
    if "error" in details2:
        return jsonify({"error": details2["error"]})

    try:
        h1 = fetch_history_cached(details1["symbol"], "1y")
        h2 = fetch_history_cached(details2["symbol"], "1y")
        comparison_chart = generate_comparison_chart(h1, h2, details1["symbol"], details2["symbol"])
    except Exception:
        comparison_chart = None

    r1, r2 = details1["return_1y"], details2["return_1y"]
    winner = details1["symbol"] if r1 > r2 else details2["symbol"] if r2 > r1 else "Tie"
    diff   = round(abs(r1 - r2), 2)

    return jsonify({
        "details1":         details1,
        "details2":         details2,
        "winner":           winner,
        "diff":             diff,
        "comparison_chart": comparison_chart,
    })


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", port=5001)