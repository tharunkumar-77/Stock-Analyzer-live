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
import requests
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "stocks.csv")

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

# Load CSV once at startup instead of on every request
try:
    _stocks_df = pd.read_csv(CSV_PATH)
    _stocks_df["name"] = _stocks_df["name"].astype(str)
    _stocks_df["ticker"] = _stocks_df["ticker"].astype(str)
except Exception as e:
    print(f"[WARN] Could not load stocks.csv from {CSV_PATH}: {e}")
    _stocks_df = pd.DataFrame(columns=["name", "ticker", "category"])


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


def get_sentiment(symbol):
    try:
        if not HF_API_KEY:
            return None, None
        ticker = yf.Ticker(symbol)
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
    if history.empty or len(history) < 2:
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
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _normalize_index(history):
    if history.index.tz is not None:
        history.index = history.index.tz_convert(None)
    return history


def _fetch_with_retry(fn, retries=3, delay=2):
    for i in range(retries):
        try:
            result = fn()
            if result is not None and (not hasattr(result, 'empty') or not result.empty):
                return result
        except Exception:
            pass
        if i < retries - 1:
            time.sleep(delay)
    return fn()


def _encode_figure(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_chart(history, color):
    if history.empty:
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
    if history.empty:
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
    if monthly_history.empty:
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
    if h1.empty or h2.empty:
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

def get_detail(stock):
    """Look up an Indian stock/ETF/index purely from stocks.csv, then pull data from Yahoo Finance."""
    stock = (stock or "").strip()
    if not stock:
        return {"error": "No stock provided"}

    resolved = resolve_ticker(stock)
    if not resolved:
        return {"error": f"'{stock}' was not found in our list of Indian stocks/ETFs."}

    try:
        ticker = yf.Ticker(resolved)
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_info = ex.submit(lambda: _fetch_with_retry(lambda: ticker.info))
            f_1y   = ex.submit(lambda: _fetch_with_retry(lambda: ticker.history(period="1y")))
            f_3y   = ex.submit(lambda: _fetch_with_retry(lambda: ticker.history(period="3y")))
            f_5y   = ex.submit(lambda: _fetch_with_retry(lambda: ticker.history(period="5y")))
            info       = f_info.result()
            history_1y = _normalize_index(f_1y.result())
            history_3y = _normalize_index(f_3y.result())
            history_5y = _normalize_index(f_5y.result())
        name          = info.get("longName") or info.get("shortName") or "Not Available"
        symbol        = info.get("symbol") or resolved
        price         = info.get("currentPrice") or info.get("regularMarketPrice") or "Not Available"
        asset_type    = info.get("quoteType") or "Not Available"
        raw_expense   = info.get("annualReportExpenseRatio") or info.get("netExpenseRatio")
        expense_ratio = round(raw_expense * 100, 2) if raw_expense else "Not Available"
        raw_exp       = info.get("longBusinessSummary", "")
        explanation   = (raw_exp[:510] + "...") if len(raw_exp) > 510 else raw_exp or "Not Available"

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
            return {"error": f"No price data found for '{stock}'. Try a different stock."}
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
        ticker  = yf.Ticker(resolved)
        history = ticker.history(start=start_date, end=end_date)
        history = _normalize_index(history)

        if history.empty:
            return jsonify({"error": "No data found for the selected date range"})

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
        ticker    = yf.Ticker(resolved_stock)
        history5y = _normalize_index(ticker.history(period="5y"))
        info      = ticker.info
        symbol    = info.get("symbol") or resolved_stock.upper()
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
        h1 = _normalize_index(yf.Ticker(details1["symbol"]).history(period="1y"))
        h2 = _normalize_index(yf.Ticker(details2["symbol"]).history(period="1y"))
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
