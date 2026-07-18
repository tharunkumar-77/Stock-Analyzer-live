# Stock-Analyzer

A full-stack web app that helps beginner investors make sense of Indian stocks, ETFs, funds, and indices — real historical performance, sentiment-adjusted future projections, and side-by-side comparisons, all in one place.

Live app: https://stock-analyzer-live-1.onrender.com/
(hosted on Render's free tier — may take 20-30 seconds to wake up on first load)

Features


Search any Indian stock, ETF, fund, or index by name or ticker
Historical performance — 1Y / 3Y / 5Y returns with charts
Past returns calculator — see actual profit/loss for a lump sum or monthly SIP invested between any two dates, using real historical price data
Future returns projection — lump sum or SIP, projected using 5-year annualized growth adjusted by current news sentiment
News sentiment analysis — powered by FinBERT (via Hugging Face), scored from recent headlines
Compare two investments — side-by-side stats with an overlaid normalized performance chart


Tech Stack


Backend: Flask, Python
Data: yfinance, Pandas
Sentiment: FinBERT (Hugging Face Inference API)
Charts: Matplotlib
Frontend: Bootstrap, vanilla JS, AJAX/JSON API architecture
Deployment: Render


Engineering notes


API responses are cached in-memory with different TTLs for successful vs. empty/failed fetches, to reduce redundant calls to Yahoo Finance.
Failed requests to the data provider are retried with exponential backoff and jitter, to handle rate limiting gracefully instead of failing outright.
Info and historical data are fetched concurrently using a thread pool to reduce response latency.
FinBERT/torch are excluded from requirements.txt in the deployed version to stay within Render's free-tier memory limits; sentiment analysis runs via the Hugging Face Inference API instead of a local model.



bashgit clone https://github.com/tharunkumar-77/Stock-Analyzer-live.git
cd Stock-Analyzer-live
pip install -r requirements.txt
python app.py

Set a HF_API_KEY environment variable if you want news sentiment analysis to work locally.

Disclaimer

This project is for educational purposes only and does not constitute financial advice. Projections are based on historical data and simple models, not guarantees of future performance.
