"""Yahoo Finance chart API (no key): long DAILY histories for stocks, indices and commodity futures, and recent
intraday bars for live forward tests.

Yahoo serves intraday history only for the recent past (15m: ~60 days, 1h: ~730 days), so 10+ year research on
these markets runs on D1 (`factory run --symbols AAPL,... --timeframes D1`). FX / spot gold and silver come from
Dukascopy or the GitHub mirror instead (M1/M15 back to 2003/2012).

Our symbol -> Yahoo ticker map below; any other symbol is passed through unchanged (e.g. `--symbols NFLX`).
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "propfloor/0.1 (research)"}
TICKERS = {
    # commodities (front-month futures, continuous)
    "WTI": "CL=F", "BRENT": "BZ=F", "GOLDF": "GC=F", "SILVERF": "SI=F", "NATGAS": "NG=F", "COPPER": "HG=F",
    # indices
    "US500": "^GSPC", "US100": "^NDX", "US30": "^DJI", "GER40": "^GDAXI", "UK100": "^FTSE", "JPN225": "^N225",
    # FX (Yahoo quotes; prefer Dukascopy for research history)
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X",
    "USDCHF": "CHF=X", "NZDUSD": "NZDUSD=X",
}
MAJOR_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "BRK-B"]
INTERVALS = {"D1": "1d", "H1": "60m", "M15": "15m", "M5": "5m", "M1": "1m"}


def ticker(symbol: str) -> str:
    return TICKERS.get(symbol.upper(), symbol.upper())


def parse(js: dict) -> pd.DataFrame:
    res = (js.get("chart") or {}).get("result") or []
    if not res or not res[0].get("timestamp"):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    r = res[0]
    q = r["indicators"]["quote"][0]
    idx = pd.to_datetime(r["timestamp"], unit="s", utc=True).as_unit("ns")
    df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")}, index=idx, dtype=float)
    df.index.name = "time"
    df["volume"] = df["volume"].fillna(0.0)
    return df.dropna(subset=["open", "high", "low", "close"])


def _epoch(t) -> int:
    t = pd.Timestamp(t)
    return int((t.tz_localize("UTC") if t.tzinfo is None else t).timestamp())


def fetch(symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp | None = None, tf: str = "D1",
          session: requests.Session | None = None, retries: int = 4) -> pd.DataFrame:
    p1 = _epoch(start)
    p2 = _epoch(end) if end is not None else int(time.time())
    params = {"period1": p1, "period2": p2, "interval": INTERVALS[tf], "includePrePost": "false", "events": ""}
    s = session or requests.Session()
    url = URL.format(ticker=ticker(symbol))
    for i in range(retries):
        try:
            r = s.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                raise ValueError(f"unknown Yahoo ticker {ticker(symbol)}")
            r.raise_for_status()
            return parse(r.json())
        except requests.RequestException as e:
            log.warning("yahoo %s attempt %d: %s", symbol, i + 1, e)
            time.sleep(2 ** (i + 1))
    raise ConnectionError(f"failed to download {url}")


def download(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Daily bars (the only resolution Yahoo keeps for 10+ years)."""
    df = fetch(symbol, start, end, tf="D1")
    # daily bars are stamped at the exchange's midnight; keep the date at 00:00 UTC so D1 resampling is stable
    df.index = df.index.normalize()
    return df[~df.index.duplicated(keep="last")]


def recent(symbol: str, since: pd.Timestamp, tf: str = "D1") -> pd.DataFrame:
    """Recent bars for live forward tests (intraday only goes back ~60 days for 15m)."""
    df = fetch(symbol, since, None, tf=tf)
    if tf == "D1":
        df.index = df.index.normalize()
    return df[~df.index.duplicated(keep="last")]
