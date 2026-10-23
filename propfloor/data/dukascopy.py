"""Dukascopy free historical data (daily files of 1-minute BID candles, LZMA-compressed .bi5).

URL: https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM-1:02d}/{DD:02d}/BID_candles_min_1.bi5
Record (24 bytes, big endian): uint32 seconds-from-midnight-UTC, uint32 open, close, low, high
(prices in points, divide by 10**digits), float32 volume.
"""
from __future__ import annotations

import logging
import lzma
import struct
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
REC = struct.Struct(">5If")
CACHE = Path("data/downloads/dukascopy")


def digits(symbol: str) -> int:
    s = symbol.upper()
    if s.endswith("JPY") or s.startswith("XAG"):
        return 3
    if s.startswith("XAU"):
        return 3
    return 5


def decode_bi5(raw: bytes, day: date, symbol: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    data = lzma.decompress(raw)
    arr = np.frombuffer(data, dtype=np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")]))
    scale = 10.0 ** digits(symbol)
    idx = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(arr["t"].astype(np.int64), unit="s")
    df = pd.DataFrame({"open": arr["o"] / scale, "high": arr["h"] / scale, "low": arr["l"] / scale,
                       "close": arr["c"] / scale, "volume": arr["v"].astype(float)}, index=pd.DatetimeIndex(idx).as_unit("ns"))
    df.index.name = "time"
    return df[df["volume"] > 0]  # zero-volume rows are flat fillers when the market is closed


def fetch_day(symbol: str, day: date, session: requests.Session | None = None, side: str = "BID",
              retries: int = 4) -> pd.DataFrame:
    sym = symbol.upper()
    cache = CACHE / sym / f"{day.isoformat()}_{side}.bi5"
    if cache.exists():
        return decode_bi5(cache.read_bytes(), day, sym)
    url = URL.format(sym=sym, y=day.year, m=day.month - 1, d=day.day, side=side)
    s = session or requests.Session()
    for i in range(retries):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 404:
                return decode_bi5(b"", day, sym)
            r.raise_for_status()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(r.content)
            return decode_bi5(r.content, day, sym)
        except requests.RequestException as e:
            log.warning("dukascopy %s %s attempt %d: %s", sym, day, i + 1, e)
            time.sleep(2 ** (i + 1))
    raise ConnectionError(f"failed to download {url}")


def download(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    d0 = pd.Timestamp(start).date()
    d1 = pd.Timestamp(end).date() if end else date.today() - timedelta(days=1)
    sess = requests.Session()
    frames = []
    d = d0
    while d <= d1:
        if d.weekday() != 5:  # Saturdays are always empty
            frames.append(fetch_day(symbol, d, sess))
        d += timedelta(days=1)
    frames = [f for f in frames if len(f)]
    return pd.concat(frames) if frames else pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def encode_bi5(df: pd.DataFrame, day: date, symbol: str) -> bytes:
    """Inverse of decode_bi5 (used by tests)."""
    scale = 10 ** digits(symbol)
    base = pd.Timestamp(day, tz="UTC")
    out = b"".join(REC.pack(int((t - base).total_seconds()), round(r.open * scale), round(r.close * scale),
                            round(r.low * scale), round(r.high * scale), float(r.volume)) for t, r in df.iterrows())
    return lzma.compress(out, format=lzma.FORMAT_ALONE)
