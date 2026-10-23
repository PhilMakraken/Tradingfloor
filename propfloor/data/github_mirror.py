"""Free FX bars mirrored on GitHub (ejtraderLabs/historical-data, MetaTrader exports).

Files: {SYM}/{SYM}{tf}.csv with tf in m15, m30, h1, h4, d1; columns Date,open,high,low,close,tick_volume.
Prices are integer points (divide by 10**digits); timestamps are broker server time = New York + 7h
(midnight = 17:00 NY roll). Coverage roughly 2012-11 .. 2022-03.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/{sym}/{sym}{tf}.csv"
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY", "AUDJPY",
           "EURCHF", "XAUUSD"]
CACHE = Path("data/downloads/github_mirror")
PLAUSIBLE = {"JPY": (50, 250), "XAU": (500, 5000)}


def _scale(sym: str, first_close: float) -> float:
    lo, hi = next((v for k, v in PLAUSIBLE.items() if k in sym), (0.3, 3.0))
    for d in range(0, 7):
        if lo <= first_close / 10**d <= hi:
            return 10.0**d
    raise ValueError(f"cannot infer price scale for {sym} ({first_close})")


def server_to_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Broker time = NY + 7h, so subtract 7h and localise as New York (DST-correct)."""
    ny = (idx - pd.Timedelta(hours=7)).tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    return ny.tz_convert("UTC")


def parse(text: str, sym: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    idx = server_to_utc(pd.DatetimeIndex(pd.to_datetime(df["Date"])))
    sc = _scale(sym, float(df["close"].iloc[0]))
    out = pd.DataFrame({k: df[k].to_numpy(float) / sc for k in ("open", "high", "low", "close")}, index=idx.as_unit("ns"))
    out["volume"] = df["tick_volume"].to_numpy(float)
    out = out[~np.asarray(out.index.isna())]
    out.index.name = "time"
    return out


def download(symbol: str, start: str | None = None, end: str | None = None, tf: str = "m15") -> pd.DataFrame:
    sym = symbol.upper()
    cache = CACHE / f"{sym}{tf}.csv"
    if cache.exists():
        text = cache.read_text()
    else:
        r = requests.get(URL.format(sym=sym, tf=tf), timeout=300)
        r.raise_for_status()
        text = r.text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    df = parse(text, sym)
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df
