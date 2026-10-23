"""Auto-detecting loader for user-provided OHLC / tick CSV files (data/raw/)."""
from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PAIR_RE = re.compile(r"(?<![A-Z])(XAU|XAG|EUR|GBP|USD|JPY|AUD|NZD|CAD|CHF|SEK|NOK|SGD|HKD|MXN|ZAR|TRY|PLN)"
                     r"[_\-/]?(USD|JPY|EUR|GBP|AUD|NZD|CAD|CHF|SEK|NOK|SGD|HKD|MXN|ZAR|TRY|PLN)(?![A-Z])", re.I)

ALIASES = {
    "open": ["open", "<open>", "o", "openbid", "bidopen", "open_bid"],
    "high": ["high", "<high>", "h", "highbid", "bidhigh", "high_bid"],
    "low": ["low", "<low>", "l", "lowbid", "bidlow", "low_bid"],
    "close": ["close", "<close>", "c", "closebid", "bidclose", "close_bid", "price"],
    "volume": ["volume", "<vol>", "<tickvol>", "tickvol", "tick_volume", "vol", "v"],
    "datetime": ["datetime", "date_time", "timestamp", "time (utc)", "gmt time", "local time", "time", "date"],
    "date": ["date", "<date>", "day"],
    "time": ["time", "<time>", "hour"],
    "bid": ["bid", "bidprice", "bid_price"],
    "ask": ["ask", "askprice", "ask_price", "offer"],
}

# Known source timezones (HistData ASCII is EST without DST)
FORMAT_TZ = {"histdata": "Etc/GMT+5", "mt4": "UTC", "mt5": "UTC", "dukascopy": "UTC", "generic": "UTC", "tick": "UTC"}


@dataclass
class Detected:
    fmt: str
    symbol: str | None
    tz: str
    df: pd.DataFrame  # UTC-indexed OHLC(V)


def symbol_from_name(name: str) -> str | None:
    m = PAIR_RE.search(name.upper())
    return (m.group(1) + m.group(2)).upper() if m else None


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))]
            return z.read(names[0]).decode("utf-8", errors="replace")
    return path.read_text(errors="replace")


def _sniff(text: str) -> tuple[str, bool]:
    sample = "\n".join(text.splitlines()[:20])
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t| ").delimiter
    except csv.Error:
        delim = max([",", ";", "\t", "|"], key=sample.count)
    first = text.splitlines()[0]
    has_header = bool(re.search(r"[A-Za-z<]", first.split(delim)[-1] + first.split(delim)[0])) and not re.match(
        r"^\d{4}[.\-/]?\d{2}", first)
    return delim, has_header


def _col(cols: list[str], key: str) -> str | None:
    low = {c.strip().lower(): c for c in cols}
    for a in ALIASES[key]:
        if a in low:
            return low[a]
    return None


def _parse_dt(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    sample = s.iloc[0]
    fmts = [
        (r"^\d{8} \d{6}$", "%Y%m%d %H%M%S"),
        (r"^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}(:\d{2})?$", None),
        (r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}(\.\d+)?", "%d.%m.%Y %H:%M:%S.%f"),
        (r"^\d{8}$", "%Y%m%d"),
    ]
    for pat, fmt in fmts:
        if re.match(pat, sample):
            if fmt is None:
                return pd.to_datetime(s.str.replace(".", "-", n=2), format="mixed")
            if "%f" in fmt and "." not in sample:
                fmt = fmt.replace(".%f", "")
            return pd.to_datetime(s, format=fmt)
    if re.match(r"^\d{10,13}$", sample):
        unit = "ms" if len(sample) == 13 else "s"
        return pd.to_datetime(s.astype("int64"), unit=unit)
    return pd.to_datetime(s, format="mixed", dayfirst=False)


def detect_and_load(path: str | Path, tz: str | None = None) -> Detected:
    path = Path(path)
    text = _read_text(path)
    delim, has_header = _sniff(text)
    df = pd.read_csv(io.StringIO(text), sep=delim, header=0 if has_header else None, engine="python")
    df = df.dropna(axis=1, how="all")
    fmt = "generic"

    if not has_header:
        n = df.shape[1]
        first = str(df.iloc[0, 0])
        if re.match(r"^\d{8} \d{6}$", first):  # HistData ASCII M1
            fmt = "histdata"
            df.columns = ["datetime", "open", "high", "low", "close", "volume"][:n]
            dt = _parse_dt(df["datetime"])
        elif re.match(r"^\d{8} \d{9}$", first):  # HistData ticks: YYYYMMDD HHMMSSmmm,bid,ask,vol
            fmt = "histdata"
            df.columns = ["datetime", "bid", "ask", "volume"][:n]
            dt = pd.to_datetime(df["datetime"].astype(str), format="%Y%m%d %H%M%S%f")
        elif re.match(r"^\d{4}[.\-/]\d{2}[.\-/]\d{2}$", first) and n >= 6:  # MT4: date,time,o,h,l,c[,v]
            fmt = "mt4"
            df.columns = ["date", "time", "open", "high", "low", "close", "volume"][:n] + [f"x{i}" for i in range(max(0, n - 7))]
            dt = _parse_dt(df["date"].astype(str) + " " + df["time"].astype(str))
        else:
            df.columns = ["datetime", "open", "high", "low", "close", "volume"][:n] + [f"x{i}" for i in range(max(0, n - 6))]
            dt = _parse_dt(df["datetime"])
    else:
        cols = list(df.columns)
        if any(c.startswith("<") for c in cols):
            fmt = "mt5"
        elif any(c.strip().lower() in ("gmt time", "local time") for c in cols):
            fmt = "dukascopy"
        dcol, tcol, dtcol = _col(cols, "date"), _col(cols, "time"), _col(cols, "datetime")
        if dcol and tcol and dcol != tcol:
            dt = _parse_dt(df[dcol].astype(str) + " " + df[tcol].astype(str))
        else:
            dt = _parse_dt(df[dtcol])
        ren = {}
        for k in ("open", "high", "low", "close", "volume", "bid", "ask"):
            c = _col(cols, k)
            if c is not None and c not in (dcol, tcol, dtcol):
                ren[c] = k
        df = df.rename(columns=ren)

    if "close" not in df.columns and "bid" in df.columns:  # tick data -> M1 bars on mid
        fmt = "tick" if fmt == "generic" else fmt
        mid = (df["bid"].astype(float) + df.get("ask", df["bid"]).astype(float)) / 2
        s = pd.Series(mid.to_numpy(), index=pd.DatetimeIndex(dt))
        bars = s.resample("1min").ohlc().dropna()
        bars["volume"] = s.resample("1min").count()
        out = bars
    else:
        out = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]].astype(float)
        out.index = pd.DatetimeIndex(dt)
        if "volume" not in out:
            out["volume"] = 0.0

    zone = tz or FORMAT_TZ.get(fmt, "UTC")
    idx = out.index
    idx = idx.tz_localize(zone) if idx.tz is None else idx
    out.index = idx.tz_convert("UTC").as_unit("ns")
    out.index.name = "time"
    return Detected(fmt, symbol_from_name(path.name), zone, out)


def load_raw_dir(raw_dir: str | Path = "data/raw", tz_overrides: dict[str, str] | None = None) -> dict[str, list[Detected]]:
    """Load every CSV/TXT/ZIP in raw_dir, grouped by symbol."""
    res: dict[str, list[Detected]] = {}
    for p in sorted(Path(raw_dir).glob("**/*")):
        if p.suffix.lower() not in (".csv", ".txt", ".zip") or not p.is_file():
            continue
        try:
            d = detect_and_load(p, (tz_overrides or {}).get(p.name))
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse %s: %s", p, e)
            continue
        if d.symbol is None:
            log.warning("no symbol in filename %s; skipped", p.name)
            continue
        log.info("loaded %s as %s (%s, %d bars)", p.name, d.symbol, d.fmt, len(d.df))
        res.setdefault(d.symbol, []).append(d)
    return res
