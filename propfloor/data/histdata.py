"""HistData.com free 1-minute ASCII bars (EST, no DST). Yearly zips, monthly for the current year.

The site requires a token scraped from the download page, then a POST to get.php.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile

import pandas as pd
import requests

from propfloor.data.loader import FORMAT_TZ

log = logging.getLogger(__name__)

PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{year}{month}"
GET = "https://www.histdata.com/get.php"


def parse_token(html: str) -> str | None:
    m = re.search(r'id="tk"\s+value="([^"]+)"', html) or re.search(r'name="tk"[^>]*value="([^"]+)"', html)
    return m.group(1) if m else None


def parse_ascii(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep=";", header=None, names=["dt", "open", "high", "low", "close", "volume"])
    idx = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S").dt.tz_localize(FORMAT_TZ["histdata"]).dt.tz_convert("UTC")
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = pd.DatetimeIndex(idx).as_unit("ns")
    out.index.name = "time"
    return out


def download_year(pair: str, year: int, month: int | None = None) -> pd.DataFrame:
    pair = pair.lower()
    page = PAGE.format(pair=pair, year=year, month=f"/{month}" if month else "")
    s = requests.Session()
    html = s.get(page, timeout=30).text
    tk = parse_token(html)
    if not tk:
        raise RuntimeError("histdata token not found (site layout changed?)")
    form = {"tk": tk, "date": str(year), "datemonth": f"{year}{month:02d}" if month else str(year),
            "platform": "ASCII", "timeframe": "M1", "fxpair": pair.upper()}
    r = s.post(GET, data=form, headers={"Referer": page}, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return parse_ascii(z.read(name).decode())


def download(pair: str, start: str, end: str | None = None) -> pd.DataFrame:
    y0 = pd.Timestamp(start).year
    y1 = pd.Timestamp(end).year if end else pd.Timestamp.now().year
    frames = []
    for y in range(y0, y1 + 1):
        try:
            frames.append(download_year(pair, y))
        except Exception as e:  # noqa: BLE001 - current year is published monthly
            log.warning("histdata %s %d yearly failed (%s); trying months", pair, y, e)
            for m in range(1, 13):
                try:
                    frames.append(download_year(pair, y, m))
                except Exception:  # noqa: BLE001
                    break
    return pd.concat(frames) if frames else pd.DataFrame()
