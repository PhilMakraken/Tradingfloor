"""Cleaning & quality checks for UTC-indexed OHLCV bars."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    n_in: int = 0
    n_out: int = 0
    duplicates: int = 0
    invalid: int = 0
    spikes: int = 0
    weekend_bars: int = 0
    gaps: list[tuple[str, str, float]] = field(default_factory=list)  # (start, end, minutes)
    notes: list[str] = field(default_factory=list)


def is_fx_closed(idx: pd.DatetimeIndex) -> np.ndarray:
    """FX is closed Fri 17:00 -> Sun 17:00 New York time."""
    ny = idx.tz_convert("America/New_York")
    wd, h = ny.weekday, ny.hour
    return ((wd == 4) & (h >= 17)) | (wd == 5) | ((wd == 6) & (h < 17))


def clean_bars(df: pd.DataFrame, spike_k: float = 25.0, gap_minutes: int = 60, bar_minutes: int = 1,
               drop_weekend: bool = True) -> tuple[pd.DataFrame, QualityReport]:
    rep = QualityReport(n_in=len(df))
    df = df.sort_index()
    dup = df.index.duplicated(keep="last")
    rep.duplicates = int(dup.sum())
    df = df[~dup].copy()

    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | df[["open", "high", "low", "close"]].isna().any(axis=1)
    rep.invalid = int(bad.sum())
    df = df[~bad].copy()
    # repair inconsistent high/low
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    # spike filter: bar range or close-to-close move far beyond robust scale
    lr = np.log(df["close"]).diff().abs()
    rng = np.log(df["high"] / df["low"])
    scale = lr.rolling(500, min_periods=50).median().bfill().fillna(lr.median()) + 1e-12
    spikes = (lr > spike_k * scale * 6) & (lr.shift(-1) > spike_k * scale * 6)  # jump that immediately reverts
    spikes |= rng > spike_k * 6 * (rng.rolling(500, min_periods=50).median().bfill().fillna(rng.median()) + 1e-12)
    rep.spikes = int(spikes.sum())
    df = df[~spikes]

    if drop_weekend and len(df):
        wk = is_fx_closed(df.index)
        rep.weekend_bars = int(wk.sum())
        df = df[~wk]

    if len(df) > 1:
        d = df.index.to_series().diff().dt.total_seconds().div(60)
        big = d[d > gap_minutes]
        for t, m in big.items():
            start = df.index[df.index.get_loc(t) - 1]
            closed = is_fx_closed(pd.date_range(start, t, freq=f"{bar_minutes}min", inclusive="neither"))
            open_minutes = (~closed).sum() * bar_minutes
            if open_minutes > gap_minutes:
                rep.gaps.append((str(start), str(t), float(open_minutes)))
    if len(rep.gaps):
        rep.notes.append(f"{len(rep.gaps)} gaps > {gap_minutes} market minutes")
    rep.n_out = len(df)
    return df, rep


TF_RULES = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "D1"}
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample M1 bars to a timeframe (M5..H4, D1). D1 uses the 17:00 New York roll."""
    rule = TF_RULES.get(tf.upper(), tf)
    if rule == "D1":
        shifted = df.copy()
        shifted.index = shifted.index.tz_convert("America/New_York") + pd.Timedelta(hours=7)
        out = shifted.resample("1D").agg(_AGG).dropna()
        out.index = (out.index - pd.Timedelta(hours=7)).tz_convert("UTC")
        return out
    return df.resample(rule, label="left", closed="left").agg(_AGG).dropna()
