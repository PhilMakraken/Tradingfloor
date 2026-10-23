"""Synthetic FX-like M1 bars for smoke tests and offline development.

Regime-switching stochastic volatility, intraday/session seasonality, weekend gaps and mild
short-horizon autocorrelation. NEVER used to validate strategies (store marks it synthetic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from propfloor.data.clean import is_fx_closed


def make_bars(start="2019-01-01", end="2021-01-01", seed: int = 0, price0: float = 1.15,
              annual_vol: float = 0.08, freq: str = "1min") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    idx = idx[~is_fx_closed(idx)]
    n = len(idx)
    minutes_per_year = 252 * 1440
    step = pd.Timedelta(freq).total_seconds() / 60
    base_sigma = annual_vol / np.sqrt(minutes_per_year / step)
    # session seasonality (London/NY busier)
    h = idx.hour.to_numpy()
    season = np.where((h >= 7) & (h < 17), 1.4, np.where((h >= 12) & (h < 21), 1.2, 0.6))
    # regime switching log-vol (AR(1))
    lv = np.zeros(n)
    eps = rng.standard_normal(n)
    phi = 0.9995
    for i in range(1, n):
        lv[i] = phi * lv[i - 1] + 0.02 * eps[i]
    sigma = base_sigma * season * np.exp(lv)
    z = rng.standard_t(5, n) / np.sqrt(5 / 3)
    r = sigma * z
    r[1:] += 0.03 * r[:-1]  # slight momentum
    gaps = np.r_[False, np.diff(idx.asi8) > 3600 * 1e9]
    r[gaps] += rng.normal(0, base_sigma * 30, gaps.sum())
    close = price0 * np.exp(np.cumsum(r))
    open_ = np.r_[price0, close[:-1]]
    wick = np.abs(rng.normal(0, 1, (2, n))) * sigma * close * 0.7
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    vol = rng.poisson(50 * season).astype(float)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx.as_unit("ns"))
    df.index.name = "time"
    return df
