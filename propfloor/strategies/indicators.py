"""Vectorised / numba indicators on numpy float arrays. NaN during warm-up."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


@njit(cache=True)
def _ewm(x, alpha):
    out = np.empty_like(x)
    out[:] = np.nan
    s = np.nan
    for i in range(x.shape[0]):
        v = x[i]
        if np.isnan(v):
            out[i] = s
            continue
        s = v if np.isnan(s) else alpha * v + (1 - alpha) * s
        out[i] = s
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = _ewm(np.asarray(x, dtype=np.float64), 2.0 / (n + 1))
    out[: n - 1] = np.nan
    return out


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    out = _ewm(np.asarray(x, dtype=np.float64), 1.0 / n)
    out[: n - 1] = np.nan
    return out


def stdev(x, n):
    return pd.Series(x).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def true_range(h, l, c):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def atr(h, l, c, n=14):
    return wilder(true_range(h, l, c), n)


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = wilder(np.where(d > 0, d, 0.0), n)
    dn = wilder(np.where(d < 0, -d, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = up / dn
        out = 100 - 100 / (1 + rs)
    out[(dn == 0) & ~np.isnan(up)] = 100.0
    return out


def macd(c, fast=12, slow=26, signal=9):
    m = ema(c, fast) - ema(c, slow)
    s = ema(np.nan_to_num(m, nan=0.0), signal)
    s[np.isnan(m)] = np.nan
    return m, s, m - s


def bollinger(c, n=20, k=2.0):
    mid = sma(c, n)
    sd = stdev(c, n)
    return mid - k * sd, mid, mid + k * sd


def stochastic(h, l, c, k=14, d=3):
    hh = rolling_max(h, k)
    ll = rolling_min(l, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        kk = 100 * (c - ll) / (hh - ll)
    return kk, sma(kk, d)


def adx(h, l, c, n=14):
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = wilder(true_range(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * wilder(pdm, n) / tr
        ndi = 100 * wilder(ndm, n) / tr
        dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
    return wilder(np.nan_to_num(dx, nan=0.0), n), pdi, ndi


def rolling_max(x, n):
    return pd.Series(x).rolling(n, min_periods=n).max().to_numpy()


def rolling_min(x, n):
    return pd.Series(x).rolling(n, min_periods=n).min().to_numpy()


def roc(c, n):
    return c / np.r_[np.full(n, np.nan), c[:-n]] - 1


def zscore(c, n):
    return (c - sma(c, n)) / stdev(c, n)


def percentile_rank(x, n):
    """Rolling percentile rank of the last value within the window (0..1)."""
    return pd.Series(x).rolling(n, min_periods=n).rank(pct=True).to_numpy()


@njit(cache=True)
def streak(c):
    """Signed count of consecutive up (+) / down (-) closes."""
    out = np.zeros(c.shape[0])
    for i in range(1, c.shape[0]):
        if c[i] > c[i - 1]:
            out[i] = out[i - 1] + 1 if out[i - 1] > 0 else 1
        elif c[i] < c[i - 1]:
            out[i] = out[i - 1] - 1 if out[i - 1] < 0 else -1
    return out


@njit(cache=True)
def swing_levels(h, l, k):
    """Most recent confirmed swing high/low (fractal with k bars each side), known k bars later."""
    n = h.shape[0]
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    lh = np.nan
    ll = np.nan
    for i in range(2 * k, n):
        j = i - k
        ish = True
        isl = True
        for m in range(j - k, j + k + 1):
            if m == j:
                continue
            if h[m] >= h[j]:
                ish = False
            if l[m] <= l[j]:
                isl = False
        if ish:
            lh = h[j]
        if isl:
            ll = l[j]
        sh[i] = lh
        sl[i] = ll
    return sh, sl


def shift(x, n=1):
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return x
    return np.r_[np.full(n, np.nan), x[:-n]]
