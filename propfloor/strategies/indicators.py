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


# ---------------------------------------------------------------- extended indicators
@njit(cache=True)
def _supertrend(h, l, c, atr, mult):
    n = c.shape[0]
    dirn = np.zeros(n)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(atr[i]):
            continue
        mid = (h[i] + l[i]) / 2
        bu = mid + mult * atr[i]
        bl = mid - mult * atr[i]
        if np.isnan(up[i - 1]):
            up[i], dn[i], dirn[i] = bu, bl, 1.0
            continue
        up[i] = bu if (bu < up[i - 1] or c[i - 1] > up[i - 1]) else up[i - 1]
        dn[i] = bl if (bl > dn[i - 1] or c[i - 1] < dn[i - 1]) else dn[i - 1]
        if dirn[i - 1] >= 0:
            dirn[i] = -1.0 if c[i] < dn[i] else 1.0
        else:
            dirn[i] = 1.0 if c[i] > up[i] else -1.0
    return dirn


def supertrend(h, l, c, n=10, mult=3.0):
    """+1 / -1 trend direction."""
    return _supertrend(h, l, c, atr(h, l, c, n), mult)


@njit(cache=True)
def psar(h, l, step, max_step):
    """Parabolic SAR direction (+1 long / -1 short)."""
    n = h.shape[0]
    d = np.zeros(n)
    if n < 3:
        return d
    up = True
    sar = l[0]
    ep = h[0]
    af = step
    d[0] = 1.0
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[max(i - 2, 0)])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            sar = max(sar, h[i - 1], h[max(i - 2, 0)])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_step)
        d[i] = 1.0 if up else -1.0
    return d


def keltner(h, l, c, n=20, mult=2.0):
    mid = ema(c, n)
    a = atr(h, l, c, n)
    return mid - mult * a, mid, mid + mult * a


def ichimoku(h, l, c, tenkan=9, kijun=26, senkou=52):
    """Returns tenkan, kijun, cloud_top, cloud_bottom (cloud shifted forward by kijun, i.e. causal)."""
    t = (rolling_max(h, tenkan) + rolling_min(l, tenkan)) / 2
    k = (rolling_max(h, kijun) + rolling_min(l, kijun)) / 2
    a = shift((t + k) / 2, kijun)
    b = shift((rolling_max(h, senkou) + rolling_min(l, senkou)) / 2, kijun)
    return t, k, np.fmax(a, b), np.fmin(a, b)


def session_vwap(h, l, c, v, day_id):
    """Session VWAP and volume-weighted stdev of typical price since session start."""
    tp = (h + l + c) / 3
    v = np.where(v > 0, v, 1.0)
    df = pd.DataFrame({"d": day_id, "pv": tp * v, "v": v, "pv2": tp * tp * v})
    g = df.groupby("d")
    cpv, cv, cpv2 = g["pv"].cumsum().to_numpy(), g["v"].cumsum().to_numpy(), g["pv2"].cumsum().to_numpy()
    vw = cpv / cv
    sd = np.sqrt(np.maximum(cpv2 / cv - vw * vw, 0))
    return vw, sd


@njit(cache=True)
def kalman_trend(c, q, r):
    """Local linear trend Kalman filter; returns (level, slope)."""
    n = c.shape[0]
    lv = np.zeros(n)
    sl = np.zeros(n)
    x0, x1 = c[0], 0.0
    p00, p01, p11 = 1.0, 0.0, 1.0
    for i in range(n):
        # predict
        x0 = x0 + x1
        p00 = p00 + 2 * p01 + p11 + q
        p01 = p01 + p11
        p11 = p11 + q
        # update
        s = p00 + r
        k0 = p00 / s
        k1 = p01 / s
        y = c[i] - x0
        x0 += k0 * y
        x1 += k1 * y
        p11 = p11 - k1 * p01
        p01 = p01 - k0 * p01
        p00 = p00 - k0 * p00
        lv[i] = x0
        sl[i] = x1
    return lv, sl


def linreg(c, n):
    """Rolling OLS line value at the last point, slope and residual stdev."""
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    s = pd.Series(c)
    ym = s.rolling(n).mean()
    # sum_j x_j * y_{i-n+1+j} via convolution (x reversed), then centre
    sxy_raw = np.convolve(np.nan_to_num(c), x[::-1], mode="full")[: len(c)]
    sxy = sxy_raw - n * xm * ym.to_numpy()
    slope = sxy / sxx
    slope[: n - 1] = np.nan
    line = ym.to_numpy() + slope * (n - 1 - xm)
    # residual std ~ sqrt(var(y) - slope^2 var(x))
    var_y = s.rolling(n).var(ddof=0).to_numpy()
    res = np.sqrt(np.maximum(var_y - slope ** 2 * sxx / n, 0))
    return line, slope, res


def autocorr(c, n, lag=1):
    r = pd.Series(np.diff(np.log(c), prepend=np.log(c[0])))
    return r.rolling(n).corr(r.shift(lag)).to_numpy()


def hurst(c, n, k=4):
    """Rolling variance-ratio Hurst estimate: var(k-bar returns) / var(1-bar returns) = k^(2H)."""
    lc = pd.Series(np.log(c))
    r1 = lc.diff()
    rk = lc.diff(k)
    v1 = r1.rolling(n).var()
    vk = rk.rolling(n).var()
    with np.errstate(divide="ignore", invalid="ignore"):
        return (0.5 * np.log(vk / v1) / np.log(k)).to_numpy()


def choppiness(h, l, c, n=14):
    """Choppiness index (high = range/overlap, low = trend)."""
    tr = pd.Series(true_range(h, l, c)).rolling(n).sum().to_numpy()
    rng = rolling_max(h, n) - rolling_min(l, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * np.log10(tr / rng) / np.log10(n)


def clv(h, l, c):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(((c - l) - (h - c)) / (h - l))


@njit(cache=True)
def structure(h, l, k):
    """Last three confirmed swing highs/lows (fractals, known k bars later) per bar.
    Returns sh[n,3], sl[n,3] with column 0 = most recent, and the bar index of the most recent ones."""
    n = h.shape[0]
    sh = np.full((n, 3), np.nan)
    sl = np.full((n, 3), np.nan)
    shi = np.full(n, -1.0)
    sli = np.full(n, -1.0)
    a0 = a1 = a2 = np.nan
    b0 = b1 = b2 = np.nan
    ai = -1.0
    bi = -1.0
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
            a2, a1, a0, ai = a1, a0, h[j], j
        if isl:
            b2, b1, b0, bi = b1, b0, l[j], j
        sh[i, 0], sh[i, 1], sh[i, 2] = a0, a1, a2
        sl[i, 0], sl[i, 1], sl[i, 2] = b0, b1, b2
        shi[i] = ai
        sli[i] = bi
    return sh, sl, shi, sli


@njit(cache=True)
def level_touches(l_or_h, swings_idx_level, lookback, tol):
    """For each bar, number of confirmed swing levels in the last `lookback` bars within tol[i] of price[i]."""
    n = l_or_h.shape[0]
    out = np.zeros(n)
    for i in range(n):
        cnt = 0
        for j in range(max(0, i - lookback), i):
            v = swings_idx_level[j]
            if not np.isnan(v) and abs(v - l_or_h[i]) <= tol[i]:
                cnt += 1
        out[i] = cnt
    return out


@njit(cache=True)
def bucket_tstat(ret_next_bucket, bucket, nb, min_n):
    """Causal per-bucket t-stat of returns: value at i uses only returns observed strictly before i.
    ret_next_bucket[i] is the return of bar i (known at i's close); bucket[i] is bar i's bucket."""
    n = bucket.shape[0]
    s = np.zeros(nb)
    s2 = np.zeros(nb)
    cnt = np.zeros(nb)
    out = np.zeros(n)
    for i in range(n):
        b = bucket[i]
        if cnt[b] >= min_n:
            m = s[b] / cnt[b]
            v = s2[b] / cnt[b] - m * m
            out[i] = m / np.sqrt(v / cnt[b]) if v > 0 else 0.0
        r = ret_next_bucket[i]
        if not np.isnan(r):
            s[b] += r
            s2[b] += r * r
            cnt[b] += 1
    return out
