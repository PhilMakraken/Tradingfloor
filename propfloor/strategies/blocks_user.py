"""Entry blocks from the user XAUUSD idea workbooks (research/user_ideas/mapping.md, "Top 10 genuinely
untested ideas"). Only ideas without an existing block are implemented here:

dollar_residual_xau - XAU n-bar return minus beta x USD-basket return (beta fitted on in-sample bars only);
                      fade / follow the standardised residual when it crosses +-thr
ar_innovation       - rolling AR(p) one-step forecast fitted on the previous `window` returns; trade the
                      innovation z: fade it when ADX is low, follow it when ADX is high
gk_vol_surprise     - Garman-Klass bar vol / EWMA forecast (strictly past): surprise follow on a close beyond
                      the prior bar's extreme, or compression-armed box breakout
kalman_innovation   - local linear trend Kalman filter on log price (noise from past returns only); fade a
                      large innovation that goes against the predicted slope
daily_pivot         - prior-day floor pivots: reclaim P / S1 after an intraday dip below (shorts mirror at P / R1)
flag_continuation   - impulse >= m ATR in imp_n bars, then a <= 50% retrace, narrowing flag; close through edge
failed_failure      - range break, false re-entry, then reclaim of the original break level within k bars
anchored_vwap       - VWAP anchored at the last confirmed swing (or day open); first pullback touch that holds
cci_resume          - CCI crosses zero in the EMA-trend direction after an opposite excursion beyond +-ext
acceptance_break    - the mapping's `confirm_closes` idea as its own block (so existing breakout genomes keep
                      their keys): N consecutive closes beyond a level; level=vwap is the `vwap_reclaim` mode

`kalman_slope` (blocks_ext) already trades the Kalman slope sign; kalman_innovation trades the innovation instead.
Every signal at bar t uses bars <= t only; intermarket data comes from ctx.peers (causally forward-filled) and the
only fitted statistic (dollar_residual_xau beta) uses ctx.train_mask bars only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from propfloor.strategies import indicators as ind
from propfloor.strategies.blocks import Ctx, _b, _s, entry

USD_PEERS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "USDCHF"]
LN2 = np.log(2.0)


# ------------------------------------------------------------------ helpers
def _first(a) -> np.ndarray:
    a = _b(a)
    return a & ~_b(_s(a.astype(float)))


def _rstd(a: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(a).rolling(w, min_periods=w // 2).std().to_numpy()


def usd_basket_ret(x: Ctx, n: int) -> np.ndarray | None:
    """n-bar log return of a USD index proxy: mean of +log(USDxxx) / -log(xxxUSD) n-bar returns over the
    available USD peers (needs >= 3 finite components on a bar). None when fewer than 3 peers exist."""
    def f():
        comps = []
        for s in USD_PEERS:
            c = x.peer_close(s)
            if c is None or s == x.symbol:
                continue
            lc = np.log(c)
            r = lc - _s(lc, n)
            comps.append(r if s.startswith("USD") else -r)
        if len(comps) < 3:
            return None
        m = np.vstack(comps)
        fin = np.isfinite(m)
        cnt = fin.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(fin, m, 0.0).sum(axis=0) / cnt
        return np.where(cnt >= 3, out, np.nan)
    return x.get(("usd_basket", n), f)


def dollar_residual_z(x: Ctx, n: int) -> np.ndarray | None:
    """Standardised residual of XAU n-bar return on the USD basket n-bar return (beta fitted in-sample)."""
    def f():
        u = usd_basket_ret(x, n)
        if u is None:
            return None
        lc = np.log(x.c)
        g = lc - _s(lc, n)
        ok = x.train_mask & np.isfinite(g) & np.isfinite(u)
        if ok.sum() < 200:
            return None
        beta = float((g[ok] * u[ok]).sum() / max((u[ok] ** 2).sum(), 1e-18))
        e = g - beta * u
        with np.errstate(divide="ignore", invalid="ignore"):
            z = e / _rstd(e, 500)
        return np.where(np.isfinite(z), z, np.nan)
    return x.get(("dres_z", n), f)


# ------------------------------------------------------------------ numba kernels
@njit(cache=True)
def _ar_innov(r, order, w):
    """z of r[t] vs an AR(order) forecast (OLS, no intercept) fitted on the targets r[t-w..t-1] with their own
    lags; sliding sums, so O(n * order^2). NaN while the window (or its lags) holds a NaN."""
    n = r.shape[0]
    z = np.full(n, np.nan)
    xtx = np.zeros((order, order))
    xty = np.zeros(order)
    yy = 0.0
    nbad = 0  # targets in the window with a NaN target or lag
    for t in range(order, n):
        # the window for bar t holds targets s = t-w .. t-1: add s = t-1, drop s = t-1-w
        for sgn, s in ((1.0, t - 1), (-1.0, t - 1 - w)):
            if s < order:
                continue
            bad = np.isnan(r[s])
            for i in range(order):
                bad = bad or np.isnan(r[s - 1 - i])
            if bad:
                nbad += 1 if sgn > 0 else -1
                continue
            yy += sgn * r[s] * r[s]
            for i in range(order):
                ri = r[s - 1 - i]
                xty[i] += sgn * ri * r[s]
                for j in range(order):
                    xtx[i, j] += sgn * ri * r[s - 1 - j]
        if t < w + order or nbad > 0 or np.isnan(r[t]):
            continue
        a = xtx.copy()
        for i in range(order):
            a[i, i] += 1e-18
        beta = np.linalg.solve(a, xty)
        sse = yy
        for i in range(order):
            sse -= 2 * beta[i] * xty[i]
            for j in range(order):
                sse += beta[i] * xtx[i, j] * beta[j]
        if sse <= 0:
            continue
        sig = np.sqrt(sse / (w - order))
        pr = 0.0
        bad = False
        for i in range(order):
            if np.isnan(r[t - 1 - i]):
                bad = True
            pr += beta[i] * r[t - 1 - i]
        if not bad:
            z[t] = (r[t] - pr) / sig
    return z


@njit(cache=True)
def _kalman_innov(y, rv, q_ratio):
    """Local linear trend filter on y with measurement noise rv[t] (known before bar t) and level/slope noise
    q_ratio * rv[t]. Returns the standardised innovation and the PREDICTED slope (both before the update)."""
    n = y.shape[0]
    z = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    started = False
    x0 = 0.0
    x1 = 0.0
    p00 = 0.0
    p01 = 0.0
    p11 = 0.0
    for i in range(n):
        r = rv[i]
        if np.isnan(y[i]) or np.isnan(r) or r <= 0:
            continue
        if not started:
            x0, x1 = y[i], 0.0
            p00, p01, p11 = r, 0.0, r
            started = True
            continue
        q = q_ratio * r
        x0 = x0 + x1
        p00 = p00 + 2 * p01 + p11 + q
        p01 = p01 + p11
        p11 = p11 + q
        s = p00 + r
        v = y[i] - x0
        z[i] = v / np.sqrt(s)
        slope[i] = x1
        k0 = p00 / s
        k1 = p01 / s
        x0 += k0 * v
        x1 += k1 * v
        p11 = p11 - k1 * p01
        p01 = p01 - k0 * p01
        p00 = p00 - k0 * p00
    return z, slope


@njit(cache=True)
def _failed_failure(c, hh, ll, k):
    """Break of the prior range, close back inside within k bars, then close beyond the break level again
    within k bars of the failure."""
    n = c.shape[0]
    lo = np.zeros(n, np.bool_)
    sh = np.zeros(n, np.bool_)
    us, ua, ul = 0, 0, 0.0  # long-side state, age, break level
    ds, da, dl = 0, 0, 0.0
    for t in range(1, n):
        # ---- up break -> failure -> reclaim
        if us == 1:
            ua += 1
            if ua > k:
                us = 0
            elif c[t] < ul:
                us, ua = 2, 0
        elif us == 2:
            ua += 1
            if ua > k:
                us = 0
            elif c[t] > ul:
                lo[t] = True
                us = 0
        if us == 0 and not lo[t] and not np.isnan(hh[t]) and c[t] > hh[t] and c[t - 1] <= hh[t]:
            us, ua, ul = 1, 0, hh[t]
        # ---- down break -> failure -> reclaim
        if ds == 1:
            da += 1
            if da > k:
                ds = 0
            elif c[t] > dl:
                ds, da = 2, 0
        elif ds == 2:
            da += 1
            if da > k:
                ds = 0
            elif c[t] < dl:
                sh[t] = True
                ds = 0
        if ds == 0 and not sh[t] and not np.isnan(ll[t]) and c[t] < ll[t] and c[t - 1] >= ll[t]:
            ds, da, dl = 1, 0, ll[t]
    return lo, sh


@njit(cache=True)
def _avwap_touch(h, l, c, cpv, cv, anchor, trend_ok, below):
    """First touch since the anchor of the VWAP anchored at bar anchor[t] (-1 = none) that holds.
    below=True: long side (low touches, close above). Returns (signal, avwap)."""
    n = c.shape[0]
    sig = np.zeros(n, np.bool_)
    av = np.full(n, np.nan)
    cur = -1
    touched = False
    for t in range(n):
        a = int(anchor[t])
        if a < 0:
            continue
        if a != cur:
            cur = a
            touched = False
        pv0 = cpv[a - 1] if a > 0 else 0.0
        v0 = cv[a - 1] if a > 0 else 0.0
        den = cv[t] - v0
        if den <= 0:
            continue
        av[t] = (cpv[t] - pv0) / den
        if t <= a or touched:
            continue
        if below:
            if l[t] <= av[t]:
                touched = True
                if c[t] > av[t] and trend_ok[t]:
                    sig[t] = True
        else:
            if h[t] >= av[t]:
                touched = True
                if c[t] < av[t] and trend_ok[t]:
                    sig[t] = True
    return sig, av


@njit(cache=True)
def _mean_abs_dev(x, n):
    m = x.shape[0]
    out = np.full(m, np.nan)
    for t in range(n - 1, m):
        s = 0.0
        for j in range(t - n + 1, t + 1):
            s += x[j]
        mu = s / n
        d = 0.0
        for j in range(t - n + 1, t + 1):
            d += abs(x[j] - mu)
        out[t] = d / n
    return out


@njit(cache=True)
def _cci_resume(cci, up_ok, dn_ok, ext):
    """Zero cross of CCI after the preceding opposite-sign run reached beyond -ext / +ext."""
    n = cci.shape[0]
    lo = np.zeros(n, np.bool_)
    sh = np.zeros(n, np.bool_)
    run_min = 0.0
    run_max = 0.0
    for t in range(1, n):
        a, b = cci[t - 1], cci[t]
        if np.isnan(a) or np.isnan(b):
            run_min = 0.0
            run_max = 0.0
            continue
        if a < 0:
            run_min = min(run_min, a)
        else:
            run_max = max(run_max, a)
        if a < 0 <= b:
            if run_min <= -ext and up_ok[t]:
                lo[t] = True
            run_min = 0.0
        elif a >= 0 > b:
            if run_max >= ext and dn_ok[t]:
                sh[t] = True
            run_max = 0.0
    return lo, sh


# ================================================================== blocks
@entry("dollar_residual_xau", "intermarket", n=("int", 2, 24), thr=("float", 1.5, 3.0),
       mode=("choice", ["fade", "follow"]))
def _u1(x: Ctx, p):
    """XAUUSD only. Residual z > thr: gold rich vs the USD basket -> fade = short, follow = long."""
    z0 = np.zeros(x.n, bool)
    if x.symbol != "XAUUSD":
        return z0, z0
    z = dollar_residual_z(x, int(p["n"]))
    if z is None:
        return z0, z0
    t = p["thr"]
    rich, cheap = _first(z > t), _first(z < -t)
    return (cheap, rich) if p["mode"] == "fade" else (rich, cheap)


@entry("ar_innovation", "statistical", order=("int", 1, 3), window=("choice", [50, 100, 200]), k=("float", 1.5, 3.0),
       adx_thr=("float", 15, 30))
def _u2(x: Ctx, p):
    """|AR innovation z| > k: fade it when ADX < adx_thr, follow it otherwise."""
    o, w = int(p["order"]), int(p["window"])
    z = x.get(("ar_innov", o, w), lambda: _ar_innov(np.r_[np.nan, np.diff(np.log(x.c))], o, w))
    k = p["k"]
    trend = _b(x.adx()[0] >= p["adx_thr"])
    up, dn = _b(z > k), _b(z < -k)
    return _b((up & trend) | (dn & ~trend)), _b((dn & trend) | (up & ~trend))


@entry("gk_vol_surprise", "volatility", span=("int", 10, 50), k=("float", 1.5, 3.0),
       mode=("choice", ["surprise", "compression"]))
def _u3(x: Ctx, p):
    """ratio = sqrt(GK var / EWMA forecast of GK var from bars <= t-1). surprise: ratio > k and close beyond the
    prior bar's high/low. compression: a bar with ratio < 1/k within the last 5 bars arms a breakout of the
    prior 5-bar box (first close through it)."""
    def gk():
        with np.errstate(divide="ignore", invalid="ignore"):
            v = 0.5 * np.log(x.h / x.l) ** 2 - (2 * LN2 - 1) * np.log(x.c / x.o) ** 2
        return np.where(np.isfinite(v), np.maximum(v, 0.0), np.nan)
    g = x.get("gk_var", gk)
    span = int(p["span"])

    def ratio():
        fc = _s(ind.ema(np.nan_to_num(g), span))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(g / fc)
        return np.where(np.isfinite(r), r, np.nan)
    r = x.get(("gk_ratio", span), ratio)
    k = p["k"]
    if p["mode"] == "surprise":
        big = _b(r > k)
        return _b(big & (x.c > _s(x.h))), _b(big & (x.c < _s(x.l)))
    calm = _b(r < 1.0 / k).astype(float)
    armed = _b(_s(ind.rolling_max(calm, 5)) > 0)
    hh, ll = _s(ind.rolling_max(x.h, 5)), _s(ind.rolling_min(x.l, 5))
    return _first(armed & (x.c > hh)), _first(armed & (x.c < ll))


@entry("kalman_innovation", "mean_reversion", q_ratio=("choice", [1e-3, 1e-2, 1e-1, 1.0]), k=("float", 1.5, 3.0))
def _u4(x: Ctx, p):
    """Innovation z > k while the predicted slope is down -> short (fade the counter-trend spike); mirror long."""
    def f():
        y = np.log(x.c)
        d = np.r_[np.nan, np.diff(y)]
        rv = _s(ind.ema(np.nan_to_num(d * d), 100))  # noise level from bars <= t-1
        rv[: 100] = np.nan
        return _kalman_innov(y, rv, float(p["q_ratio"]))
    z, slope = x.get(("kal_innov", p["q_ratio"]), f)
    k = p["k"]
    return _b((z < -k) & (slope > 0)), _b((z > k) & (slope < 0))


@entry("daily_pivot", "structure", level=("choice", ["P", "S1R1"]), min_dip_atr=("float", 0.0, 0.5))
def _u5(x: Ctx, p):
    """Prior-day floor pivot P=(H+L+C)/3, S1=2P-H, R1=2P-L. Long: the day opened above the level, today's low
    went at least min_dip_atr ATR below it and this bar's close reclaims it (prev close <= level). Short mirror."""
    ph, pl, pc, do = x.day_hl()
    piv = (ph + pl + pc) / 3
    lv_l, lv_s = (piv, piv) if p["level"] == "P" else (2 * piv - ph, 2 * piv - pl)

    def dayx():
        s = pd.DataFrame({"d": x.day_id, "h": x.h, "l": x.l})
        g = s.groupby("d")
        return g["l"].cummin().to_numpy(), g["h"].cummax().to_numpy()
    dlo, dhi = x.get("day_running_hl", dayx)
    same_day = np.r_[False, x.day_id[1:] == x.day_id[:-1]]
    t = p["min_dip_atr"] * x.atr()
    lo = same_day & (do > lv_l) & (dlo < lv_l - t) & (_s(x.c) <= lv_l) & (x.c > lv_l)
    sh = same_day & (do < lv_s) & (dhi > lv_s + t) & (_s(x.c) >= lv_s) & (x.c < lv_s)
    return _b(lo), _b(sh)


@entry("flag_continuation", "breakout", imp_n=("int", 3, 8), m=("float", 1.0, 2.5), flag_n=("int", 3, 10))
def _u6(x: Ctx, p):
    """Impulse: close move over imp_n bars >= m ATR, ending flag_n+1 bars ago. Flag (the flag_n bars before this
    one): stays within 50% retrace, makes no new extreme beyond the impulse, and its second half has a smaller
    range than the first. Entry: close through the flag high (bull) / low (bear)."""
    n, f = int(p["imp_n"]), int(p["flag_n"])
    h1 = f // 2
    c_end = _s(x.c, f + 1)
    imp = c_end - _s(x.c, f + 1 + n)
    a = _s(x.atr(), f + 1)
    imp_hi, imp_lo = _s(ind.rolling_max(x.h, n + 1), f + 1), _s(ind.rolling_min(x.l, n + 1), f + 1)
    fh, fl = _s(ind.rolling_max(x.h, f)), _s(ind.rolling_min(x.l, f))
    r1 = _s(ind.rolling_max(x.h, h1), f - h1 + 1) - _s(ind.rolling_min(x.l, h1), f - h1 + 1)
    r2 = _s(ind.rolling_max(x.h, f - h1)) - _s(ind.rolling_min(x.l, f - h1))
    narrow = r2 <= r1
    bull = (imp >= p["m"] * a) & (fl >= c_end - 0.5 * imp) & (fh <= imp_hi) & narrow & (x.c > fh)
    bear = (imp <= -p["m"] * a) & (fh <= c_end - 0.5 * imp) & (fl >= imp_lo) & narrow & (x.c < fl)
    return _b(bull), _b(bear)


@entry("failed_failure", "breakout", n=("int", 10, 60), k=("int", 2, 8))
def _u7(x: Ctx, p):
    """Second-order trap: n-bar range break, close back inside within k bars (false break), then a close beyond
    the original break level within k bars -> enter in the original break direction."""
    n, k = int(p["n"]), int(p["k"])

    def f():
        hh, ll = _s(ind.rolling_max(x.h, n)), _s(ind.rolling_min(x.l, n))
        return _failed_failure(x.c, hh, ll, k)
    return x.get(("ffail", n, k), f)


@entry("anchored_vwap", "trend", anchor=("choice", ["swing", "day"]), k=("int", 2, 5), trend=("choice", [50, 100, 200]))
def _u8(x: Ctx, p):
    """VWAP anchored at the last confirmed swing low (longs) / high (shorts), or at the FX-day open. Long on the
    first bar since the anchor whose low touches the AVWAP and closes above it, with close > EMA(trend)."""
    tp = (x.h + x.l + x.c) / 3
    v = np.where(x.v > 0, x.v, 1.0)
    cpv, cv = np.cumsum(tp * v), np.cumsum(v)
    m = x.ma(int(p["trend"]))
    up_ok, dn_ok = _b(x.c > m), _b(x.c < m)
    if p["anchor"] == "day":
        start = np.r_[True, x.day_id[1:] != x.day_id[:-1]]
        a = np.maximum.accumulate(np.where(start, np.arange(x.n), -1)).astype(float)
        al = ash = a
    else:
        _, _, shi, sli = x.get(("struct", int(p["k"])), lambda: ind.structure(x.h, x.l, int(p["k"])))
        al, ash = sli, shi
    lo, _ = _avwap_touch(x.h, x.l, x.c, cpv, cv, al, up_ok, True)
    sh, _ = _avwap_touch(x.h, x.l, x.c, cpv, cv, ash, dn_ok, False)
    return lo, sh


@entry("cci_resume", "trend", n=("int", 14, 30), ext=("float", 80, 150), trend=("choice", [100, 200]))
def _u9(x: Ctx, p):
    """CCI(n) crosses zero in the direction of the EMA(trend) slope (vs 5 bars ago) after the preceding run on the
    other side of zero went beyond -ext (longs) / +ext (shorts)."""
    n = int(p["n"])

    def cci():
        tp = (x.h + x.l + x.c) / 3
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (tp - ind.sma(tp, n)) / (0.015 * _mean_abs_dev(tp, n))
        return np.where(np.isfinite(r), r, np.nan)
    c = x.get(("cci", n), cci)
    m = x.ma(int(p["trend"]))
    slope = m - _s(m, 5)
    return _cci_resume(c, _b(slope > 0), _b(slope < 0), float(p["ext"]))


@entry("acceptance_break", "breakout", level=("choice", ["donchian", "prev_day", "swing", "vwap"]),
       confirm_closes=("int", 2, 4))
def _u10(x: Ctx, p):
    """confirm_closes consecutive closes beyond a level, the close before them not beyond. donchian (20-bar),
    prev_day (PDH/PDL) and swing (k=3) levels are frozen at the first close beyond; vwap (= vwap_reclaim mode)
    compares each close with the session VWAP of its own bar."""
    N = int(p["confirm_closes"])
    lv = p["level"]
    if lv == "vwap":
        vw, _ = x.get("vwap", lambda: ind.session_vwap(x.h, x.l, x.c, x.v, x.day_id))
        up = dn = vw
        a_up, a_dn = _b(x.c > up), _b(x.c < dn)
        lo = np.ones(x.n, bool)
        sh = np.ones(x.n, bool)
        for j in range(N):
            lo &= _b(_s(a_up.astype(float), j))
            sh &= _b(_s(a_dn.astype(float), j))
        lo &= ~_b(_s(a_up.astype(float), N)) & np.isfinite(_s(up, N))
        sh &= ~_b(_s(a_dn.astype(float), N)) & np.isfinite(_s(dn, N))
        return _b(lo), _b(sh)
    if lv == "donchian":
        up, dn = _s(ind.rolling_max(x.h, 20)), _s(ind.rolling_min(x.l, 20))
    elif lv == "prev_day":
        ph, pl, _, _ = x.day_hl()
        up, dn = ph, pl
    else:
        up, dn = x.swings(3)
    fu, fd = _s(up, N - 1), _s(dn, N - 1)  # level at the first close beyond
    lo = (_s(x.c, N) <= _s(up, N)) & (_s(x.c, N - 1) > fu)
    sh = (_s(x.c, N) >= _s(dn, N)) & (_s(x.c, N - 1) < fd)
    for j in range(N - 1):
        lo &= _s(x.c, j) > fu
        sh &= _s(x.c, j) < fd
    return _b(lo), _b(sh)
