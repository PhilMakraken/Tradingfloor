"""Extended building blocks from the strategy catalogues (sections A-L, N, O).

Blocks that already existed in blocks.py are not duplicated (e.g. NR7 = nr_breakout, Asian-range/ORB =
session_range_breakout, Donchian/Turtle = donchian_breakout + exit_mode=channel, RSI extremes in range =
rsi_extreme + adx_ranging filter + exit_mode=rsi50, Z-score = zscore_reversion, streak MR = streak_reversal,
HTF trend + LTF trigger = any entry + mtf_trend filter, H4 vol-regime switching = regime_router).
All signals are causal: they use bars up to and including the signal bar; fills happen at the next open.
Blocks that estimate statistics (H1/H2/L3) use expanding, strictly-past windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from propfloor.data.store import pip_size
from propfloor.strategies import indicators as ind
from propfloor.strategies.blocks import Ctx, _b, _s, entry, filt


# ------------------------------------------------------------------ helpers
def _ffill_event(values: np.ndarray, event: np.ndarray, max_age: int) -> tuple[np.ndarray, np.ndarray]:
    """Carry `values` forward from bars where `event` is True for at most max_age bars.
    Returns (value, age) with age 0 on the event bar, NaN when expired/never."""
    n = len(values)
    idx = np.where(event, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    age = np.arange(n) - last
    ok = (last >= 0) & (age <= max_age)
    val = np.where(ok, np.asarray(values, float)[np.clip(last, 0, None)], np.nan)
    return val, np.where(ok, age, np.nan)


def _local_minutes(x: Ctx, tz: str) -> np.ndarray:
    def f():
        t = x.time.tz_convert(tz)
        return (t.hour * 60 + t.minute).to_numpy()
    return x.get(("lmin", tz), f)


def _first_bar_at(x: Ctx, tz: str, minute: int) -> np.ndarray:
    """First bar of each FX day whose local time-of-day >= minute."""
    m = _local_minutes(x, tz)
    ok = m >= minute
    prev_ok = np.r_[False, ok[:-1]] & (np.r_[-1, x.day_id[:-1]] == x.day_id)
    return ok & ~prev_ok


def _day_open(x: Ctx) -> np.ndarray:
    return x.day_hl()[3]


def _struct(x: Ctx, k: int):
    return x.get(("struct", k), lambda: ind.structure(x.h, x.l, k))


def _sign_ret(x: Ctx, n: int) -> np.ndarray:
    return np.sign(np.nan_to_num(x.c - _s(x.c, n)))


def _fu(up, dn, mode):
    return (_b(up), _b(dn)) if mode == "follow" else (_b(dn), _b(up))


# ================================================================== A. candlestick patterns
@entry("engulfing_at_swing", "candles", body_mult=("float", 1.0, 2.0), lookback=("int", 10, 50))
def _a1(x: Ctx, p):
    b, pb = x.body, _s(x.body)
    big = np.abs(b) > p["body_mult"] * np.abs(pb)
    n = p["lookback"]
    at_low = np.fmin(x.l, _s(x.l)) <= ind.rolling_min(x.l, n) + 1e-12
    at_high = np.fmax(x.h, _s(x.h)) >= ind.rolling_max(x.h, n) - 1e-12
    bull = (pb < 0) & (b > 0) & (x.c >= _s(x.o)) & (x.o <= _s(x.c)) & big & at_low
    bear = (pb > 0) & (b < 0) & (x.c <= _s(x.o)) & (x.o >= _s(x.c)) & big & at_high
    return _b(bull), _b(bear)


@entry("pin_bar_break", "candles", wick_ratio=("float", 2.0, 4.0), close_zone=("float", 0.25, 0.4),
       level=("choice", ["ema20", "ema50", "swing", "any"]))
def _a2(x: Ctx, p):
    body = np.abs(x.body) + 1e-12
    upper = x.h - np.maximum(x.o, x.c)
    lower = np.minimum(x.o, x.c) - x.l
    pos = (x.c - x.l) / (x.rng + 1e-12)
    bull_pin = (lower >= p["wick_ratio"] * body) & (pos >= 1 - p["close_zone"])
    bear_pin = (upper >= p["wick_ratio"] * body) & (pos <= p["close_zone"])
    if p["level"] in ("ema20", "ema50"):
        m = x.ma(20 if p["level"] == "ema20" else 50)
        at_bull = (x.l <= m) & (x.c >= m)
        at_bear = (x.h >= m) & (x.c <= m)
    elif p["level"] == "swing":
        sh, sl = x.swings(3)
        t = 0.3 * x.atr()
        at_bull = np.abs(x.l - sl) < t
        at_bear = np.abs(x.h - sh) < t
    else:
        at_bull = at_bear = np.ones(x.n, bool)
    su = _b(_s((bull_pin & at_bull).astype(float)))
    sd = _b(_s((bear_pin & at_bear).astype(float)))
    return _b(su & (x.c > _s(x.h))), _b(sd & (x.c < _s(x.l)))  # entry on break of the pin bar


@entry("outside_bar_reversal", "candles", range_atr=("float", 1.5, 2.5), trend_n=("int", 3, 10))
def _a4(x: Ctx, p):
    out = (x.h > _s(x.h)) & (x.l < _s(x.l)) & (x.rng > p["range_atr"] * _s(x.atr()))
    prior_up = _s(x.c) > _s(x.c, p["trend_n"] + 1)
    return _b(out & ~prior_up & (x.c > _s(x.h))), _b(out & prior_up & (x.c < _s(x.l)))


@entry("doji_exhaustion", "candles", body_frac=("float", 0.05, 0.15), run=("int", 3, 6))
def _a5(x: Ctx, p):
    doji = np.abs(x.body) <= p["body_frac"] * (x.rng + 1e-12)
    st = x.get("streak", lambda: ind.streak(x.c))
    d_prev = _b(_s(doji.astype(float)))
    run_up = _s(st, 2) >= p["run"]
    run_dn = _s(st, 2) <= -p["run"]
    return _b(d_prev & run_dn & (x.c > _s(x.h))), _b(d_prev & run_up & (x.c < _s(x.l)))


@entry("three_soldiers_pullback", "candles", min_body_atr=("float", 0.3, 1.0), wait=("int", 3, 10))
def _a7(x: Ctx, p):
    a = x.atr()
    up = np.ones(x.n, bool)
    dn = np.ones(x.n, bool)
    for k in range(3):
        up &= _b(_s(x.body, k) > p["min_body_atr"] * _s(a, k))
        dn &= _b(_s(x.body, k) < -p["min_body_atr"] * _s(a, k))
    mid = (_s(x.o, 2) + x.c) / 2
    mu, au = _ffill_event(mid, up, p["wait"])
    md, ad = _ffill_event(mid, dn, p["wait"])
    lo = (au > 0) & (x.l <= mu) & (x.c > mu)
    sh = (ad > 0) & (x.h >= md) & (x.c < md)
    return _b(lo), _b(sh)


@entry("tweezer_at_level", "candles", tol_atr=("float", 0.1, 0.3), lookback=("int", 10, 50))
def _a8(x: Ctx, p):
    t = p["tol_atr"] * x.atr()
    n = p["lookback"]
    bot = (np.abs(x.l - _s(x.l)) < t) & (_s(x.body) < 0) & (x.body > 0) & (np.fmin(x.l, _s(x.l)) <= ind.rolling_min(x.l, n) + t)
    top = (np.abs(x.h - _s(x.h)) < t) & (_s(x.body) > 0) & (x.body < 0) & (np.fmax(x.h, _s(x.h)) >= ind.rolling_max(x.h, n) - t)
    return _b(bot), _b(top)


@entry("hammer_volume", "candles", wick_ratio=("float", 2.0, 3.5), range_mult=("float", 1.2, 2.0))
def _a9(x: Ctx, p):
    body = np.abs(x.body) + 1e-12
    upper = x.h - np.maximum(x.o, x.c)
    lower = np.minimum(x.o, x.c) - x.l
    big = x.rng > p["range_mult"] * _s(ind.sma(x.rng, 20))  # range as volume proxy
    down_before = _s(x.c) < _s(x.c, 5)
    return (_b((lower > p["wick_ratio"] * body) & (upper < body) & big & down_before),
            _b((upper > p["wick_ratio"] * body) & (lower < body) & big & ~down_before))


# ================================================================== B. candle size / range / structure
@entry("wrb_continuation", "candles", range_atr=("float", 1.5, 3.0), close_zone=("float", 0.1, 0.3),
       mode=("choice", ["next", "pullback"]))
def _b1(x: Ctx, p):
    pos = (x.c - x.l) / (x.rng + 1e-12)
    wrb = x.rng > p["range_atr"] * _s(x.atr())
    up = wrb & (pos >= 1 - p["close_zone"])
    dn = wrb & (pos <= p["close_zone"])
    if p["mode"] == "next":
        return _b(up), _b(dn)
    mid = (x.h + x.l) / 2
    mu, au = _ffill_event(mid, up, 5)
    md, ad = _ffill_event(mid, dn, 5)
    return _b((au > 0) & (x.l <= mu) & (x.c > mu)), _b((ad > 0) & (x.h >= md) & (x.c < md))


@entry("wrb_fade", "mean_reversion", range_atr=("float", 1.5, 3.0), trend=("int", 100, 200))
def _b2(x: Ctx, p):
    wrb = x.rng > p["range_atr"] * _s(x.atr())
    m = x.ma(p["trend"])
    return _b(wrb & (x.body < 0) & (x.c > m)), _b(wrb & (x.body > 0) & (x.c < m))  # fade WRBs against HTF trend


@entry("vol_contraction_breakout", "volatility", ratio=("float", 0.5, 0.8), range_n=("int", 10, 30))
def _b4(x: Ctx, p):
    r = x.atr(5) / x.atr(50)
    tight = _s(r) < p["ratio"]
    hh, ll = _s(ind.rolling_max(x.h, p["range_n"])), _s(ind.rolling_min(x.l, p["range_n"]))
    return _b(tight & (x.c > hh)), _b(tight & (x.c < ll))


@entry("body_momentum_run", "momentum", n=("int", 2, 4), ratio=("float", 0.6, 0.8))
def _b5(x: Ctx, p):
    r = np.abs(x.body) / (x.rng + 1e-12)
    up = np.ones(x.n, bool)
    dn = np.ones(x.n, bool)
    for k in range(p["n"]):
        up &= _b(_s(r, k) > p["ratio"]) & _b(_s(x.body, k) > 0)
        dn &= _b(_s(r, k) > p["ratio"]) & _b(_s(x.body, k) < 0)
    prev_up = _b(_s(up.astype(float)))
    prev_dn = _b(_s(dn.astype(float)))
    return up & ~prev_up, dn & ~prev_dn


@entry("close_location", "momentum", n=("int", 5, 30), thr=("float", 0.3, 0.8), mode=("choice", ["follow", "fade"]))
def _b7(x: Ctx, p):
    v = ind.sma(ind.clv(x.h, x.l, x.c), p["n"])
    up = (v > p["thr"]) & (_s(v) <= p["thr"])
    dn = (v < -p["thr"]) & (_s(v) >= -p["thr"])
    return _fu(up, dn, p["mode"])


@entry("weekly_gap_fade", "gap", min_gap_atr=("float", 0.3, 1.0))
def _b8(x: Ctx, p):
    dt = np.diff(x.t_ns, prepend=x.t_ns[0])
    week_open = dt > 24 * 3_600_000_000_000
    gap = x.o - _s(x.c)
    big = week_open & (np.abs(gap) > p["min_gap_atr"] * _s(x.atr()))
    still = np.sign(x.c - _s(x.c)) == np.sign(gap)  # gap not yet filled at the close of the first bar
    return _b(big & (gap < 0) & still), _b(big & (gap > 0) & still)


# ================================================================== C. trend following
@entry("ema_stack_pullback", "trend", fast=("int", 15, 25), mid=("int", 40, 60), slow=("int", 150, 250))
def _c2(x: Ctx, p):
    f, m, s = x.ma(p["fast"]), x.ma(p["mid"]), x.ma(p["slow"])
    up = (x.c > s) & (f > m) & (x.l <= f) & (x.c > f) & (x.body > 0)
    dn = (x.c < s) & (f < m) & (x.h >= f) & (x.c < f) & (x.body < 0)
    return _b(up), _b(dn)


@entry("supertrend_flip", "trend", n=("int", 7, 14), mult=("float", 2.0, 4.0), htf=("int", 0, 24))
def _c4(x: Ctx, p):
    d = x.get(("st", p["n"], p["mult"]), lambda: ind.supertrend(x.h, x.l, x.c, p["n"], p["mult"]))
    up = (d > 0) & (_s(d) < 0)
    dn = (d < 0) & (_s(d) > 0)
    if p["htf"] > 1:  # HTF confirmation: slope of a long SMA (n*htf)
        m = x.ma(20 * p["htf"], "sma")
        up &= _b(m > _s(m, p["htf"]))
        dn &= _b(m < _s(m, p["htf"]))
    return _b(up), _b(dn)


@entry("macd_hist_trend", "trend", fast=("int", 8, 15), slow=("int", 20, 35), signal=("int", 5, 12), trend=("int", 100, 200))
def _c5(x: Ctx, p):
    _, _, hist = x.get(("macd", p["fast"], p["slow"], p["signal"]), lambda: ind.macd(x.c, p["fast"], p["slow"], p["signal"]))
    m = x.ma(p["trend"])
    return _b((hist > 0) & (_s(hist) <= 0) & (x.c > m)), _b((hist < 0) & (_s(hist) >= 0) & (x.c < m))


@entry("psar_adx", "trend", step=("float", 0.01, 0.03), max_step=("float", 0.1, 0.3), min_adx=("float", 15, 30))
def _c6(x: Ctx, p):
    d = x.get(("psar", p["step"], p["max_step"]), lambda: ind.psar(x.h, x.l, p["step"], p["max_step"]))
    a = x.adx(14)[0]
    rising = (a > _s(a, 3)) & (a > p["min_adx"])
    return _b((d > 0) & (_s(d) < 0) & rising), _b((d < 0) & (_s(d) > 0) & rising)


@entry("keltner_breakout", "breakout", n=("int", 15, 30), mult=("float", 1.5, 2.5), trend=("int", 100, 200))
def _c7(x: Ctx, p):
    lo, mid, hi = x.get(("kc", p["n"], p["mult"]), lambda: ind.keltner(x.h, x.l, x.c, p["n"], p["mult"]))
    t = x.ma(p["trend"])
    return (_b((x.c > hi) & (_s(x.c) <= _s(hi)) & (x.c > t)), _b((x.c < lo) & (_s(x.c) >= _s(lo)) & (x.c < t)))


@entry("ichimoku_break", "trend", tenkan=("int", 7, 12), kijun=("int", 20, 30), senkou=("int", 40, 60))
def _c8(x: Ctx, p):
    t, k, top, bot = x.get(("ichi", p["tenkan"], p["kijun"], p["senkou"]),
                           lambda: ind.ichimoku(x.h, x.l, x.c, p["tenkan"], p["kijun"], p["senkou"]))
    chik_up = x.c > _s(x.c, p["kijun"])  # chikou (close plotted back) clear of past price
    up = (x.c > top) & (_s(x.c) <= _s(top)) & (t > k) & chik_up
    dn = (x.c < bot) & (_s(x.c) >= _s(bot)) & (t < k) & ~chik_up
    return _b(up), _b(dn)


# ================================================================== D. mean reversion
@entry("stoch_at_level", "mean_reversion", k=("int", 5, 21), zone=("float", 10, 30), tol_atr=("float", 0.2, 0.6))
def _d3(x: Ctx, p):
    kk, dd = x.get(("stoch", p["k"], 3), lambda: ind.stochastic(x.h, x.l, x.c, p["k"], 3))
    diff = kk - dd
    sh, sl = x.swings(3)
    t = p["tol_atr"] * x.atr()
    z = p["zone"]
    up = (diff > 0) & (_s(diff) <= 0) & (_s(kk) < z) & (np.abs(x.l - sl) < t)
    dn = (diff < 0) & (_s(diff) >= 0) & (_s(kk) > 100 - z) & (np.abs(x.h - sh) < t)
    return _b(up), _b(dn)


@entry("vwap_deviation", "mean_reversion", k=("float", 1.5, 2.5))
def _d5(x: Ctx, p):
    vw, sd = x.get("vwap", lambda: ind.session_vwap(x.h, x.l, x.c, x.v, x.day_id))
    lo, hi = vw - p["k"] * sd, vw + p["k"] * sd
    ok = sd > 0.2 * x.atr()
    return _b(ok & (_s(x.c) < _s(lo)) & (x.c > lo)), _b(ok & (_s(x.c) > _s(hi)) & (x.c < hi))


@entry("connors_rsi2", "mean_reversion", thr=("float", 3, 10), trend=("int", 150, 250))
def _d6(x: Ctx, p):
    r = x.rsi(2)
    m = x.ma(p["trend"], "sma")
    return _b((x.c > m) & (r < p["thr"])), _b((x.c < m) & (r > 100 - p["thr"]))


# ================================================================== E. breakout & session
@entry("london_fakeout", "session", asia_end=("int", 6, 8), window=("int", 1, 3))
def _e2(x: Ctx, p):
    key = ("orb", 0, p["asia_end"])

    def f():
        s = pd.DataFrame({"d": x.day_id, "h": x.h, "l": x.l, "hr": x.hour})
        g = s[(s.hr >= 0) & (s.hr < p["asia_end"])].groupby("d").agg(h=("h", "max"), l=("l", "min"))
        return g["h"].reindex(x.day_id).to_numpy(), g["l"].reindex(x.day_id).to_numpy()

    rh, rl = x.get(key, f)
    live = (x.hour >= p["asia_end"]) & (x.hour < p["asia_end"] + 4)
    w = p["window"]
    broke_up = _s(ind.rolling_max(x.h, w)) > rh
    broke_dn = _s(ind.rolling_min(x.l, w)) < rl
    return _b(live & broke_dn & (x.c > rl) & (_s(x.c) <= rl)), _b(live & broke_up & (x.c < rh) & (_s(x.c) >= rh))


@entry("ny_open_momentum", "session", minutes=("choice", [15, 30, 45, 60]), min_move_atr=("float", 0.0, 1.0))
def _e3(x: Ctx, p):
    start = _first_bar_at(x, "America/New_York", 9 * 60 + 30)
    o930, _ = _ffill_event(x.o, start, 400)
    trig = _first_bar_at(x, "America/New_York", 9 * 60 + 30 + p["minutes"] - 15)
    mv = x.c - o930
    thr = p["min_move_atr"] * x.atr()
    return _b(trig & (mv > thr)), _b(trig & (mv < -thr))


@entry("pdh_break_retest", "breakout", window=("int", 3, 12))
def _e5(x: Ctx, p):
    ph, pl, _, _ = x.day_hl()
    w = p["window"]
    was_above = _s(ind.rolling_max(x.c, w)) > ph
    was_below = _s(ind.rolling_min(x.c, w)) < pl
    return _b(was_above & (x.l <= ph) & (x.c > ph)), _b(was_below & (x.h >= pl) & (x.c < pl))


@entry("pdh_sweep_reverse", "mean_reversion", min_wick_atr=("float", 0.0, 0.5))
def _e6(x: Ctx, p):
    ph, pl, _, _ = x.day_hl()
    t = p["min_wick_atr"] * x.atr()
    return _b((x.l < pl - t) & (x.c > pl)), _b((x.h > ph + t) & (x.c < ph))


@entry("weekly_hl_break", "breakout", trend=("int", 0, 1))
def _e7(x: Ctx, p):
    def f():
        rolled = x.time.tz_convert("America/New_York") + pd.Timedelta(hours=7)
        wk = (rolled.normalize() - pd.to_timedelta(rolled.weekday, unit="D")).asi8
        s = pd.DataFrame({"w": wk, "h": x.h, "l": x.l, "c": x.c})
        g = s.groupby("w").agg(h=("h", "max"), l=("l", "min")).shift(1)
        return g["h"].reindex(wk).to_numpy(), g["l"].reindex(wk).to_numpy()

    wh, wl = x.get("week_hl", f)
    up = (x.c > wh) & (_s(x.c) <= wh)
    dn = (x.c < wl) & (_s(x.c) >= wl)
    if p["trend"]:
        m = x.ma(20 * 24 if x.tf in ("M15", "M30", "H1") else 200, "sma")
        up &= x.c > m
        dn &= x.c < m
    return _b(up), _b(dn)


@entry("london_close_reversal", "session", minute=("choice", [900, 930, 960]), min_move_atr=("float", 1.0, 3.0))
def _e8(x: Ctx, p):
    trig = _first_bar_at(x, "Europe/London", p["minute"])
    mv = x.c - _day_open(x)
    thr = p["min_move_atr"] * x.atr()
    return _b(trig & (mv < -thr)), _b(trig & (mv > thr))


# ================================================================== F. market structure / price action
@entry("bos_pullback", "structure", k=("int", 2, 5), window=("int", 3, 15))
def _f1(x: Ctx, p):
    sh, sl = x.swings(p["k"])
    bos_up = (x.c > sh) & (_s(x.c) <= _s(sh))
    bos_dn = (x.c < sl) & (_s(x.c) >= _s(sl))
    lu, au = _ffill_event(_s(sh, 0), bos_up, p["window"])
    ld, ad = _ffill_event(_s(sl, 0), bos_dn, p["window"])
    return _b((au > 0) & (x.l <= lu) & (x.c > lu)), _b((ad > 0) & (x.h >= ld) & (x.c < ld))


@entry("choch_reversal", "structure", k=("int", 2, 5))
def _f2(x: Ctx, p):
    sh, sl, _, _ = _struct(x, p["k"])
    lower_highs = sh[:, 0] < sh[:, 1]
    higher_lows = sl[:, 0] > sl[:, 1]
    up = lower_highs & (x.c > sh[:, 0]) & (_s(x.c) <= _s(sh[:, 0]))  # first higher high after a downtrend
    dn = higher_lows & (x.c < sl[:, 0]) & (_s(x.c) >= _s(sl[:, 0]))
    return _b(up), _b(dn)


@entry("sr_cluster_bounce", "structure", lookback=("int", 100, 300), tol_atr=("float", 0.2, 0.5), min_touches=("int", 3, 5))
def _f3(x: Ctx, p):
    k = 3

    def lvls():
        n = x.n
        hi = np.full(n, np.nan)
        lo = np.full(n, np.nan)
        for j in range(k, n - k):  # level stored at the bar it becomes known (j + k)
            seg_h = x.h[j - k: j + k + 1]
            seg_l = x.l[j - k: j + k + 1]
            if x.h[j] == seg_h.max():
                hi[j + k] = x.h[j]
            if x.l[j] == seg_l.min():
                lo[j + k] = x.l[j]
        return np.where(np.isnan(hi), lo, hi)

    lv = x.get(("swinglv", k), lvls)
    t = p["tol_atr"] * np.nan_to_num(x.atr(), nan=1e9)
    tl = x.get(("touch_l", p["lookback"], p["tol_atr"]), lambda: ind.level_touches(x.l, lv, p["lookback"], t))
    th = x.get(("touch_h", p["lookback"], p["tol_atr"]), lambda: ind.level_touches(x.h, lv, p["lookback"], t))
    rej_up = (x.body > 0) & ((x.c - x.l) > 0.6 * x.rng)
    rej_dn = (x.body < 0) & ((x.h - x.c) > 0.6 * x.rng)
    return _b((tl >= p["min_touches"]) & rej_up), _b((th >= p["min_touches"]) & rej_dn)


@entry("round_number", "structure", step_pips=("choice", [50, 100]), mode=("choice", ["fade", "break"]))
def _f4(x: Ctx, p):
    step = p["step_pips"] * pip_size(x.symbol)
    below = np.floor(x.c / step) * step  # nearest level under the close
    above = below + step
    if p["mode"] == "break":
        lvl_prev = np.floor(_s(x.c) / step) * step
        return _b(below > lvl_prev), _b(below < lvl_prev)
    t = 0.2 * x.atr()
    up = (x.l <= below + t) & (x.c > below) & (x.body > 0) & ((x.c - x.l) > 0.6 * x.rng)
    dn = (x.h >= above - t) & (x.c < above) & (x.body < 0) & ((x.h - x.c) > 0.6 * x.rng)
    return _b(up), _b(dn)


def _fvg(x: Ctx):
    bull = x.l > _s(x.h, 2)  # gap between bar i-2 high and bar i low
    bear = x.h < _s(x.l, 2)
    return bull, bear


@entry("fvg_fill", "structure", trend=("int", 50, 200), max_age=("int", 5, 30))
def _f5(x: Ctx, p):
    bull, bear = _fvg(x)
    m = x.ma(p["trend"])
    top_u, au = _ffill_event(x.l, bull, p["max_age"])
    bot_u, _ = _ffill_event(_s(x.h, 2), bull, p["max_age"])
    bot_d, ad = _ffill_event(x.h, bear, p["max_age"])
    top_d, _ = _ffill_event(_s(x.l, 2), bear, p["max_age"])
    up = (au > 0) & (x.l <= top_u) & (x.c >= bot_u) & (x.c > m)
    dn = (ad > 0) & (x.h >= bot_d) & (x.c <= top_d) & (x.c < m)
    return _b(up), _b(dn)


@entry("liquidity_sweep", "structure", k=("int", 2, 5), tol_atr=("float", 0.1, 0.3))
def _f6(x: Ctx, p):
    sh, sl, _, _ = _struct(x, p["k"])
    t = p["tol_atr"] * x.atr()
    eq_h = np.abs(sh[:, 0] - sh[:, 1]) < t
    eq_l = np.abs(sl[:, 0] - sl[:, 1]) < t
    top = np.fmax(sh[:, 0], sh[:, 1])
    bot = np.fmin(sl[:, 0], sl[:, 1])
    return _b(eq_l & (x.l < bot) & (x.c > bot)), _b(eq_h & (x.h > top) & (x.c < top))


@entry("double_top_bottom", "structure", k=("int", 2, 5), tol_atr=("float", 0.2, 0.5))
def _f7(x: Ctx, p):
    sh, sl, shi, sli = _struct(x, p["k"])
    t = p["tol_atr"] * x.atr()
    dtop = np.abs(sh[:, 0] - sh[:, 1]) < t
    dbot = np.abs(sl[:, 0] - sl[:, 1]) < t
    neck_dn = sl[:, 0]  # swing low between the two tops (most recent low)
    neck_up = sh[:, 0]
    return (_b(dbot & (x.c > neck_up) & (_s(x.c) <= _s(neck_up))),
            _b(dtop & (x.c < neck_dn) & (_s(x.c) >= _s(neck_dn))))


@entry("fib_retracement", "structure", k=("int", 3, 5), level=("choice", [0.382, 0.5, 0.618]), tol_atr=("float", 0.1, 0.3))
def _f8(x: Ctx, p):
    sh, sl, shi, sli = _struct(x, p["k"])
    up_trend = (sh[:, 0] > sh[:, 1]) & (sl[:, 0] > sl[:, 1]) & (shi > sli)  # last leg was up
    dn_trend = (sh[:, 0] < sh[:, 1]) & (sl[:, 0] < sl[:, 1]) & (sli > shi)
    leg = sh[:, 0] - sl[:, 0]
    lvl_u = sh[:, 0] - p["level"] * leg
    lvl_d = sl[:, 0] + p["level"] * leg
    t = p["tol_atr"] * x.atr()
    up = up_trend & (x.l <= lvl_u + t) & (x.c > lvl_u) & (x.body > 0)
    dn = dn_trend & (x.h >= lvl_d - t) & (x.c < lvl_d) & (x.body < 0)
    return _b(up), _b(dn)


# ================================================================== G. multi-timeframe & confluence
@entry("triple_screen", "trend", mult=("int", 4, 24), osc_n=("int", 7, 14), osc_lo=("float", 20, 40))
def _g2(x: Ctx, p):
    m = x.ma(13 * p["mult"])
    htf_up = m > _s(m, p["mult"])
    r = x.rsi(p["osc_n"])
    pulled_dn = _b(ind.rolling_min(np.nan_to_num(r, nan=50), 5) < p["osc_lo"])
    pulled_up = _b(ind.rolling_max(np.nan_to_num(r, nan=50), 5) > 100 - p["osc_lo"])
    return (_b(htf_up & pulled_dn & (x.c > _s(x.h))), _b(~htf_up & pulled_up & (x.c < _s(x.l))))


def _divergence(x: Ctx, k: int):
    """Bullish: newest confirmed swing low lower than previous while RSI there higher (and mirror)."""
    def f():
        sh, sl, shi, sli = _struct(x, k)
        r = x.rsi(14)
        n = x.n
        bull = np.zeros(n, bool)
        bear = np.zeros(n, bool)
        prev_li = -1
        prev_hi = -1
        last_li = -1
        last_hi = -1
        for i in range(n):
            li = int(sli[i])
            hi = int(shi[i])
            if li != last_li and li >= 0:
                if prev_li >= 0 and x.l[li] < x.l[prev_li] and r[li] > r[prev_li]:
                    bull[i] = True
                prev_li = li
                last_li = li
            if hi != last_hi and hi >= 0:
                if prev_hi >= 0 and x.h[hi] > x.h[prev_hi] and r[hi] < r[prev_hi]:
                    bear[i] = True
                prev_hi = hi
                last_hi = hi
        return bull, bear
    return x.get(("div", k), f)


@entry("confluence_score", "confluence", min_score=("int", 3, 4))
def _g3(x: Ctx, p):
    m = x.ma(200)
    sh, sl = x.swings(3)
    t = 0.5 * x.atr()
    eu, ed = ENTRIES_FN("engulfing")(x, {"min_body_atr": 0.3})
    pu, pd_ = ENTRIES_FN("pin_bar")(x, {"wick_ratio": 2.0, "min_range_atr": 0.7})
    dvu, dvd = _divergence(x, 3)
    dvu = _b(ind.rolling_max(dvu.astype(float), 5))
    dvd = _b(ind.rolling_max(dvd.astype(float), 5))
    sess = (x.hour >= 7) & (x.hour < 17)
    su = _b(x.c > m).astype(int) + _b(np.abs(x.l - sl) < t) + (eu | pu) + dvu + sess
    sd = _b(x.c < m).astype(int) + _b(np.abs(x.h - sh) < t) + (ed | pd_) + dvd + sess
    return _b((su >= p["min_score"]) & (eu | pu)), _b((sd >= p["min_score"]) & (ed | pd_))


@entry("divergence_at_level", "mean_reversion", k=("int", 3, 5), lookback=("int", 20, 100))
def _g4(x: Ctx, p):
    bu, bd = _divergence(x, p["k"])
    at_low = ind.rolling_min(x.l, p["k"] * 2 + 1) <= _s(ind.rolling_min(x.l, p["lookback"])) + 0.5 * x.atr()
    at_high = ind.rolling_max(x.h, p["k"] * 2 + 1) >= _s(ind.rolling_max(x.h, p["lookback"])) - 0.5 * x.atr()
    return _b(bu & at_low), _b(bd & at_high)


# ================================================================== H. statistical & time based
@entry("time_bucket_bias", "statistical", grain=("choice", ["dow_hour", "hour"]), min_n=("int", 50, 200),
       thr_t=("float", 1.5, 3.0))
def _h1(x: Ctx, p):
    """Expanding (strictly past) mean next-bar return per weekday/hour bucket; trade buckets with |t| > thr."""
    def f():
        ret = np.diff(np.log(x.c), prepend=np.nan)
        b = x.hour + (24 * x.dow if p["grain"] == "dow_hour" else 0)
        t = ind.bucket_tstat(ret, b.astype(np.int64), 7 * 24, p["min_n"])
        nxt = np.r_[t[1:], 0.0]  # stat for the next bar's bucket, built only from returns up to this bar
        hr_change = np.r_[x.hour[1:] != x.hour[:-1], False]
        return nxt, hr_change
    t, hc = x.get(("tbb", p["grain"], p["min_n"]), f)
    return _b(hc & (t > p["thr_t"])), _b(hc & (t < -p["thr_t"]))


def _usd_sign(sym: str) -> int:
    return 1 if sym.startswith("USD") else (-1 if sym.endswith("USD") else 0)


def _month_days(x: Ctx):
    """Trading days (weekdays, from the calendar - known in advance) left in / elapsed in the month."""
    def f():
        rolled = x.time.tz_convert("America/New_York") + pd.Timedelta(hours=7)
        d = rolled.tz_localize(None).normalize().to_numpy().astype("datetime64[D]")
        m0 = d.astype("datetime64[M]").astype("datetime64[D]")
        m1 = (d.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]")
        left = np.busday_count(d + np.timedelta64(1, "D"), m1)
        done = np.busday_count(m0, d)
        first_bar = np.r_[True, x.day_id[1:] != x.day_id[:-1]]
        return left, done, rolled.month.to_numpy(), first_bar
    return x.get("month_days", f)


@entry("month_end_flow", "calendar", days=("int", 1, 3), usd=("choice", ["long_usd", "short_usd"]))
def _h3(x: Ctx, p):
    re, _, _, fb = _month_days(x)
    s = _usd_sign(x.symbol) * (1 if p["usd"] == "long_usd" else -1)
    trig = fb & (re < p["days"])
    return (trig & (s > 0)), (trig & (s < 0))


@entry("first_day_month", "calendar", direction=("choice", ["long", "short"]))
def _n5(x: Ctx, p):
    _, rs, _, fb = _month_days(x)
    t = fb & (rs == 0)
    z = np.zeros(x.n, bool)
    return (t, z) if p["direction"] == "long" else (z, t)


@entry("quarter_end", "calendar", days=("int", 1, 3), usd=("choice", ["long_usd", "short_usd"]))
def _n6(x: Ctx, p):
    re, _, mo, fb = _month_days(x)
    s = _usd_sign(x.symbol) * (1 if p["usd"] == "long_usd" else -1)
    t = fb & (re < p["days"]) & np.isin(mo, [3, 6, 9, 12])
    return (t & (s > 0)), (t & (s < 0))


# ================================================================== K. advanced candle & bar behaviour
@entry("turtle_soup", "mean_reversion", n=("int", 15, 30), within=("int", 1, 2))
def _k1(x: Ctx, p):
    ll = _s(ind.rolling_min(x.l, p["n"]), p["within"] + 1)  # prior n-bar low before the probe
    hh = _s(ind.rolling_max(x.h, p["n"]), p["within"] + 1)
    probe_l = ind.rolling_min(x.l, p["within"] + 1) < ll
    probe_h = ind.rolling_max(x.h, p["within"] + 1) > hh
    return _b(probe_l & (x.c > ll) & (_s(x.c) <= ll)), _b(probe_h & (x.c < hh) & (_s(x.c) >= hh))


@entry("three_bar_reversal", "candles", min_range_atr=("float", 0.0, 1.0))
def _k2(x: Ctx, p):
    ok = _s(x.rng) > p["min_range_atr"] * x.atr()
    bull = (_s(x.l) < _s(x.l, 2)) & (_s(x.l) < x.l) & (x.c > _s(x.h, 2))
    bear = (_s(x.h) > _s(x.h, 2)) & (_s(x.h) > x.h) & (x.c < _s(x.l, 2))
    return _b(bull & ok), _b(bear & ok)


@entry("hikkake", "candles", window=("int", 2, 4))
def _k3(x: Ctx, p):
    inside = (x.h < _s(x.h)) & (x.l > _s(x.l))
    ih, ai = _ffill_event(x.h, inside, p["window"] + 1)
    il, _ = _ffill_event(x.l, inside, p["window"] + 1)
    fake_dn = _b(ind.rolling_min(x.l, p["window"]) < il)
    fake_up = _b(ind.rolling_max(x.h, p["window"]) > ih)
    return _b((ai > 1) & fake_dn & (x.c > ih)), _b((ai > 1) & fake_up & (x.c < il))


@entry("kicker", "candles", min_body_atr=("float", 0.3, 1.0))
def _k4(x: Ctx, p):
    big = np.abs(x.body) > p["min_body_atr"] * x.atr()
    bull = (_s(x.body) < 0) & (x.body > 0) & (x.o >= _s(x.o)) & big
    bear = (_s(x.body) > 0) & (x.body < 0) & (x.o <= _s(x.o)) & big
    return _b(bull), _b(bear)


@entry("marubozu", "momentum", wick_frac=("float", 0.02, 0.1), min_range_atr=("float", 0.8, 2.0))
def _k5(x: Ctx, p):
    wick = x.rng - np.abs(x.body)
    m = (wick <= p["wick_frac"] * x.rng) & (x.rng > p["min_range_atr"] * x.atr())
    return _b(m & (x.body > 0)), _b(m & (x.body < 0))


@entry("exhaustion_bar", "mean_reversion", n=("int", 15, 30), ext_atr=("float", 2.0, 4.0))
def _k6(x: Ctx, p):
    biggest = x.rng >= ind.rolling_max(x.rng, p["n"])
    dist = (x.c - x.ma(50)) / x.atr()
    return _b(biggest & (dist < -p["ext_atr"]) & (x.body < 0)), _b(biggest & (dist > p["ext_atr"]) & (x.body > 0))


@entry("reversal_day", "mean_reversion", n=("int", 10, 30), zone=("float", 0.2, 0.4))
def _k7(x: Ctx, p):
    pos = (x.c - x.l) / (x.rng + 1e-12)
    new_lo = x.l < _s(ind.rolling_min(x.l, p["n"]))
    new_hi = x.h > _s(ind.rolling_max(x.h, p["n"]))
    return _b(new_lo & (pos > 1 - p["zone"])), _b(new_hi & (pos < p["zone"]))


@entry("upthrust_spring", "mean_reversion", n=("int", 20, 50), range_mult=("float", 1.2, 2.0))
def _k8(x: Ctx, p):
    hh, ll = _s(ind.rolling_max(x.h, p["n"])), _s(ind.rolling_min(x.l, p["n"]))
    big = x.rng > p["range_mult"] * _s(ind.sma(x.rng, 20))
    return _b(big & (x.l < ll) & (x.c > ll)), _b(big & (x.h > hh) & (x.c < hh))


@entry("higher_low_coil", "breakout", k=("int", 2, 4), tol_atr=("float", 0.2, 0.5))
def _k10(x: Ctx, p):
    sh, sl, _, _ = _struct(x, p["k"])
    t = p["tol_atr"] * x.atr()
    asc = (sl[:, 0] > sl[:, 1]) & (sl[:, 1] > sl[:, 2]) & (np.abs(sh[:, 0] - sh[:, 1]) < t)
    desc = (sh[:, 0] < sh[:, 1]) & (sh[:, 1] < sh[:, 2]) & (np.abs(sl[:, 0] - sl[:, 1]) < t)
    res = np.fmax(sh[:, 0], sh[:, 1])
    sup = np.fmin(sl[:, 0], sl[:, 1])
    return _b(asc & (x.c > res) & (_s(x.c) <= res)), _b(desc & (x.c < sup) & (_s(x.c) >= sup))


# ================================================================== L. volatility & statistical
@entry("atr_percentile_breakout", "volatility", pct=("float", 0.1, 0.2), lookback=("int", 500, 2000), range_n=("int", 10, 30))
def _l1(x: Ctx, p):
    rk = x.get(("atr_rank", p["lookback"]), lambda: ind.percentile_rank(x.atr(), p["lookback"]))
    low = _b(ind.rolling_min(np.nan_to_num(_s(rk), nan=1.0), 5) < p["pct"])
    hh, ll = _s(ind.rolling_max(x.h, p["range_n"])), _s(ind.rolling_min(x.l, p["range_n"]))
    return _b(low & (x.c > hh)), _b(low & (x.c < ll))


@entry("ttm_squeeze", "volatility", n=("int", 15, 25), kc_mult=("float", 1.2, 2.0))
def _l2(x: Ctx, p):
    blo, mid, bhi = x.get(("bb", p["n"], 2.0), lambda: ind.bollinger(x.c, p["n"], 2.0))
    klo, kmid, khi = x.get(("kc", p["n"], p["kc_mult"]), lambda: ind.keltner(x.h, x.l, x.c, p["n"], p["kc_mult"]))
    sq = (bhi < khi) & (blo > klo)
    release = _b(_s(sq.astype(float))) & ~sq
    mom = x.c - (ind.rolling_max(x.h, p["n"]) + ind.rolling_min(x.l, p["n"]) + mid * 2) / 4
    return _b(release & (mom > 0)), _b(release & (mom < 0))


@entry("session_drift", "statistical", min_n=("int", 50, 200), thr_t=("float", 1.5, 3.0))
def _l3(x: Ctx, p):
    """Asian (0-7 UTC) vs London (7-12) vs New York (12-21) session returns; trade the persistent one."""
    sess = np.where(x.hour < 7, 0, np.where(x.hour < 12, 1, np.where(x.hour < 21, 2, 3)))
    ret = np.diff(np.log(x.c), prepend=np.nan)
    t = x.get(("sdrift", p["min_n"]), lambda: ind.bucket_tstat(ret, sess.astype(np.int64), 4, p["min_n"] * 10))
    nxt = np.r_[t[1:], 0.0]
    start = np.r_[sess[1:] != sess[:-1], False]
    return _b(start & (nxt > p["thr_t"])), _b(start & (nxt < -p["thr_t"]))


@entry("vol_spike_fade", "mean_reversion", mult=("float", 2.0, 3.0), lookback=("int", 3, 8))
def _l6(x: Ctx, p):
    r = x.atr(5) / x.atr(50)
    spike = _b(ind.rolling_max(np.nan_to_num(r), p["lookback"]) > p["mult"])
    falling = r < _s(r)
    mv = x.c - _s(x.c, p["lookback"])
    return _b(spike & falling & (mv < 0)), _b(spike & falling & (mv > 0))


@entry("range_position", "momentum", n=("int", 50, 200), pct=("float", 0.03, 0.1), mode=("choice", ["follow", "fade"]))
def _l7(x: Ctx, p):
    hh, ll = ind.rolling_max(x.h, p["n"]), ind.rolling_min(x.l, p["n"])
    pos = (x.c - ll) / (hh - ll + 1e-12)
    up = (pos > 1 - p["pct"]) & (_s(pos) <= 1 - p["pct"])
    dn = (pos < p["pct"]) & (_s(pos) >= p["pct"])
    return _fu(up, dn, p["mode"])


@entry("kalman_slope", "trend", q_ratio=("choice", [1e-5, 1e-4, 1e-3, 1e-2]))
def _l8(x: Ctx, p):
    def f():
        scale = np.nanmedian(np.abs(np.diff(x.c))) ** 2 + 1e-18
        return ind.kalman_trend(x.c, p["q_ratio"] * scale, scale)[1]
    sl = x.get(("kal", p["q_ratio"]), f)
    return _b((sl > 0) & (_s(sl) <= 0)), _b((sl < 0) & (_s(sl) >= 0))


@entry("linreg_channel_fade", "mean_reversion", n=("int", 50, 200), k=("float", 1.5, 2.5))
def _l9(x: Ctx, p):
    line, slope, res = x.get(("lr", p["n"]), lambda: ind.linreg(x.c, p["n"]))
    return _b((slope > 0) & (x.c < line - p["k"] * res)), _b((slope < 0) & (x.c > line + p["k"] * res))


# ================================================================== N. time & calendar
@entry("tokyo_fix", "calendar", mode=("choice", ["follow", "fade"]), pre_hours=("int", 1, 3))
def _n1(x: Ctx, p):
    trig = _first_bar_at(x, "UTC", 60)  # first bar after the 00:55 UTC fix
    mv = x.c - _s(x.c, max(1, int(p["pre_hours"] * _bars_per_hour(x))))
    return _fu(trig & (mv > 0), trig & (mv < 0), p["mode"])


def _bars_per_hour(x: Ctx) -> float:
    return {"M5": 12, "M15": 4, "M30": 2, "H1": 1, "H4": 0.25, "D1": 1 / 24}.get(x.tf, 1)


@entry("london_fix_fade", "calendar", min_move_atr=("float", 0.5, 2.0))
def _n2(x: Ctx, p):
    trig = _first_bar_at(x, "Europe/London", 16 * 60)
    start, _ = _ffill_event(x.o, _first_bar_at(x, "Europe/London", 15 * 60 + 30), 12)
    mv = x.c - start
    t = p["min_move_atr"] * x.atr()
    return _b(trig & (mv < -t)), _b(trig & (mv > t))


@entry("friday_fade", "calendar", hour_ny=("int", 10, 14), min_move_atr=("float", 1.0, 5.0))
def _n3(x: Ctx, p):
    ny = x.time.tz_convert("America/New_York")
    trig = _first_bar_at(x, "America/New_York", p["hour_ny"] * 60) & np.asarray(ny.weekday == 4)
    dt = np.diff(x.t_ns, prepend=x.t_ns[0])
    wopen, _ = _ffill_event(x.o, dt > 24 * 3_600_000_000_000, 10_000)
    mv = x.c - wopen
    t = p["min_move_atr"] * x.atr() * np.sqrt(_bars_per_hour(x) * 24)
    return _b(trig & (mv < -t)), _b(trig & (mv > t))


@entry("monday_asia", "calendar", mode=("choice", ["follow", "fade"]))
def _n4(x: Ctx, p):
    dt = np.diff(x.t_ns, prepend=x.t_ns[0])
    wk = dt > 24 * 3_600_000_000_000
    _, _, pc, _ = x.day_hl()
    fri_dir = _s(x.c) - _s(_day_open(x))  # last bar of the previous (Friday) session vs its open
    return _fu(wk & (fri_dir > 0), wk & (fri_dir < 0), p["mode"])


@entry("lunch_lull_fade", "mean_reversion", z=("float", 1.0, 2.5))
def _n7(x: Ctx, p):
    m = _local_minutes(x, "America/New_York")
    lunch = (m >= 12 * 60) & (m < 13 * 60 + 30)
    z = x.get(("z", 20), lambda: ind.zscore(x.c, 20))
    return _b(lunch & (z < -p["z"])), _b(lunch & (z > p["z"]))


@entry("session_handover", "session", mode=("choice", ["follow", "fade"]), min_move_atr=("float", 0.0, 1.5))
def _n8(x: Ctx, p):
    trig = _first_bar_at(x, "Europe/London", 8 * 60)
    asia_open, _ = _ffill_event(x.o, _first_bar_at(x, "UTC", 0), 200)
    mv = x.c - asia_open
    t = p["min_move_atr"] * x.atr()
    return _fu(trig & (mv > t), trig & (mv < -t), p["mode"])


# ================================================================== O. order-flow proxies
@entry("absorption", "mean_reversion", n=("int", 3, 6), range_mult=("float", 1.2, 2.0), max_progress=("float", 0.1, 0.4))
def _o1(x: Ctx, p):
    n = p["n"]
    big = ind.sma(x.rng, n) > p["range_mult"] * x.atr(50)
    prog = np.abs(x.c - _s(x.c, n)) < p["max_progress"] * pd.Series(x.rng).rolling(n).sum().to_numpy()
    prior = x.c - _s(x.c, 2 * n)
    return _b(big & prog & (prior < 0)), _b(big & prog & (prior > 0))  # reversal of the absorbed move


@entry("effort_vs_result", "mean_reversion", n=("int", 5, 10))
def _o2(x: Ctx, p):
    n = p["n"]
    r = ind.sma(x.rng, n)
    effort = r > 1.2 * _s(r, n)
    res = np.abs(x.c - _s(x.c, n))
    prior = x.c - _s(x.c, n)
    weak = res < 0.5 * np.abs(_s(x.c, n) - _s(x.c, 2 * n))
    prior_dir = _s(x.c, n) - _s(x.c, 2 * n)
    return _b(effort & weak & (prior_dir < 0) & (prior > -1e9)), _b(effort & weak & (prior_dir > 0))


@entry("stop_run", "mean_reversion", k=("int", 2, 5), pips=("float", 1.0, 5.0))
def _o3(x: Ctx, p):
    sh, sl = x.swings(p["k"])
    d = p["pips"] * pip_size(x.symbol)
    return (_b((x.l < sl) & (x.l >= sl - d) & (x.c > sl)), _b((x.h > sh) & (x.h <= sh + d) & (x.c < sh)))


@entry("fvg_stack", "trend", stack=("int", 2, 3), window=("int", 5, 15))
def _o4(x: Ctx, p):
    bull, bear = _fvg(x)
    nb = pd.Series(bull.astype(float)).rolling(p["window"]).sum().to_numpy()
    ns = pd.Series(bear.astype(float)).rolling(p["window"]).sum().to_numpy()
    top_u, au = _ffill_event(x.l, bull, p["window"])
    top_d, ad = _ffill_event(x.h, bear, p["window"])
    return (_b((nb >= p["stack"]) & (au > 0) & (x.l <= top_u) & (x.c > top_u)),
            _b((ns >= p["stack"]) & (ad > 0) & (x.h >= top_d) & (x.c < top_d)))


# ================================================================== regime filters (K9, L4, L5, O5)
@filt("choppiness_regime", n=("int", 10, 30), mode=("choice", ["trend", "range"]), thr=("float", 45, 60))
def _fk9(x: Ctx, p):
    ci = x.get(("chop", p["n"]), lambda: ind.choppiness(x.h, x.l, x.c, p["n"]))
    a = _b(ci < p["thr"]) if p["mode"] == "trend" else _b(ci > p["thr"])
    return a, a


@filt("autocorr_regime", n=("int", 50, 300), mode=("choice", ["momentum", "reversion"]))
def _fl4(x: Ctx, p):
    ac = x.get(("ac", p["n"]), lambda: ind.autocorr(x.c, p["n"]))
    a = _b(ac > 0) if p["mode"] == "momentum" else _b(ac < 0)
    return a, a


@filt("hurst_regime", n=("int", 100, 400), mode=("choice", ["trend", "reversion"]), band=("float", 0.02, 0.08))
def _fl5(x: Ctx, p):
    hu = x.get(("hurst", p["n"]), lambda: ind.hurst(x.c, p["n"]))
    a = _b(hu > 0.5 + p["band"]) if p["mode"] == "trend" else _b(hu < 0.5 - p["band"])
    return a, a


@filt("premium_discount", k=("int", 3, 6))
def _fo5(x: Ctx, p):
    sh, sl = x.swings(p["k"])
    mid = (sh + sl) / 2
    return _b(x.c < mid), _b(x.c > mid)  # longs only in discount, shorts only in premium


def ENTRIES_FN(name):  # late lookup to reuse existing blocks
    from propfloor.strategies.blocks import ENTRIES

    return ENTRIES[name].fn
