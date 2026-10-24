"""Entry logic of the user's gold cBots 6 and 7 (research/user_bots/*.cs, standalone replays in
propfloor/ports/cbots.py) as factory entry blocks, so the factory can search their neighbourhood, combine them with
filters / its own EXIT_SPACE and try them on other symbols, timeframes and the short side.

breakout_runner - bot 6 (Breakout Trend Runner v3, H1): while fast EMA > slow EMA on the last CLOSED bar, the bar's
                  high breaks the highest high of the `lookback` closed bars before it (the bot buys that break
                  tick-by-tick; here the signal is raised at the close of the breaking bar and filled next open),
                  optionally only when the last COMPLETED broker day closed above its daily EMA(200).
                  Shorts mirror it (low < lowest low, fast < slow, daily close below EMA200).
macd_dema       - bot 7 (MACD-histogram + DEMA v4, H4 entry / H12 filter): 2-bar MACD-histogram state machine
                  (rises twice -> long state, falls twice -> short state, else keep), entry on the first bar of a
                  state leg where the DEMA(dema_n) of the last COMPLETED higher-timeframe bar (htf_mult x the bar
                  size, built on the broker clock NY+7 like the bots) slopes the same way; one signal per leg.

Only entry signals are produced; stops/targets/trails come from the genome's EXIT_SPACE. EMAs are seeded with the
first value as in the C# (cbots.ema_seeded). Every signal at bar t uses bars <= t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from propfloor.strategies import indicators as ind
from propfloor.strategies.blocks import Ctx, _b, entry

_TF_MIN = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H2": 120, "H4": 240, "H6": 360, "H8": 480,
           "H12": 720, "D1": 1440}
NS_MIN = 60 * 10**9


def _ema0(x: np.ndarray, n: int) -> np.ndarray:
    """EMA seeded with the first value (cTrader / the bots' hand-coded EMAs)."""
    return ind._ewm(np.asarray(x, dtype=np.float64), 2.0 / (n + 1))


def _srv_ns(x: Ctx) -> np.ndarray:
    """Bar open times on the broker clock (New York + 7h, naive) in ns."""
    def f():
        s = x.time.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
        return s.as_unit("ns").asi8
    return x.get("srv_ns", f)


def _daily_trend(x: Ctx, n: int = 200, min_days: int = 210) -> np.ndarray:
    """+1/-1 = the last COMPLETED broker day (the one before the bar's day) closed above/below its seeded EMA(n),
    0 while fewer than min_days days of history (cbots.daily_trend)."""
    def f():
        codes, _ = pd.factorize(x.day_id)  # day_id is monotone -> codes increase
        dc = pd.Series(x.c).groupby(codes).last().to_numpy()
        tr = np.where(dc > _ema0(dc, n), 1, -1)
        tr[:min_days] = 0
        prev = codes - 1
        return np.where(prev >= 0, tr[np.clip(prev, 0, None)], 0)
    return x.get(("cbot_daily_trend", n, min_days), f)


def _htf_last_closed(x: Ctx, mult: int) -> tuple[np.ndarray, np.ndarray]:
    """Higher timeframe = mult x the bar size on the broker clock. Returns (closes of the HTF bars, index of the last
    HTF bar COMPLETE at each base bar's close, -1 if none). The current HTF bar counts only once its clock end is
    reached, so its (still forming) close is never read early."""
    def f():
        bar_ns = _TF_MIN.get(x.tf.upper(), 60) * NS_MIN
        per = mult * bar_ns
        srv = _srv_ns(x)
        g = srv // per
        codes, _ = pd.factorize(g)
        hc = pd.Series(x.c).groupby(codes).last().to_numpy()
        done = srv + bar_ns >= (g + 1) * per
        return hc, np.where(done, codes, codes - 1)
    return x.get(("cbot_htf", mult), f)


def _dema0(x: np.ndarray, n: int) -> np.ndarray:
    e1 = _ema0(x, n)
    out = 2 * e1 - _ema0(e1, n)
    if len(out):
        out[0] = x[0]
    return out


# ------------------------------------------------------------------ bot 6
@entry("breakout_runner", "breakout", lookback=("int", 18, 30), fast=("int", 40, 60), slow=("int", 150, 250),
       daily_gate=("choice", [True, False]))
def _breakout_runner(x: Ctx, p):
    lb, fa, sl = int(p["lookback"]), int(p["fast"]), int(p["slow"])
    ef = x.get(("ema0", fa), lambda: _ema0(x.c, fa))
    es = x.get(("ema0", sl), lambda: _ema0(x.c, sl))
    hh = pd.Series(x.h).rolling(lb).max().shift(1).to_numpy()  # highest high of the lb closed bars before t
    ll = pd.Series(x.l).rolling(lb).min().shift(1).to_numpy()
    up_prev = np.r_[False, ef[:-1] > es[:-1]]  # EMA alignment on the last closed bar before the break
    dn_prev = np.r_[False, ef[:-1] < es[:-1]]
    warm = np.arange(x.n) - 1 >= max(250, lb - 1)  # bot: last closed bar index >= 250
    with np.errstate(invalid="ignore"):
        lo = warm & up_prev & (x.h > hh)
        sh = warm & dn_prev & (x.l < ll)
    if p["daily_gate"]:
        dt = _daily_trend(x)
        lo &= dt == 1
        sh &= dt == -1
    return _b(lo), _b(sh)


# ------------------------------------------------------------------ bot 7
def _macd_state(c: np.ndarray, fast: int, slow: int, sig_n: int) -> np.ndarray:
    """cbots.macd_hist_state with shorts on: +1 after two rising histogram bars, -1 after two falling, else keep."""
    macd = _ema0(c, fast) - _ema0(c, slow)
    y = macd.copy()
    if len(y):
        y[0] = 0.0
    hist = macd - ind._ewm(y, 2.0 / (sig_n + 1))  # signal line seeded at 0
    if len(hist):
        hist[0] = 0.0
    h1, h2 = np.r_[np.nan, hist[:-1]], np.r_[np.nan, np.nan, hist[:-2]][: len(hist)]
    with np.errstate(invalid="ignore"):
        ev = np.where((hist > h1) & (h1 > h2), 1.0, np.where((hist < h1) & (h1 < h2), -1.0, np.nan))
    return pd.Series(ev).ffill().fillna(0.0).to_numpy()


@entry("macd_dema", "trend", fast=("int", 9, 15), slow=("int", 20, 32), signal=("int", 7, 11),
       dema_n=("int", 75, 125), htf_mult=("choice", [2, 3, 4, 6]))
def _macd_dema(x: Ctx, p):
    fa, sl, sg, dn, m = int(p["fast"]), int(p["slow"]), int(p["signal"]), int(p["dema_n"]), int(p["htf_mult"])
    st = x.get(("cbot_macd_state", fa, sl, sg), lambda: _macd_state(x.c, fa, sl, sg))
    hc, last = _htf_last_closed(x, m)
    fd = x.get(("cbot_htf_dema", m, dn), lambda: _dema0(hc, dn))
    ok = last >= 2 * dn
    li = np.clip(last, 1, None)
    slope = np.where(ok, fd[li] - fd[li - 1], 0.0)
    filt = np.sign(slope)
    leg = np.cumsum(np.r_[0, st[1:] != st[:-1]])
    warm = np.arange(x.n) >= 100  # bot: EntryWarmup = 100 entry bars
    out = []
    for d in (1, -1):
        cond = warm & (st == d) & (filt == d)
        seen = pd.Series(cond).groupby(leg).cummax().to_numpy(bool)  # leg already had a qualifying bar
        prev_seen = np.r_[False, seen[:-1]] & np.r_[False, leg[1:] == leg[:-1]]
        out.append(_b(cond & ~prev_seen))  # one entry per state leg (the bot's traded_leg)
    return out[0], out[1]
