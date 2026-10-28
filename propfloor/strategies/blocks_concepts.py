"""Discretionary price-action concepts the user asked the factory to automate, as entry blocks.

icc             - Indication / Correction / Continuation (the "ICC" concept taught by Trades by Sci):
                  Indication   = a close through the last confirmed swing high (break of structure),
                  Correction   = price pulls back between `min_retr` and `max_retr` of the indication leg
                                 (leg = last confirmed swing low -> highest high since the break) without
                                 taking out the leg's origin,
                  Continuation = a close above the highest high of the previous `trig_n` bars (the correction's
                                 micro-structure breaks back in the indication's direction) within `max_bars`.
                  One signal per indication. Shorts mirror it.
session_reclaim - Session reclaim continuation: the range built between `range_start`..`range_end` UTC breaks
                  (a close beyond it) during the session (until `latest`), price retraces into the
                  `zone_a`..`zone_b` (~0.70-0.886) Fibonacci zone of the breakout leg, and a candle closes back
                  out of the zone in the breakout's direction (the reclaim). A close through the deep end of the
                  zone cancels the setup. At most one signal per direction per day.
                  Fib anchors: `range` = far side of the range -> breakout extreme, `level` = the broken range edge
                  -> extreme (a retest of the broken level), `leg` = lowest low of the `leg_n` bars before the
                  breakout -> extreme.

Only entries are produced; stops/targets/trails come from the genome's EXIT_SPACE (swing stops suit both). Every
signal at bar t uses bars <= t only (state machines walk forward; ranges are read only after they are complete).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from propfloor.strategies.blocks import Ctx, _b, entry


@njit(cache=True)
def _icc_long(h, l, c, sw_hi, sw_lo, atr, min_retr, max_retr, max_bars, trig_n, min_leg_atr):
    n = h.shape[0]
    out = np.zeros(n, np.bool_)
    state = 0
    start = 0
    leg_lo = 0.0
    leg_hi = 0.0
    corr_lo = np.inf
    for i in range(1, n):
        if state == 1:
            if l[i] < leg_lo or i - start > max_bars:
                state = 0
            else:
                depth = (leg_hi - corr_lo) / (leg_hi - leg_lo) if np.isfinite(corr_lo) else 0.0
                if depth > max_retr:
                    state = 0
                elif depth >= min_retr and i >= trig_n:
                    mx = -np.inf
                    for j in range(i - trig_n, i):
                        if h[j] > mx:
                            mx = h[j]
                    if c[i] > mx:
                        out[i] = True
                        state = 0
                        continue
                if state == 1:
                    if h[i] > leg_hi:
                        leg_hi = h[i]
                        corr_lo = np.inf  # a new extension restarts the correction
                    elif l[i] < corr_lo:
                        corr_lo = l[i]
                    if (leg_hi - corr_lo) / (leg_hi - leg_lo) > max_retr:
                        state = 0
        if state == 0:
            s = sw_hi[i]
            lo = sw_lo[i]
            if np.isfinite(s) and np.isfinite(lo) and c[i] > s and c[i - 1] <= s and lo < s \
                    and np.isfinite(atr[i]) and h[i] - lo >= min_leg_atr * atr[i]:
                state = 1
                start = i
                leg_lo = lo
                leg_hi = h[i]
                corr_lo = np.inf
    return out


@entry("icc", "structure", k=("int", 2, 6), min_retr=("float", 0.2, 0.5), max_retr=("float", 0.618, 0.9),
       max_bars=("int", 8, 60), trig_n=("int", 1, 4), min_leg_atr=("float", 0.5, 3.0))
def _icc(x: Ctx, p):
    sh, sl = x.swings(int(p["k"]))
    args = (float(p["min_retr"]), float(p["max_retr"]), int(p["max_bars"]), int(p["trig_n"]), float(p["min_leg_atr"]))
    a = x.atr()
    lo = _icc_long(x.h, x.l, x.c, sh, sl, a, *args)
    # shorts: the same machine on the price mirror (highs <-> -lows)
    sho = _icc_long(-x.l, -x.h, -x.c, -sl, -sh, a, *args)
    return _b(lo), _b(sho)


@njit(cache=True)
def _reclaim_long(o, h, l, c, atr, live, day, rh, rl, anchor, leg_n, za, zb, min_ext_atr):
    n = h.shape[0]
    out = np.zeros(n, np.bool_)
    state = 0  # 0 waiting for breakout, 1 broken out, 2 done for the day
    cur_day = -1
    anc = 0.0
    hi = 0.0
    touched = False
    for i in range(1, n):
        if day[i] != cur_day:
            cur_day = day[i]
            state = 0
        if not live[i] or state == 2 or not np.isfinite(rh[i]) or not np.isfinite(rl[i]):
            continue
        if state == 0:
            if c[i] > rh[i]:
                state = 1
                hi = h[i]
                touched = False
                if anchor == 0:
                    anc = rl[i]
                elif anchor == 1:
                    anc = rh[i]
                else:
                    anc = np.inf
                    for j in range(max(0, i - leg_n), i):
                        if l[j] < anc:
                            anc = l[j]
            continue
        ext = hi - anc
        if ext <= 0 or not np.isfinite(atr[i]) or ext < min_ext_atr * atr[i]:
            if h[i] > hi:
                hi = h[i]
            continue
        lvl_a = hi - za * ext
        lvl_b = hi - zb * ext
        if c[i] < lvl_b:
            state = 2  # closed through the deep end of the zone: the breakout failed
            continue
        if l[i] <= lvl_a:
            touched = True
        if touched and c[i] > lvl_a and c[i] > o[i]:
            out[i] = True
            state = 2
            continue
        if h[i] > hi:
            hi = h[i]
            touched = False  # new extension: the zone moves up and must be revisited
    return out


@entry("session_reclaim", "session", range_start=("int", 0, 6), range_end=("int", 6, 9), latest=("int", 11, 20),
       zone_a=("float", 0.618, 0.75), zone_b=("float", 0.786, 0.9), anchor=("choice", ["range", "level", "leg"]),
       leg_n=("int", 4, 24), min_ext_atr=("float", 0.3, 2.0))
def _reclaim(x: Ctx, p):
    rs, re_ = int(p["range_start"]), int(p["range_end"])

    def f():
        s = pd.DataFrame({"d": x.day_id, "h": x.h, "l": x.l, "hr": x.hour})
        inwin = (s.hr >= rs) & (s.hr < re_)
        g = s[inwin].groupby("d").agg(h=("h", "max"), l=("l", "min"))
        return g["h"].reindex(x.day_id).to_numpy(), g["l"].reindex(x.day_id).to_numpy()

    rh, rl = x.get(("orb", rs, re_), f)  # same key/values as session_range_breakout
    live = np.asarray((x.hour >= re_) & (x.hour < int(p["latest"])))
    za, zb = sorted((float(p["zone_a"]), float(p["zone_b"])))
    anchor = {"range": 0, "level": 1, "leg": 2}[p["anchor"]]
    args = (anchor, int(p["leg_n"]), za, zb, float(p["min_ext_atr"]))
    a = x.atr()
    lo = _reclaim_long(x.o, x.h, x.l, x.c, a, live, x.day_id, rh, rl, *args)
    sho = _reclaim_long(-x.o, -x.l, -x.h, -x.c, a, live, x.day_id, -rl, -rh, *args)
    return _b(lo), _b(sho)
