"""Composable strategy building blocks.

ENTRIES  : f(ctx, params) -> (long_signal, short_signal)   evaluated on bar close, filled next open
FILTERS  : f(ctx, params) -> (allow_long, allow_short)
Exits and sizing are parameter sets interpreted by the backtester / risk manager.

Every block declares a parameter space so the factory can sample, mutate and cross them over.
All arrays only use information available at the close of the current bar (no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from propfloor.strategies import indicators as ind

Space = dict[str, tuple]  # name -> ("int", lo, hi) | ("float", lo, hi) | ("choice", [..])


class Ctx:
    """Bar arrays + memoised indicators for one symbol/timeframe."""

    def __init__(self, df: pd.DataFrame, symbol: str = "EURUSD", tf: str = "H1") -> None:
        self.df = df
        self.symbol = symbol
        self.tf = tf
        self.o = df["open"].to_numpy(np.float64)
        self.h = df["high"].to_numpy(np.float64)
        self.l = df["low"].to_numpy(np.float64)
        self.c = df["close"].to_numpy(np.float64)
        self.n = len(df)
        idx = df.index
        self.time = idx
        self.t_ns = idx.as_unit("ns").asi8
        self.hour = idx.hour.to_numpy()
        ny = idx.tz_convert("America/New_York")
        rolled = ny + pd.Timedelta(hours=7)  # FX day starts 17:00 NY
        self.dow = rolled.weekday.to_numpy()
        self.day_id = (rolled.normalize().asi8 // 86_400_000_000_000).astype(np.int64)
        self.v = df["volume"].to_numpy(np.float64) if "volume" in df else np.ones(self.n)
        self.peers: dict[str, "Ctx"] = {}  # other symbols on the same timeframe (intermarket blocks)
        self.train_end: pd.Timestamp | None = None  # in-sample end; statistics-fitting blocks must not look past it
        self._cache: dict[Any, Any] = {}

    def peer_close(self, sym: str) -> np.ndarray | None:
        """Close of another symbol aligned to this context's bars (forward-filled, causal)."""
        if sym == self.symbol:
            return self.c
        p = self.peers.get(sym)
        if p is None:
            return None
        return self.get(("peer", sym), lambda: pd.Series(p.c, index=p.time).reindex(self.time, method="ffill",
                                                                                    limit=8).to_numpy())

    @property
    def train_mask(self) -> np.ndarray:
        """True for bars inside the in-sample period (default: first 75%)."""
        def f():
            if self.train_end is None:
                m = np.zeros(self.n, bool)
                m[: int(self.n * 0.75)] = True
                return m
            return np.asarray(self.time < self.train_end)
        return self.get("train_mask", f)

    def get(self, key, fn: Callable[[], Any]):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # common indicators
    def atr(self, n=14):
        return self.get(("atr", n), lambda: ind.atr(self.h, self.l, self.c, n))

    def ma(self, n, kind="ema"):
        f = ind.ema if kind == "ema" else ind.sma
        return self.get(("ma", n, kind), lambda: f(self.c, n))

    def rsi(self, n=14):
        return self.get(("rsi", n), lambda: ind.rsi(self.c, n))

    def adx(self, n=14):
        return self.get(("adx", n), lambda: ind.adx(self.h, self.l, self.c, n))

    def swings(self, k=3):
        return self.get(("swing", k), lambda: ind.swing_levels(self.h, self.l, k))

    def day_hl(self):
        """Previous FX day's high/low/close and current day open aligned to each bar."""
        def f():
            s = pd.DataFrame({"d": self.day_id, "h": self.h, "l": self.l, "c": self.c, "o": self.o})
            g = s.groupby("d").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), o=("o", "first"))
            prev = g.shift(1)
            m = prev.reindex(self.day_id)
            return (m["h"].to_numpy(), m["l"].to_numpy(), m["c"].to_numpy(),
                    g["o"].reindex(self.day_id).to_numpy())
        return self.get("day_hl", f)

    @property
    def body(self):
        return self.get("body", lambda: self.c - self.o)

    @property
    def rng(self):
        return self.get("rng", lambda: self.h - self.l)


def _s(x, n=1):
    return ind.shift(x, n)


def _b(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0).astype(bool)


@dataclass
class Block:
    name: str
    fn: Callable
    space: Space
    family: str = ""


ENTRIES: dict[str, Block] = {}
FILTERS: dict[str, Block] = {}


def entry(name, family, **space):
    def deco(fn):
        ENTRIES[name] = Block(name, fn, space, family)
        return fn
    return deco


def filt(name, **space):
    def deco(fn):
        FILTERS[name] = Block(name, fn, space, "filter")
        return fn
    return deco


# ============================================================ candlestick patterns
@entry("engulfing", "candles", min_body_atr=("float", 0.2, 1.5))
def _engulf(x: Ctx, p):
    b, pb = x.body, _s(x.body)
    big = np.abs(b) > p["min_body_atr"] * x.atr()
    bull = (pb < 0) & (b > 0) & (x.c >= _s(x.o)) & (x.o <= _s(x.c)) & big
    bear = (pb > 0) & (b < 0) & (x.c <= _s(x.o)) & (x.o >= _s(x.c)) & big
    return _b(bull), _b(bear)


@entry("pin_bar", "candles", wick_ratio=("float", 1.5, 4.0), min_range_atr=("float", 0.5, 2.0))
def _pin(x: Ctx, p):
    upper = x.h - np.maximum(x.o, x.c)
    lower = np.minimum(x.o, x.c) - x.l
    body = np.abs(x.body) + 1e-12
    ok = x.rng > p["min_range_atr"] * x.atr()
    return _b((lower > p["wick_ratio"] * body) & (lower > 2 * upper) & ok), _b((upper > p["wick_ratio"] * body) & (upper > 2 * lower) & ok)


@entry("inside_bar_breakout", "candles", mother_range_atr=("float", 0.5, 2.5))
def _inside(x: Ctx, p):
    inside_prev = (_s(x.h) < _s(x.h, 2)) & (_s(x.l) > _s(x.l, 2)) & (_s(x.rng, 2) > p["mother_range_atr"] * _s(x.atr(), 2))
    return _b(inside_prev & (x.c > _s(x.h, 2))), _b(inside_prev & (x.c < _s(x.l, 2)))


@entry("outside_bar", "candles", mode=("choice", ["follow", "fade"]))
def _outside(x: Ctx, p):
    out = (x.h > _s(x.h)) & (x.l < _s(x.l))
    up, dn = out & (x.body > 0), out & (x.body < 0)
    return (_b(up), _b(dn)) if p["mode"] == "follow" else (_b(dn), _b(up))


@entry("doji_reversal", "candles", body_frac=("float", 0.05, 0.2), trend_n=("int", 3, 10))
def _doji(x: Ctx, p):
    doji = np.abs(x.body) <= p["body_frac"] * (x.rng + 1e-12)
    n = p["trend_n"]
    up_trend = x.c > _s(x.c, n)
    return _b(doji & ~up_trend & (x.c < _s(x.c, n))), _b(doji & up_trend)


@entry("three_soldiers", "candles", min_body_atr=("float", 0.2, 1.0), mode=("choice", ["follow", "fade"]))
def _three(x: Ctx, p):
    a = x.atr()
    up = np.ones(x.n, bool)
    dn = np.ones(x.n, bool)
    for k in range(3):
        b = _s(x.body, k)
        aa = _s(a, k)
        up &= _b(b > p["min_body_atr"] * aa)
        dn &= _b(b < -p["min_body_atr"] * aa)
    return (up, dn) if p["mode"] == "follow" else (dn, up)


@entry("star_reversal", "candles", small_body=("float", 0.1, 0.4))
def _star(x: Ctx, p):
    a = x.atr()
    b1, b2, b3 = _s(x.body, 2), _s(x.body, 1), x.body
    small = np.abs(b2) < p["small_body"] * _s(a)
    morning = (b1 < -0.6 * a) & small & (b3 > 0.6 * a) & (x.c > _s(x.o, 2) + b1 / 2)
    evening = (b1 > 0.6 * a) & small & (b3 < -0.6 * a) & (x.c < _s(x.o, 2) + b1 / 2)
    return _b(morning), _b(evening)


@entry("harami", "candles", mother_body_atr=("float", 0.5, 2.0))
def _harami(x: Ctx, p):
    pb = _s(x.body)
    inside = (np.maximum(x.o, x.c) < np.maximum(_s(x.o), _s(x.c))) & (np.minimum(x.o, x.c) > np.minimum(_s(x.o), _s(x.c)))
    big = np.abs(pb) > p["mother_body_atr"] * x.atr()
    return _b(inside & big & (pb < 0) & (x.body > 0)), _b(inside & big & (pb > 0) & (x.body < 0))


@entry("tweezer", "candles", tol_atr=("float", 0.02, 0.15))
def _tweezer(x: Ctx, p):
    t = p["tol_atr"] * x.atr()
    bot = (np.abs(x.l - _s(x.l)) < t) & (_s(x.body) < 0) & (x.body > 0)
    top = (np.abs(x.h - _s(x.h)) < t) & (_s(x.body) > 0) & (x.body < 0)
    return _b(bot), _b(top)


@entry("big_candle", "candles", range_atr=("float", 1.5, 4.0), mode=("choice", ["follow", "fade"]))
def _big(x: Ctx, p):
    big = x.rng > p["range_atr"] * _s(x.atr())
    up, dn = big & (x.body > 0), big & (x.body < 0)
    return (_b(up), _b(dn)) if p["mode"] == "follow" else (_b(dn), _b(up))


@entry("body_ratio_extreme", "candles", min_ratio=("float", 0.7, 0.95), mode=("choice", ["follow", "fade"]))
def _bodyr(x: Ctx, p):
    r = np.abs(x.body) / (x.rng + 1e-12)
    strong = (r > p["min_ratio"]) & (x.rng > x.atr())
    up, dn = strong & (x.body > 0), strong & (x.body < 0)
    return (_b(up), _b(dn)) if p["mode"] == "follow" else (_b(dn), _b(up))


@entry("nr_breakout", "volatility", n=("int", 4, 10))
def _nr(x: Ctx, p):
    n = p["n"]
    nr = _s(x.rng) <= _s(ind.rolling_min(x.rng, n))
    return _b(nr & (x.c > _s(x.h))), _b(nr & (x.c < _s(x.l)))


# ============================================================ indicators
@entry("ma_cross", "trend", fast=("int", 5, 30), slow=("int", 20, 200), kind=("choice", ["ema", "sma"]))
def _macross(x: Ctx, p):
    f, s = x.ma(p["fast"], p["kind"]), x.ma(max(p["slow"], p["fast"] + 5), p["kind"])
    d = f - s
    return _b((d > 0) & (_s(d) <= 0)), _b((d < 0) & (_s(d) >= 0))


@entry("ma_pullback", "trend", trend=("int", 50, 200), fast=("int", 10, 30))
def _mapb(x: Ctx, p):
    t, f = x.ma(p["trend"]), x.ma(p["fast"])
    up = (f > t) & (_s(x.l) <= _s(f)) & (x.c > f) & (x.body > 0)
    dn = (f < t) & (_s(x.h) >= _s(f)) & (x.c < f) & (x.body < 0)
    return _b(up), _b(dn)


@entry("rsi_reversal", "mean_reversion", n=("int", 2, 21), lo=("float", 5, 35))
def _rsir(x: Ctx, p):
    r = x.rsi(p["n"])
    lo, hi = p["lo"], 100 - p["lo"]
    return _b((r > lo) & (_s(r) <= lo)), _b((r < hi) & (_s(r) >= hi))


@entry("rsi_extreme", "mean_reversion", n=("int", 2, 14), lo=("float", 3, 25))
def _rsix(x: Ctx, p):
    r = x.rsi(p["n"])
    return _b(r < p["lo"]), _b(r > 100 - p["lo"])


@entry("rsi_momentum", "momentum", n=("int", 7, 21), level=("float", 50, 65))
def _rsim(x: Ctx, p):
    r = x.rsi(p["n"])
    L = p["level"]
    return _b((r > L) & (_s(r) <= L)), _b((r < 100 - L) & (_s(r) >= 100 - L))


@entry("macd_cross", "trend", fast=("int", 6, 15), slow=("int", 18, 40), signal=("int", 5, 12), zero_side=("choice", [True, False]))
def _macd(x: Ctx, p):
    m, s, _ = x.get(("macd", p["fast"], p["slow"], p["signal"]), lambda: ind.macd(x.c, p["fast"], p["slow"], p["signal"]))
    d = m - s
    up = (d > 0) & (_s(d) <= 0)
    dn = (d < 0) & (_s(d) >= 0)
    if p["zero_side"]:
        up &= m < 0
        dn &= m > 0
    return _b(up), _b(dn)


@entry("bb_reversion", "mean_reversion", n=("int", 10, 40), k=("float", 1.5, 3.0))
def _bbr(x: Ctx, p):
    lo, mid, hi = x.get(("bb", p["n"], p["k"]), lambda: ind.bollinger(x.c, p["n"], p["k"]))
    return _b((_s(x.c) < _s(lo)) & (x.c > lo)), _b((_s(x.c) > _s(hi)) & (x.c < hi))


@entry("bb_breakout", "breakout", n=("int", 10, 40), k=("float", 1.5, 3.0))
def _bbb(x: Ctx, p):
    lo, mid, hi = x.get(("bb", p["n"], p["k"]), lambda: ind.bollinger(x.c, p["n"], p["k"]))
    return _b((x.c > hi) & (_s(x.c) <= _s(hi))), _b((x.c < lo) & (_s(x.c) >= _s(lo)))


@entry("squeeze_breakout", "volatility", n=("int", 15, 30), pct=("float", 0.05, 0.3), lookback=("int", 100, 300))
def _sq(x: Ctx, p):
    lo, mid, hi = ind.bollinger(x.c, p["n"], 2.0)
    width = (hi - lo) / mid
    rank = x.get(("bbw_rank", p["n"], p["lookback"]), lambda: ind.percentile_rank(width, p["lookback"]))
    sq = _s(rank) < p["pct"]
    return _b(sq & (x.c > hi)), _b(sq & (x.c < lo))


@entry("stoch_cross", "mean_reversion", k=("int", 5, 21), d=("int", 3, 5), zone=("float", 10, 30))
def _stoch(x: Ctx, p):
    kk, dd = x.get(("stoch", p["k"], p["d"]), lambda: ind.stochastic(x.h, x.l, x.c, p["k"], p["d"]))
    diff = kk - dd
    z = p["zone"]
    return _b((diff > 0) & (_s(diff) <= 0) & (_s(kk) < z)), _b((diff < 0) & (_s(diff) >= 0) & (_s(kk) > 100 - z))


@entry("adx_di_cross", "trend", n=("int", 10, 21), min_adx=("float", 15, 30))
def _adx(x: Ctx, p):
    a, pdi, ndi = x.adx(p["n"])
    d = pdi - ndi
    ok = a > p["min_adx"]
    return _b(ok & (d > 0) & (_s(d) <= 0)), _b(ok & (d < 0) & (_s(d) >= 0))


@entry("zscore_reversion", "mean_reversion", n=("int", 10, 60), z=("float", 1.5, 3.0))
def _z(x: Ctx, p):
    z = x.get(("z", p["n"]), lambda: ind.zscore(x.c, p["n"]))
    return _b((z > -p["z"]) & (_s(z) <= -p["z"])), _b((z < p["z"]) & (_s(z) >= p["z"]))


@entry("roc_momentum", "momentum", n=("int", 3, 40), thr_atr=("float", 0.5, 3.0))
def _roc(x: Ctx, p):
    n = p["n"]
    mv = x.c - _s(x.c, n)
    a = x.atr() * np.sqrt(n)
    up = mv > p["thr_atr"] * a
    dn = mv < -p["thr_atr"] * a
    return _b(up & ~_b(_s(up.astype(float)))), _b(dn & ~_b(_s(dn.astype(float))))


# ============================================================ breakout / structure
@entry("donchian_breakout", "breakout", n=("int", 10, 100))
def _don(x: Ctx, p):
    hh, ll = _s(ind.rolling_max(x.h, p["n"])), _s(ind.rolling_min(x.l, p["n"]))
    return _b((x.c > hh) & (_s(x.c) <= _s(hh))), _b((x.c < ll) & (_s(x.c) >= _s(ll)))


@entry("donchian_fade", "mean_reversion", n=("int", 10, 60))
def _donf(x: Ctx, p):
    hh, ll = _s(ind.rolling_max(x.h, p["n"])), _s(ind.rolling_min(x.l, p["n"]))
    return _b((x.l < ll) & (x.c > ll)), _b((x.h > hh) & (x.c < hh))


@entry("swing_break", "structure", k=("int", 2, 6))
def _swb(x: Ctx, p):
    sh, sl = x.swings(p["k"])
    return _b((x.c > sh) & (_s(x.c) <= _s(sh))), _b((x.c < sl) & (_s(x.c) >= _s(sl)))


@entry("sr_rejection", "structure", k=("int", 2, 6), tol_atr=("float", 0.1, 0.5))
def _sr(x: Ctx, p):
    sh, sl = x.swings(p["k"])
    t = p["tol_atr"] * x.atr()
    sup = (np.abs(x.l - sl) < t) & (x.c > sl) & (x.body > 0)
    res = (np.abs(x.h - sh) < t) & (x.c < sh) & (x.body < 0)
    return _b(sup), _b(res)


@entry("prev_day_break", "breakout", mode=("choice", ["follow", "fade"]))
def _pdb(x: Ctx, p):
    ph, pl, _, _ = x.day_hl()
    up = (x.c > ph) & (_s(x.c) <= ph)
    dn = (x.c < pl) & (_s(x.c) >= pl)
    return (_b(up), _b(dn)) if p["mode"] == "follow" else (_b(dn), _b(up))


@entry("session_range_breakout", "session", start=("int", 0, 6), end=("int", 6, 9), latest=("int", 10, 16))
def _orb(x: Ctx, p):
    """Breakout of the range built between start..end UTC hours, valid until `latest`."""
    key = ("orb", p["start"], p["end"])

    def f():
        s = pd.DataFrame({"d": x.day_id, "h": x.h, "l": x.l, "hr": x.hour})
        inwin = (s.hr >= p["start"]) & (s.hr < p["end"])
        g = s[inwin].groupby("d").agg(h=("h", "max"), l=("l", "min"))
        return g["h"].reindex(x.day_id).to_numpy(), g["l"].reindex(x.day_id).to_numpy()

    rh, rl = x.get(key, f)
    live = (x.hour >= p["end"]) & (x.hour < p["latest"])
    return _b(live & (x.c > rh) & (_s(x.c) <= rh)), _b(live & (x.c < rl) & (_s(x.c) >= rl))


@entry("streak_reversal", "momentum", n=("int", 3, 7))
def _streakr(x: Ctx, p):
    s = x.get("streak", lambda: ind.streak(x.c))
    return _b(s <= -p["n"]), _b(s >= p["n"])


@entry("streak_momentum", "momentum", n=("int", 2, 5))
def _streakm(x: Ctx, p):
    s = x.get("streak", lambda: ind.streak(x.c))
    return _b(s == p["n"]), _b(s == -p["n"])


@entry("gap_behaviour", "gap", min_gap_atr=("float", 0.2, 2.0), mode=("choice", ["fade", "go"]))
def _gap(x: Ctx, p):
    """Gap between consecutive bars separated by >1h (weekend/session) larger than k*ATR."""
    dt = np.diff(x.t_ns, prepend=x.t_ns[0])
    gap = x.o - _s(x.c)
    big = (dt > 3_600_000_000_000 * 2) & (np.abs(gap) > p["min_gap_atr"] * _s(x.atr()))
    up, dn = big & (gap > 0), big & (gap < 0)
    return (_b(dn), _b(up)) if p["mode"] == "fade" else (_b(up), _b(dn))


@entry("day_open_drift", "session", hour=("int", 6, 14), mode=("choice", ["follow", "fade"]))
def _dod(x: Ctx, p):
    _, _, _, dopen = x.day_hl()
    at = (x.hour == p["hour"]) & (_s(x.hour.astype(float)) != p["hour"])
    up, dn = at & (x.c > dopen), at & (x.c < dopen)
    return (_b(up), _b(dn)) if p["mode"] == "follow" else (_b(dn), _b(up))


# ============================================================ filters
@filt("none")
def _none(x: Ctx, p):
    a = np.ones(x.n, bool)
    return a, a


@filt("session", start=("int", 0, 23), length=("int", 3, 14))
def _sess(x: Ctx, p):
    hrs = (x.hour - p["start"]) % 24
    a = hrs < p["length"]
    return a, a


@filt("day_of_week", skip=("choice", [0, 1, 2, 3, 4]))
def _dow(x: Ctx, p):
    a = x.dow != p["skip"]
    return a, a


@filt("atr_regime", n=("int", 100, 500), lo=("float", 0.0, 0.5), hi=("float", 0.5, 1.0))
def _atrr(x: Ctx, p):
    r = x.get(("atr_rank", p["n"]), lambda: ind.percentile_rank(x.atr(), p["n"]))
    a = _b((r >= p["lo"]) & (r <= max(p["hi"], p["lo"] + 0.1)))
    return a, a


@filt("adx_trending", n=("int", 10, 21), min_adx=("float", 18, 35))
def _adxt(x: Ctx, p):
    a = _b(x.adx(p["n"])[0] > p["min_adx"])
    return a, a


@filt("adx_ranging", n=("int", 10, 21), max_adx=("float", 15, 25))
def _adxr(x: Ctx, p):
    a = _b(x.adx(p["n"])[0] < p["max_adx"])
    return a, a


@filt("ma_side", n=("int", 50, 300), mode=("choice", ["with", "against"]))
def _mas(x: Ctx, p):
    m = x.ma(p["n"])
    up, dn = _b(x.c > m), _b(x.c < m)
    return (up, dn) if p["mode"] == "with" else (dn, up)


@filt("mtf_trend", mult=("int", 4, 24), n=("int", 20, 60))
def _mtf(x: Ctx, p):
    """Higher-timeframe trend proxy: slope of an MA of length n*mult (e.g. H1 x 24 = D1)."""
    L = p["n"] * p["mult"]
    m = x.ma(L, "sma")
    slope = m - _s(m, p["mult"])
    return _b(slope > 0), _b(slope < 0)


@filt("candle_size", lo=("float", 0.2, 1.0), hi=("float", 1.0, 3.0))
def _cs(x: Ctx, p):
    r = x.rng / (x.atr() + 1e-12)
    a = _b((r >= p["lo"]) & (r <= p["hi"]))
    return a, a


@filt("vol_expanding", fast=("int", 5, 14), slow=("int", 30, 100))
def _ve(x: Ctx, p):
    a = _b(x.atr(p["fast"]) > x.atr(p["slow"]))
    return a, a


@filt("rsi_bias", n=("int", 7, 21))
def _rb(x: Ctx, p):
    r = x.rsi(p["n"])
    return _b(r > 50), _b(r < 50)


# ============================================================ exits & sizing spaces
EXIT_SPACE: Space = {
    "sl_type": ("choice", ["atr", "swing", "bar"]),
    "sl_atr": ("float", 0.75, 4.0),
    "tp_r": ("choice", [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
    "time_stop": ("choice", [0, 0, 6, 12, 24, 48, 72, -1]),  # -1 = signal-decay optimal hold (P5), fitted in-sample
    "trail_atr": ("choice", [0.0, 0.0, 1.5, 2.0, 3.0]),
    "trail_type": ("choice", ["close", "chandelier"]),
    "breakeven_r": ("choice", [0.0, 0.0, 0.5, 1.0, 1.5]),
    "partial_r": ("choice", [0.0, 0.0, 0.0, 0.75, 1.0, 1.5]),  # I / Q6: bank half early (trailing-floor aware)
    "exit_opposite": ("choice", [True, False]),
    "exit_mode": ("choice", ["none", "none", "none", "ema20", "sma5", "bb_mid", "rsi50", "channel", "session_end"]),
    "exit_n": ("int", 10, 20),  # opposite-channel length for exit_mode=channel
    "exit_hour": ("int", 15, 21),  # UTC hour for exit_mode=session_end
}
SIZING_SPACE: Space = {
    "risk_per_trade": ("choice", [0.001, 0.0025, 0.004, 0.005, 0.0075, 0.01]),
}
# Account-level overlays (sections J, P3/P4, Q) applied by the risk manager / challenge simulator.
OVERLAY_SPACE: Space = {
    "soft_daily_stop": ("choice", [0.33, 0.5, 0.6, 0.8]),        # share of the daily limit (1.0-2.5% of a 3% limit)
    "daily_profit_cap": ("choice", [0.0, 0.0, 0.01, 0.015, 0.02]),  # J profit lock / Q4
    "max_trades_per_day": ("choice", [0, 0, 1, 2, 3]),             # Q5
    "loss_streak_n": ("choice", [0, 0, 2, 3, 4]),                  # J losing-streak cooldown
    "loss_streak_mult": ("choice", [0.0, 0.5]),                    # 0 = pause rest of day, 0.5 = halve size
    "fast_pass_mult": ("choice", [1.0, 1.0, 1.25, 1.5]),           # Q1 (stage 1, until fast_pass_until of target banked)
    "fast_pass_until": ("choice", [0.5]),
    "buffer_risk_frac": ("choice", [0.0, 0.0, 0.1, 0.15, 0.2]),    # Q2 risk = frac x distance to trailing floor
    "derisk_within": ("choice", [0.0, 0.02]),                      # J halve risk within 2% of target
    "target_banking": ("choice", [True, False]),                   # Q3
    "anti_martingale_step": ("choice", [0.0, 0.0, 0.001, 0.0025]),  # P4 (+% risk per consecutive win)
    "equity_curve_n": ("choice", [0, 0, 0, 10, 20]),               # P3 trade only above own equity MA
}
TIMEFRAMES = ["M15", "M30", "H1", "H4", "D1"]


def exit_signals(x: "Ctx", e: dict) -> tuple[np.ndarray, np.ndarray]:
    """Signal-based exits (closing a long when xl is True, a short when xs is True)."""
    mode = e.get("exit_mode", "none")
    z = np.zeros(x.n, bool)
    if mode == "ema20":
        m = x.ma(20)
        return _b(x.c >= m), _b(x.c <= m)
    if mode == "sma5":
        m = x.ma(5, "sma")
        return _b(x.c > m), _b(x.c < m)
    if mode == "bb_mid":
        m = x.ma(20, "sma")
        return _b(x.c >= m), _b(x.c <= m)
    if mode == "rsi50":
        r = x.rsi(14)
        return _b(r >= 50), _b(r <= 50)
    if mode == "channel":
        n = int(e.get("exit_n", 10))
        return _b(x.c < _s(ind.rolling_min(x.l, n))), _b(x.c > _s(ind.rolling_max(x.h, n)))
    if mode == "session_end":
        a = x.hour == int(e.get("exit_hour", 20))
        return a, a
    return z, z


def sample_param(spec: tuple, rng: np.random.Generator):
    kind = spec[0]
    if kind == "int":
        return int(rng.integers(spec[1], spec[2] + 1))
    if kind == "float":
        return round(float(rng.uniform(spec[1], spec[2])), 4)
    choices = spec[1]
    return choices[int(rng.integers(len(choices)))]


def sample_params(space: Space, rng: np.random.Generator) -> dict:
    return {k: sample_param(v, rng) for k, v in space.items()}


# extended catalogues register themselves on import
from propfloor.strategies import blocks_ext, blocks_intermarket, blocks_meta, blocks_stat  # noqa: E402,F401
