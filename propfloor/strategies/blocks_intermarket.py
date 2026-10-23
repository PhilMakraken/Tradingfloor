"""Cross-market / intermarket blocks (catalogue sections M, H5, H6).

Each context carries `peers` (the other symbols on the same timeframe, aligned causally by
forward-fill). Blocks that need a peer the universe doesn't have return no signals, so a genome
on an unsupported symbol simply never trades (and is rejected for lack of trades).
"""
from __future__ import annotations

import numpy as np

from propfloor.strategies import indicators as ind
from propfloor.strategies.blocks import Ctx, _b, _s, entry, filt

CCYS = ["EUR", "GBP", "USD", "JPY", "AUD", "CAD", "CHF", "NZD", "XAU"]
RISK_W = {"AUD": 1.0, "NZD": 1.0, "CAD": 0.5, "GBP": 0.5, "EUR": 0.3, "USD": 0.0, "JPY": -1.0, "CHF": -1.0, "XAU": -1.0}


def legs(sym: str) -> tuple[str, str]:
    return sym[:3], sym[3:6]


def _universe(x: Ctx) -> list[str]:
    return [x.symbol] + [s for s in x.peers if s != x.symbol]


def log_rate(x: Ctx, a: str, b: str) -> np.ndarray | None:
    """log(a/b) from any available pair (direct or inverted)."""
    for s in _universe(x):
        if legs(s) == (a, b):
            c = x.peer_close(s)
            return None if c is None else np.log(c)
        if legs(s) == (b, a):
            c = x.peer_close(s)
            return None if c is None else -np.log(c)
    return None


def _zret(lc: np.ndarray, n: int, m: int = 200) -> np.ndarray:
    """n-bar log return standardised by its own rolling stdev (m windows)."""
    r = lc - ind.shift(lc, n)
    sd = ind.stdev(np.nan_to_num(r), m)
    with np.errstate(divide="ignore", invalid="ignore"):
        return r / sd


def related(x: Ctx) -> list[str]:
    """Peers sharing a currency with this symbol (same leg position first), deterministic order."""
    b, q = legs(x.symbol)
    same = sorted(s for s in x.peers if s != x.symbol and (legs(s)[1] == q or legs(s)[0] == b))
    other = sorted(s for s in x.peers if s != x.symbol and s not in same and (b in s or q in s))
    return same + other


def currency_strength(x: Ctx, n: int) -> dict[str, np.ndarray]:
    """Average n-bar log return of each currency against all its available counterparts."""
    def f():
        acc: dict[str, list[np.ndarray]] = {}
        for s in _universe(x):
            c = x.peer_close(s)
            if c is None:
                continue
            r = np.log(c) - ind.shift(np.log(c), n)
            sd = ind.stdev(np.nan_to_num(r), 500)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = r / sd
            b, q = legs(s)
            acc.setdefault(b, []).append(z)
            acc.setdefault(q, []).append(-z)
        return {k: np.nanmean(np.vstack(v), axis=0) for k, v in acc.items() if len(v) >= 2}
    return x.get(("cstr", n), f)


# ------------------------------------------------------------------ M1 gold vs USD basket
@entry("gold_usd_laggard", "intermarket", n=("int", 4, 20), thr=("float", 1.0, 3.0))
def _m1(x: Ctx, p):
    b, q = legs(x.symbol)
    z = np.zeros(x.n, bool)
    gold = log_rate(x, "XAU", "USD")
    if gold is None or "USD" not in (b, q) or x.symbol == "XAUUSD":
        return z, z
    gz = _zret(gold, p["n"])
    own = _zret(np.log(x.c), p["n"])
    usd_weak = gz > p["thr"]  # gold up strongly = USD weakness
    usd_strong = gz < -p["thr"]
    lag_ok = np.abs(own) < 0.5 * p["thr"]
    usd_quote = q == "USD"
    long_ = (usd_weak if usd_quote else usd_strong) & lag_ok
    short = (usd_strong if usd_quote else usd_weak) & lag_ok
    first = lambda a: _b(a) & ~_b(_s(a.astype(float)))  # noqa: E731
    return first(long_), first(short)


# ------------------------------------------------------------------ M2 JPY risk sentiment (filter)
@filt("jpy_risk_sentiment", n=("int", 10, 50))
def _m2(x: Ctx, p):
    a = np.ones(x.n, bool)
    b, q = legs(x.symbol)
    sigs = [lr for lr in (log_rate(x, "AUD", "JPY"), log_rate(x, "EUR", "JPY")) if lr is not None]
    if not sigs or "JPY" not in (b, q):
        return a, a
    mom = np.nanmean(np.vstack([s - ind.shift(s, p["n"]) for s in sigs]), axis=0)
    on = _b(mom > 0)
    off = _b(mom < 0)
    if q == "JPY":  # risk-on: JPY crosses up
        return on, off
    return off, on  # JPY as base (none in our set, kept for completeness)


# ------------------------------------------------------------------ M3 cross-rate triangulation
@entry("cross_triangulation", "intermarket", n=("int", 50, 300), z=("float", 2.0, 3.5))
def _m3(x: Ctx, p):
    b, q = legs(x.symbol)
    z0 = np.zeros(x.n, bool)

    def f():
        actual = np.log(x.c)
        for via in CCYS:
            if via in (b, q):
                continue
            lb, lq = log_rate(x, b, via), log_rate(x, q, via)
            if lb is not None and lq is not None:
                return actual - (lb - lq)
        return None

    spread = x.get(("tri",), f)
    if spread is None:
        return z0, z0
    zz = ind.zscore(np.nan_to_num(spread), p["n"])
    up = (zz < -p["z"]) & (_s(zz) >= -p["z"])  # actual cheap vs implied -> buy
    dn = (zz > p["z"]) & (_s(zz) <= p["z"])
    return _b(up), _b(dn)


# ------------------------------------------------------------------ M4 lead-lag pairs
@entry("lead_lag", "intermarket", leader=("int", 0, 2), lag=("int", 1, 3), thr=("float", 1.0, 2.5))
def _m4(x: Ctx, p):
    rel = related(x)
    z0 = np.zeros(x.n, bool)
    if len(rel) <= p["leader"]:
        return z0, z0
    lead = x.peer_close(rel[p["leader"]])
    sign = 1.0
    lb, lq = legs(rel[p["leader"]])
    b, q = legs(x.symbol)
    if lb == q or lq == b:  # leader quoted the other way round relative to us
        sign = -1.0
    lz = sign * _zret(np.log(lead), p["lag"])
    own = _zret(np.log(x.c), p["lag"])
    up = (lz > p["thr"]) & (own < 0.3 * p["thr"])
    dn = (lz < -p["thr"]) & (own > -0.3 * p["thr"])
    return _b(up) & ~_b(_s(up.astype(float))), _b(dn) & ~_b(_s(dn.astype(float)))


# ------------------------------------------------------------------ M5 CHF / gold safe-haven (filter)
@filt("safe_haven_confirm", n=("int", 5, 30))
def _m5(x: Ctx, p):
    a = np.ones(x.n, bool)
    chf = log_rate(x, "USD", "CHF")
    gold = log_rate(x, "XAU", "USD")
    if chf is None or gold is None:
        return a, a
    dchf = chf - ind.shift(chf, p["n"])
    dg = gold - ind.shift(gold, p["n"])
    risk_off = _b((dchf < 0) & (dg > 0))
    risk_on = _b((dchf > 0) & (dg < 0))
    b, q = legs(x.symbol)
    w = RISK_W.get(b, 0) - RISK_W.get(q, 0)
    if w > 0:  # long = risk-on exposure
        return ~risk_off, ~risk_on
    if w < 0:
        return ~risk_on, ~risk_off
    return a, a


# ------------------------------------------------------------------ M6 / H6 currency strength
@entry("basket_strength", "intermarket", n=("int", 10, 50), persist=("int", 1, 10), mode=("choice", ["strict", "top2"]))
def _m6(x: Ctx, p):
    cs = currency_strength(x, p["n"])
    b, q = legs(x.symbol)
    z0 = np.zeros(x.n, bool)
    if b not in cs or q not in cs or len(cs) < 4:
        return z0, z0
    names = list(cs)
    mat = np.vstack([np.nan_to_num(cs[k], nan=0.0) for k in names])
    order = np.argsort(-mat, axis=0)  # rank 0 = strongest
    rank = np.empty_like(order)
    for r in range(order.shape[0]):
        rank[order[r], np.arange(mat.shape[1])] = r
    rb, rq = rank[names.index(b)], rank[names.index(q)]
    last = len(names) - 1
    top = 0 if p["mode"] == "strict" else 1
    strong_b = (rb <= top) & (rq >= last - top)
    strong_q = (rq <= top) & (rb >= last - top)
    k = p["persist"]
    pers_b = _b(ind.rolling_min(strong_b.astype(float), k))
    pers_q = _b(ind.rolling_min(strong_q.astype(float), k))
    return pers_b & ~_b(_s(pers_b.astype(float))), pers_q & ~_b(_s(pers_q.astype(float)))


# ------------------------------------------------------------------ H5 correlation divergence
@entry("pair_divergence", "intermarket", peer=("int", 0, 2), n=("int", 10, 50), z=("float", 2.0, 3.0))
def _h5(x: Ctx, p):
    rel = related(x)
    z0 = np.zeros(x.n, bool)
    if len(rel) <= p["peer"]:
        return z0, z0
    ps = rel[p["peer"]]
    pc = x.peer_close(ps)
    b, q = legs(x.symbol)
    pb, pq = legs(ps)
    sign = -1.0 if (pb == q or pq == b) else 1.0
    d = (np.log(x.c) - ind.shift(np.log(x.c), p["n"])) - sign * (np.log(pc) - ind.shift(np.log(pc), p["n"]))
    zz = ind.zscore(np.nan_to_num(d), 500)
    return _b((zz < -p["z"]) & (_s(zz) >= -p["z"])), _b((zz > p["z"]) & (_s(zz) <= p["z"]))


# ------------------------------------------------------------------ M7 correlation breakdown
@entry("correlation_breakdown", "intermarket", peer=("int", 0, 2), n=("int", 50, 100), thr=("float", 0.1, 0.4))
def _m7(x: Ctx, p):
    import pandas as pd

    rel = related(x)
    z0 = np.zeros(x.n, bool)
    if len(rel) <= p["peer"]:
        return z0, z0
    ps = rel[p["peer"]]
    pc = x.peer_close(ps)
    b, q = legs(x.symbol)
    pb, pq = legs(ps)
    sign = -1.0 if (pb == q or pq == b) else 1.0
    r1 = pd.Series(np.diff(np.log(x.c), prepend=np.nan))
    r2 = pd.Series(sign * np.diff(np.log(pc), prepend=np.nan))
    short_c = r1.rolling(p["n"]).corr(r2).to_numpy()
    long_c = r1.rolling(p["n"] * 10).corr(r2).to_numpy()
    broke = (long_c > 0.6) & (short_c < p["thr"]) & (_s(short_c) >= p["thr"])
    gap = sign * (np.log(pc) - ind.shift(np.log(pc), p["n"])) - (np.log(x.c) - ind.shift(np.log(x.c), p["n"]))
    return _b(broke & (gap > 0)), _b(broke & (gap < 0))  # we lagged the peer -> expect catch-up
