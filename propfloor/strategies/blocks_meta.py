"""Meta / ensemble blocks (catalogue section P, plus H4 volatility-regime switching).

P1 vote_ensemble      - trade when >= k of 5 member signals agree within a short window
P2 regime_router      - classify regime (ADX / Hurst / choppiness / ATR percentile) and route to a
                        trend-family or reversion-family member (H4 = detector "atr_pct")
P6 ml_classifier      - gradient boosting on block outputs, strictly walk-forward (expanding window)
P3/P4 are account overlays (risk.py / challenge.py), P5 is exit time_stop=-1 (genome.py),
P7 is monthly rotation on the trading floor (floor.py).
"""
from __future__ import annotations

import logging

import numpy as np
from numba import njit

from propfloor.strategies import indicators as ind
from propfloor.strategies.blocks import ENTRIES, Ctx, _b, entry

log = logging.getLogger(__name__)


def _mid(spec):
    if spec[0] == "int":
        return int(round((spec[1] + spec[2]) / 2))
    if spec[0] == "float":
        return round((spec[1] + spec[2]) / 2, 4)
    return spec[1][0]


def default_params(name: str) -> dict:
    return {k: _mid(v) for k, v in ENTRIES[name].space.items()}


TREND_MEMBERS = ["ma_cross", "donchian_breakout", "supertrend_flip", "keltner_breakout", "macd_hist_trend",
                 "swing_break", "ema_stack_pullback", "ichimoku_break", "kalman_slope", "bos_pullback"]
REVERSION_MEMBERS = ["rsi_reversal", "bb_reversion", "zscore_reversion", "stoch_cross", "donchian_fade",
                     "connors_rsi2", "vwap_deviation", "turtle_soup", "linreg_channel_fade", "pdh_sweep_reverse"]
PATTERN_MEMBERS = ["engulfing", "pin_bar", "three_bar_reversal", "inside_bar_breakout", "outside_bar_reversal"]
MEMBERS = TREND_MEMBERS + REVERSION_MEMBERS + PATTERN_MEMBERS


def member_signal(x: Ctx, name: str):
    return x.get(("member", name), lambda: ENTRIES[name].fn(x, default_params(name)))


@entry("vote_ensemble", "ensemble", m1=("choice", MEMBERS), m2=("choice", MEMBERS), m3=("choice", MEMBERS),
       m4=("choice", MEMBERS), m5=("choice", MEMBERS), min_votes=("int", 2, 4), window=("int", 1, 3))
def _p1(x: Ctx, p):
    names = list(dict.fromkeys(p[f"m{i}"] for i in range(1, 6)))  # unique members
    vl = np.zeros(x.n)
    vs = np.zeros(x.n)
    for nm in names:
        lo, sh = member_signal(x, nm)
        vl += ind.rolling_max(lo.astype(float), p["window"]) > 0
        vs += ind.rolling_max(sh.astype(float), p["window"]) > 0
    k = min(p["min_votes"], len(names))
    up = (vl >= k) & (vs < k)
    dn = (vs >= k) & (vl < k)
    first = lambda a: a & ~np.r_[False, a[:-1]]  # noqa: E731
    return first(up), first(dn)


def regime(x: Ctx, detector: str) -> np.ndarray:
    """+1 trending / high-vol, -1 ranging / low-vol, 0 undecided."""
    def f():
        if detector == "adx":
            a = x.adx(14)[0]
            return np.where(a > 25, 1, np.where(a < 20, -1, 0))
        if detector == "hurst":
            h = ind.hurst(x.c, 200)
            return np.where(h > 0.55, 1, np.where(h < 0.45, -1, 0))
        if detector == "chop":
            c = ind.choppiness(x.h, x.l, x.c, 14)
            return np.where(c < 45, 1, np.where(c > 60, -1, 0))
        r = ind.percentile_rank(x.atr(), 500)  # H4: ATR percentile > 60% -> trend, < 40% -> reversion
        return np.where(r > 0.6, 1, np.where(r < 0.4, -1, 0))
    return x.get(("regime", detector), f)


@entry("regime_router", "ensemble", detector=("choice", ["adx", "hurst", "chop", "atr_pct"]),
       trend=("choice", TREND_MEMBERS), reversion=("choice", REVERSION_MEMBERS))
def _p2(x: Ctx, p):
    rg = regime(x, p["detector"])
    tl, ts = member_signal(x, p["trend"])
    rl, rs = member_signal(x, p["reversion"])
    return _b((rg > 0) & tl | (rg < 0) & rl), _b((rg > 0) & ts | (rg < 0) & rs)


# ------------------------------------------------------------------ P6 ML classifier
@njit(cache=True)
def _labels(h, l, c, atr, horizon, k):
    """1 if +k*ATR is reached before -k*ATR within horizon bars (long label); mirror for short."""
    n = c.shape[0]
    lo = np.full(n, np.nan)
    sh = np.full(n, np.nan)
    for i in range(n - horizon - 1):
        if np.isnan(atr[i]):
            continue
        up = c[i] + k * atr[i]
        dn = c[i] - k * atr[i]
        res_l = 0.0
        res_s = 0.0
        dl = False
        ds = False
        for j in range(i + 1, i + horizon + 1):
            if not dl:
                if l[j] <= dn:
                    dl = True
                elif h[j] >= up:
                    res_l = 1.0
                    dl = True
            if not ds:
                if h[j] >= up:
                    ds = True
                elif l[j] <= dn:
                    res_s = 1.0
                    ds = True
            if dl and ds:
                break
        lo[i] = res_l
        sh[i] = res_s
    return lo, sh


def ml_features(x: Ctx) -> np.ndarray:
    def f():
        cols = []
        for nm in MEMBERS:
            lo, sh = member_signal(x, nm)
            cols.append(ind.rolling_max(lo.astype(float), 3) - ind.rolling_max(sh.astype(float), 3))
        a = x.atr()
        cols += [x.rsi(14) / 100, x.adx(14)[0] / 100, ind.percentile_rank(a, 500), x.hour / 24, x.dow / 7,
                 ind.zscore(x.c, 20), (x.c - x.ma(50)) / a, (x.c - x.ma(200)) / a, x.rng / a, x.body / a,
                 ind.choppiness(x.h, x.l, x.c, 14) / 100, ind.clv(x.h, x.l, x.c)]
        return np.nan_to_num(np.vstack(cols).T.astype(np.float32))
    return x.get("ml_X", f)


BARS_PER_YEAR = {"M5": 72_000, "M15": 24_000, "M30": 12_000, "H1": 6_000, "H4": 1_500, "D1": 260}


def ml_probabilities(x: Ctx, horizon: int, fold_len: int | None = None, max_train: int = 40_000):
    """Walk-forward probabilities: each block of `fold_len` bars is predicted by a model trained only on
    earlier bars, with the last `horizon` training bars dropped so no label overlaps the prediction period.
    Fold boundaries are fixed bar counts, so predictions never depend on how much later data exists."""
    def f():
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError:
            log.warning("scikit-learn missing; ml_classifier disabled")
            return np.zeros(x.n), np.zeros(x.n)
        fl = fold_len or max(BARS_PER_YEAR.get(x.tf, 6_000), 250)  # retrain yearly; fixed -> causal
        X = ml_features(x)
        yl, ys = _labels(x.h, x.l, x.c, x.atr(), horizon, 1.0)
        pl = np.zeros(x.n)
        ps = np.zeros(x.n)
        edges = np.r_[np.arange(0, x.n, fl), x.n]
        rng = np.random.default_rng(0)
        for fi in range(1, len(edges) - 1):
            tr_end = edges[fi] - horizon - 1
            if tr_end < min(2000, fl):
                continue
            idx = np.arange(200, tr_end)
            if len(idx) > max_train:
                idx = np.sort(rng.choice(idx, max_train, replace=False))
            sl = slice(edges[fi], edges[fi + 1])
            for y, out in ((yl, pl), (ys, ps)):
                m = ~np.isnan(y[idx])
                if m.sum() < 1000 or len(np.unique(y[idx][m])) < 2:
                    continue
                clf = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, max_leaf_nodes=15,
                                                     l2_regularization=1.0, random_state=0)
                clf.fit(X[idx][m], y[idx][m])
                out[sl] = clf.predict_proba(X[sl])[:, 1]
        return pl, ps
    return x.get(("ml_p", horizon), f)


@entry("ml_classifier", "ensemble", horizon=("choice", [8, 16, 32]), thr=("float", 0.55, 0.7))
def _p6(x: Ctx, p):
    pl, ps = ml_probabilities(x, p["horizon"])
    up = (pl > p["thr"]) & (pl > ps)
    dn = (ps > p["thr"]) & (ps > pl)
    first = lambda a: a & ~np.r_[False, a[:-1]]  # noqa: E731
    return first(up), first(dn)
