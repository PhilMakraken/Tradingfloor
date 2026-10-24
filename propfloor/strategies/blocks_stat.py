"""Statistical path-feature blocks (user idea workbook "XAUUSD 5000 Unique Entry Models", High-feasibility
OHLC-only families F01-F03, F11, F13, F14, F16, F19, F21, F25, F26, F28, F29, F48).

Instead of 5,000 fixed combinations the workbook's grid (feature family x transformation x trigger) is
collapsed into one parameterised entry and one filter:

stat_feature  - feature x window -> V01 robust z-score F = (x - rolling median) / (1.4826 * rolling MAD)
                -> direction map (learned / momentum / fade) -> T01 score-threshold cross through +-z_thr
stat_regime   - filter: pass only while the feature's robust z-score is above (high) / below (low) zero

Every feature at bar t uses bars <= t only (rolling windows ending at t). The "learned" direction is the
sign of the correlation between z and the next-h-bar return measured ONLY on in-sample bars (ctx.train_mask,
with the h-bar label also inside the training period), so it is fixed for the whole series.
Families needing ticks, news, forecasts or external series (F04-F10, F12, F15, F17-F18, F20, F22-F24,
F27, F30-F47, F49-F50) are not implemented here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from propfloor.strategies import indicators as ind
from propfloor.strategies.blocks import Ctx, _b, _s, entry, filt

ZWIN_MIN = 250  # robust z-score reference window (bars), at least 2x the feature window
LN2 = np.log(2.0)


# ------------------------------------------------------------------ helpers
def _lr(x: Ctx) -> np.ndarray:
    """1-bar log return (NaN on the first bar)."""
    return x.get("stat_lr", lambda: np.r_[np.nan, np.diff(np.log(x.c))])


def _roll(a, w, how="sum", minp=None):
    r = pd.Series(a).rolling(w, min_periods=minp or w)
    return getattr(r, how)().to_numpy()


def _code_counts(code: np.ndarray, valid: np.ndarray, k: int, w: int) -> np.ndarray:
    """Rolling counts (n x k) of integer codes over the last w bars; NaN rows until w valid codes exist."""
    oh = np.zeros((len(code), k), np.int64)
    oh[np.where(valid)[0], code[valid]] = 1
    cs = np.cumsum(oh, axis=0)
    out = cs.astype(float)
    out[w:] -= cs[:-w]
    nv = np.cumsum(valid.astype(np.int64))
    nvw = nv.copy()
    nvw[w:] -= nv[:-w]
    out[nvw < w] = np.nan
    return out


def _entropy(counts: np.ndarray, k: int) -> np.ndarray:
    tot = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / tot
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    h[np.isnan(tot[:, 0])] = np.nan
    return h / np.log(k)


def _safe(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    return np.where(np.isfinite(a), a, np.nan)


# ------------------------------------------------------------------ feature functions (x, w) -> array
def f_fractal_dim(x: Ctx, w: int) -> np.ndarray:
    """F01 Katz fractal dimension of the log-price path over the last w steps (1 = straight line, high = rough)."""
    y = np.log(x.c)
    L = _roll(np.abs(np.r_[np.nan, np.diff(y)]), w)
    y0 = _s(y, w)
    d = np.fmax(_roll(y, w, "max") - y0, y0 - _roll(y, w, "min"))
    with np.errstate(divide="ignore", invalid="ignore"):
        return _safe(np.log10(w) / (np.log10(w) + np.log10(d / L)))


def f_hurst(x: Ctx, w: int) -> np.ndarray:
    """F02 variance-ratio Hurst exponent (4-bar vs 1-bar return variance)."""
    return _safe(ind.hurst(x.c, w, 4))


def f_perm_entropy(x: Ctx, w: int) -> np.ndarray:
    """F03 normalised permutation entropy, embedding 3 (6 ordinal patterns; ties fixed as 'not less')."""
    c = x.c
    a, b = _s(c, 2), _s(c, 1)
    valid = np.isfinite(a)
    code = ((a < b).astype(int) + 2 * (b < c) + 4 * (a < c))
    lut = np.full(8, -1)
    lut[[0, 1, 2, 5, 6, 7]] = np.arange(6)  # the six realisable (a<b, b<c, a<c) combinations
    code = lut[code]
    valid &= code >= 0
    return _entropy(_code_counts(np.where(valid, code, 0), valid, 6, w), 6)


def f_acf_sum(x: Ctx, w: int, k: int = 5) -> np.ndarray:
    """F11 autocorrelation fingerprint: sum of lag-1..k return autocorrelations over the window."""
    r = pd.Series(_lr(x))
    out = np.zeros(x.n)
    for lag in range(1, k + 1):
        out += r.rolling(w, min_periods=w).corr(r.shift(lag)).to_numpy()
    return _safe(out)


def f_skew(x: Ctx, w: int) -> np.ndarray:
    """F13 rolling return skewness."""
    return _safe(pd.Series(_lr(x)).rolling(w, min_periods=w).skew().to_numpy())


def f_kurt(x: Ctx, w: int) -> np.ndarray:
    """F14 rolling excess kurtosis of returns winsorised at 5 rolling-std (fixed policy)."""
    r = _lr(x)
    sd = _s(_roll(r, w, "std"), 1)
    rw = np.where(np.isfinite(sd), np.clip(r, -5 * sd, 5 * sd), r)
    return _safe(pd.Series(rw).rolling(w, min_periods=w).kurt().to_numpy())


def f_semivar(x: Ctx, w: int) -> np.ndarray:
    """F16 realized semivariance balance (RS+ - RS-) / RV over the window, in [-1, 1]."""
    r = _lr(x)
    r2 = r * r
    up = _roll(np.where(r > 0, r2, 0.0), w)
    dn = _roll(np.where(r < 0, r2, 0.0), w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _safe((up - dn) / (up + dn))


def f_vol_of_vol(x: Ctx, w: int) -> np.ndarray:
    """F19 vol-of-vol shock: change in log realized vol over a nested sub-window s = max(5, w//4)."""
    s = max(5, w // 4)
    with np.errstate(divide="ignore"):
        lv = np.log(_roll(_lr(x), s, "std"))
    return _safe(lv - _s(lv, s))


def f_dd_hazard(x: Ctx, w: int) -> np.ndarray:
    """F21 drawdown depth from the rolling w-bar peak in volatility units (<= 0; z vs its own history)."""
    y = np.log(x.c)
    sd = _roll(_lr(x), w, "std")
    with np.errstate(divide="ignore", invalid="ignore"):
        return _safe((y - _roll(y, w, "max")) / (sd * np.sqrt(w)))


def f_range_disagree(x: Ctx, w: int) -> np.ndarray:
    """F25 log ratio of range-based (mean of Parkinson and Garman-Klass) to close-close variance."""
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(x.h / x.l) ** 2
        co = np.log(x.c / x.o) ** 2
        pk = hl / (4 * LN2)
        gk = 0.5 * hl - (2 * LN2 - 1) * co
        cc = _lr(x) ** 2
        return _safe(np.log(_roll(0.5 * (pk + gk), w, "mean") / _roll(cc, w, "mean")))


def f_path_eff(x: Ctx, w: int) -> np.ndarray:
    """F26 OHLC path efficiency: |net log move| / (gaps + minimal intrabar path 2*(h-l) - |c-o|), in [0, 1]."""
    lo_, lh, ll, lc = np.log(x.o), np.log(x.h), np.log(x.l), np.log(x.c)
    seg = np.abs(lo_ - _s(lc)) + 2 * (lh - ll) - np.abs(lc - lo_)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _safe(np.abs(lc - _s(lc, w)) / _roll(seg, w))


def f_run_length(x: Ctx, w: int) -> np.ndarray:
    """F28 log mean run length of return signs over the window (high = persistent streaks)."""
    sg = np.sign(np.nan_to_num(_lr(x)))
    ch = np.r_[0.0, (sg[1:] != sg[:-1]) & (sg[1:] != 0)].astype(float)
    ch[0] = np.nan
    return _safe(np.log(w / (_roll(ch, w) + 1.0)))


def f_sign_entropy(x: Ctx, w: int, m: int = 3) -> np.ndarray:
    """F29 normalised Shannon entropy of return-sign m-grams (low = predictable grammar)."""
    r = _lr(x)
    up = (r > 0).astype(int)
    code = up.copy()
    valid = np.isfinite(r)
    for k in range(1, m):
        code += np.nan_to_num(_s(up.astype(float), k)).astype(int) << k
        valid &= np.isfinite(_s(r, k))
    return _entropy(_code_counts(np.where(valid, code, 0), valid, 1 << m, w), 1 << m)


def f_xagg_acf(x: Ctx, w: int) -> np.ndarray:
    """F48 cross-aggregation consistency: lag-1 autocorr of 1-bar returns minus lag-1 autocorr of
    4-bar returns (on the 4x aggregation, 4x window)."""
    r1 = pd.Series(_lr(x))
    lc = pd.Series(np.log(x.c))
    r4 = lc - lc.shift(4)
    a1 = r1.rolling(w, min_periods=w).corr(r1.shift(1)).to_numpy()
    a4 = r4.rolling(4 * w, min_periods=4 * w).corr(r4.shift(4)).to_numpy()
    return _safe(a1 - a4)


FEATURES = {
    "fractal_dim": f_fractal_dim, "hurst": f_hurst, "perm_entropy": f_perm_entropy, "acf_sum": f_acf_sum,
    "skew": f_skew, "kurt": f_kurt, "semivar": f_semivar, "vol_of_vol": f_vol_of_vol, "dd_hazard": f_dd_hazard,
    "range_disagree": f_range_disagree, "path_eff": f_path_eff, "run_length": f_run_length,
    "sign_entropy": f_sign_entropy, "xagg_acf": f_xagg_acf,
}
FEATURE_NAMES = list(FEATURES)


def feature(x: Ctx, name: str, w: int) -> np.ndarray:
    return x.get(("statf", name, w), lambda: FEATURES[name](x, int(w)))


def robust_z(a: np.ndarray, zw: int) -> np.ndarray:
    """V01: (x - rolling median) / (1.4826 * rolling median |x - rolling median|), trailing windows only."""
    s = pd.Series(a)
    med = s.rolling(zw, min_periods=zw // 2).median()
    mad = (s - med).abs().rolling(zw, min_periods=zw // 2).median()
    with np.errstate(divide="ignore", invalid="ignore"):
        return _safe(((s - med) / (1.4826 * mad)).to_numpy())


def feature_z(x: Ctx, name: str, w: int) -> np.ndarray:
    w = int(w)
    return x.get(("statz", name, w), lambda: robust_z(feature(x, name, w), max(ZWIN_MIN, 2 * w)))


def horizon(w: int) -> int:
    """Forecast horizon / recent-return lookback tied to the feature window."""
    return max(3, int(w) // 4)


def learned_sign(x: Ctx, name: str, w: int) -> float:
    """Sign of corr(z_t, log c_{t+h} - log c_t) over bars t with t and t+h both in the training period."""
    def f():
        z = feature_z(x, name, w)
        h = horizon(w)
        lc = np.log(x.c)
        fwd = np.r_[lc[h:] - lc[:-h], np.full(h, np.nan)]
        tm = x.train_mask
        ok = tm & np.r_[tm[h:], np.zeros(h, bool)] & np.isfinite(z) & np.isfinite(fwd)
        if ok.sum() < 200:
            return 0.0
        zz, ff = z[ok], fwd[ok]
        if zz.std() == 0 or ff.std() == 0:
            return 0.0
        return float(np.sign(np.corrcoef(zz, ff)[0, 1]))
    return x.get(("stat_learn", name, int(w)), f)


def _score(x: Ctx, p) -> np.ndarray:
    z = feature_z(x, p["feature"], p["window"])
    if p["mode"] == "learned":
        return z * learned_sign(x, p["feature"], p["window"])
    lc = np.log(x.c)
    d = np.sign(np.nan_to_num(lc - _s(lc, horizon(p["window"]))))
    return z * d * (1.0 if p["mode"] == "momentum" else -1.0)


# ================================================================== blocks
@entry("stat_feature", "statistical", feature=("choice", FEATURE_NAMES), window=("int", 20, 200),
       z_thr=("float", 1.0, 2.5), mode=("choice", ["learned", "momentum", "fade"]))
def _stat_feature(x: Ctx, p):
    """T01 score-threshold cross: long when S crosses up through +z_thr, short when it crosses down through -z_thr."""
    s = _score(x, p)
    ps = _s(s)
    t = p["z_thr"]
    return _b((s > t) & (ps <= t)), _b((s < -t) & (ps >= -t))


@filt("stat_regime", feature=("choice", FEATURE_NAMES), window=("int", 20, 200), side=("choice", ["high", "low"]))
def _stat_regime(x: Ctx, p):
    z = feature_z(x, p["feature"], p["window"])
    ok = _b(z > 0) if p["side"] == "high" else _b(z < 0)
    return ok, ok
