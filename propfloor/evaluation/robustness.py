"""Monte Carlo robustness / anti-overfitting tests run on candidates that already cleared the IS/WF/OOS gates.

bootstrap        - resample the trade list with replacement: P(total R > 0), 5th percentile total R and the 95th
                   percentile max drawdown. A real edge stays profitable under almost every resampling; one lucky
                   streak does not.
random_entries   - Monte Carlo permutation test: the same exits on the same history, but with entries placed at
                   random bars (same count and long/short mix). p-value = share of random runs whose expectancy is
                   at least the strategy's. If random entries do as well, the "edge" is the exit/market drift.
neighbourhood    - parameter perturbation: every numeric entry/filter parameter moved by up to +/-`jitter`. A robust
                   strategy sits on a plateau (most neighbours still profitable), an overfit one on a spike.
yearly           - calendar-year breakdown of the full history: years covered and share of profitable years.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from propfloor.evaluation.stats import max_drawdown_r
from propfloor.strategies.backtest import run_backtest
from propfloor.strategies.blocks import ENTRIES, FILTERS, Ctx
from propfloor.strategies.genome import StrategyGenome, signal_decay_hold


def bootstrap(r: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return {"p_profit": 0.0, "total_p5": 0.0, "maxdd_p95": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n, len(r)))
    samples = r[idx]
    totals = samples.sum(1)
    dd = np.array([max_drawdown_r(s) for s in samples])
    return {"p_profit": float((totals > 0).mean()), "total_p5": float(np.percentile(totals, 5)),
            "total_median": float(np.median(totals)), "maxdd_p95": float(np.percentile(dd, 95))}


def _exit_params(g: StrategyGenome, ctx: Ctx, lo: np.ndarray, sh: np.ndarray) -> dict:
    ex = g.exit
    if int(ex.get("time_stop", 0)) == -1:
        ex = {**ex, "time_stop": signal_decay_hold(ctx, lo, sh)}
    return ex


def random_entries(g: StrategyGenome, ctx: Ctx, costs: dict, n: int = 100, seed: int = 0) -> dict:
    lo, sh = g.signals(ctx)
    ex = _exit_params(g, ctx, lo, sh)
    real = run_backtest(ctx, lo, sh, ex, **costs)["pnl_r"].to_numpy()
    if len(real) < 5:
        return {"p_value": 1.0, "real_exp": float(real.mean()) if len(real) else 0.0, "random_exp_p95": 0.0}
    n_long, n_short = int(lo.sum()), int(sh.sum())
    rng = np.random.default_rng(seed)
    exps = np.empty(n)
    for k in range(n):
        rl = np.zeros(ctx.n, bool)
        rs = np.zeros(ctx.n, bool)
        pick = rng.choice(ctx.n - 1, size=min(n_long + n_short, ctx.n - 1), replace=False)
        rl[pick[:n_long]] = True
        rs[pick[n_long:]] = True
        t = run_backtest(ctx, rl, rs, ex, **costs)["pnl_r"].to_numpy()
        exps[k] = t.mean() if len(t) else 0.0
    real_exp = float(real.mean())
    return {"p_value": float((1 + (exps >= real_exp).sum()) / (n + 1)), "real_exp": real_exp,
            "random_exp_mean": float(exps.mean()), "random_exp_p95": float(np.percentile(exps, 95))}


def _jitter(params: dict, space: dict, rng: np.random.Generator, jitter: float) -> dict:
    out = dict(params)
    for k, v in params.items():
        spec = space.get(k)
        if not spec or spec[0] not in ("int", "float"):
            continue
        val = v * (1 + rng.uniform(-jitter, jitter))
        lo, hi = spec[1], spec[2]
        val = min(max(val, lo), hi)
        out[k] = int(round(val)) if spec[0] == "int" else float(val)
    return out


def neighbourhood(g: StrategyGenome, ctx: Ctx, costs: dict, n: int = 16, jitter: float = 0.15, seed: int = 0) -> dict:
    """In-sample expectancy of `n` perturbed copies (train_mask = the in-sample period)."""
    rng = np.random.default_rng(seed)
    cut = ctx.time[ctx.train_mask][-1] if ctx.train_mask.any() else ctx.time[-1]
    exps = []
    for _ in range(n):
        h = copy.deepcopy(g)
        h.entry.params = _jitter(h.entry.params, ENTRIES[h.entry.name].space, rng, jitter)
        for f in h.filters:
            f.params = _jitter(f.params, FILTERS[f.name].space, rng, jitter)
        tr = h.backtest(ctx, **costs)
        r = tr.loc[tr.exit_time <= cut, "pnl_r"].to_numpy()
        exps.append(float(r.mean()) if len(r) >= 10 else 0.0)
    exps = np.array(exps)
    return {"positive_share": float((exps > 0).mean()), "median_exp": float(np.median(exps)),
            "worst_exp": float(exps.min())}


def yearly(tr: pd.DataFrame, data_start: pd.Timestamp, data_end: pd.Timestamp) -> dict:
    years = (data_end - data_start).days / 365.25
    if tr.empty:
        return {"years": round(years, 2), "by_year": {}, "positive_years": 0.0}
    by = tr.groupby(tr.exit_time.dt.year)["pnl_r"].sum()
    return {"years": round(years, 2), "by_year": {int(k): round(float(v), 2) for k, v in by.items()},
            "positive_years": float((by > 0).mean())}
