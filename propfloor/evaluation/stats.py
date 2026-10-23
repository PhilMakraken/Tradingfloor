"""Performance statistics and multiple-testing corrections."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1))


def moments(r: np.ndarray) -> tuple[float, float]:
    r = np.asarray(r, float)
    if len(r) < 3 or r.std() == 0:
        return 0.0, 3.0
    z = (r - r.mean()) / r.std()
    return float((z**3).mean()), float((z**4).mean())


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012): P(true SR > benchmark) given non-normal returns."""
    r = np.asarray(r, float)
    T = len(r)
    if T < 3:
        return 0.0
    sr = sharpe(r)
    sk, ku = moments(r)
    denom = math.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr**2, 1e-12))
    return _N.cdf((sr - sr_benchmark) * math.sqrt(T - 1) / denom)


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum SR among n_trials independent zero-skill strategies."""
    n = max(int(n_trials), 2)
    return math.sqrt(max(var_sr, 1e-12)) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n)
                                             + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n * math.e)))


def deflated_sharpe(r: np.ndarray, n_trials: int, var_sr: float | None = None) -> float:
    """DSR (Bailey & Lopez de Prado 2014): PSR against the expected max SR of n_trials."""
    r = np.asarray(r, float)
    if var_sr is None:
        var_sr = 1.0 / max(len(r) - 1, 1)  # sampling variance of SR under the null
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, var_sr))


def profit_factor(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 else float("inf") if (r > 0).any() else 0.0


def max_drawdown_r(r: np.ndarray) -> float:
    eq = np.cumsum(np.asarray(r, float))
    return float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()) if len(eq) else 0.0


def trade_summary(r: np.ndarray) -> dict:
    r = np.asarray(r, float)
    return {
        "n": int(len(r)),
        "expectancy_r": float(r.mean()) if len(r) else 0.0,
        "win_rate": float((r > 0).mean()) if len(r) else 0.0,
        "profit_factor": profit_factor(r),
        "sharpe_trade": sharpe(r),
        "max_dd_r": max_drawdown_r(r),
    }
