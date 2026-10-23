"""Pooled multi-symbol deflated-Sharpe diagnostic for ONE pre-specified rule (read-only; one more trial itself).

Backtests the same genome (default: XAUUSD H4 donchian(20)+ma_side(200) near-miss) on every symbol, splits
IS/OOS exactly like Evaluator.validate, and reports per-symbol and pooled stats: trades, mean/std R,
per-trade Sharpe, PSR, DSR (n_trials from the meta table), walk-forward fold signs of the pooled IS series
ordered by exit time, pooled OOS expectancy, plus a weekly-aggregated pooled series (correlation-robust check:
trades on correlated symbols at the same time are not independent observations).

usage: python scripts/portfolio_check.py [--symbols A,B] [--cache DIR] [--genome JSON]
"""
import argparse
import gc
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from propfloor.config import DEFAULT_PROGRAM, load_program
from propfloor.data.store import DataStore
from propfloor.evaluation.stats import (deflated_sharpe, expected_max_sharpe, moments, probabilistic_sharpe,
                                        sharpe)
from propfloor.evaluation.walkforward import Evaluator, Universe
from propfloor.strategies.genome import StrategyGenome

GOLD_NEAR_MISS = {
    "tf": "H4", "entry": {"name": "donchian_breakout", "params": {"n": 20}},
    "filters": [{"name": "ma_side", "params": {"n": 200, "mode": "with"}}],
    "exit": {"sl_type": "atr", "sl_atr": 2.0, "tp_r": 2.0, "time_stop": 0, "trail_atr": 2.0, "trail_type": "close",
             "breakeven_r": 0.0, "partial_r": 0.75, "exit_opposite": False, "exit_mode": "none", "exit_n": 10,
             "exit_hour": 20},
    "sizing": {"risk_per_trade": 0.005}, "direction": "both", "symbol": "XAUUSD", "origin": "diag:portfolio_check",
    "overlay": {},
}
LIQUID = ["XAUUSD", "EURJPY", "GBPJPY", "USDJPY", "GBPUSD", "EURUSD"]  # pre-specified, NOT chosen by IS result


def row(name: str, r: np.ndarray, n_trials: int) -> dict:
    r = np.asarray(r, float)
    n = len(r)
    if n < 3:
        return {"set": name, "n": n}
    sk, ku = moments(r)
    sr = sharpe(r)
    sr0 = expected_max_sharpe(n_trials, 1.0 / max(n - 1, 1))
    denom = math.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr**2, 1e-12))
    return {"set": name, "n": n, "mean": r.mean(), "std": r.std(ddof=1), "sr": sr, "sr0": sr0,
            "sr_need90": sr0 + 1.2816 * denom / math.sqrt(n - 1), "psr": probabilistic_sharpe(r),
            "dsr": deflated_sharpe(r, n_trials)}


def fmt(d: dict) -> str:
    if "mean" not in d:
        return f"{d['set']:<26} n={d['n']}"
    return (f"{d['set']:<26} {d['n']:>6} {d['mean']:>+8.4f} {d['std']:>7.3f} {d['sr']:>+8.4f} {d['sr0']:>7.4f}"
            f" {d['sr_need90']:>8.4f} {d['psr']:>6.3f} {d['dsr']:>6.3f}")


HDR = f"{'set':<26} {'n':>6} {'meanR':>8} {'stdR':>7} {'SR/trd':>8} {'SR0':>7} {'SRneed':>8} {'PSR':>6} {'DSR':>6}"


def weekly(tr: pd.DataFrame) -> np.ndarray:
    """Sum of R per exit week across symbols, zero-filled weeks: one observation per calendar week."""
    s = tr.set_index(pd.to_datetime(tr.exit_time))["pnl_r"].sort_index()
    return s.resample("W").sum().fillna(0.0).to_numpy()


def pooled(name: str, is_tr: pd.DataFrame, oos_tr: pd.DataFrame, n_trials: int, n_folds: int = 4) -> None:
    is_tr = is_tr.sort_values("exit_time")
    r = is_tr["pnl_r"].to_numpy()
    print(f"\n=== POOLED {name}: {is_tr['sym'].nunique()} symbols ===")
    print(HDR)
    print(fmt(row(f"{name} IS per-trade", r, n_trials)))
    print(fmt(row(f"{name} IS weekly-sum", weekly(is_tr), n_trials)))
    print(fmt(row(f"{name} OOS per-trade", oos_tr["pnl_r"].to_numpy(), n_trials)))
    folds = [float(f.mean()) for f in np.array_split(r, n_folds)] if len(r) >= n_folds else []
    print(f"IS folds (exit-time ordered) mean R: {[round(x, 4) for x in folds]}  signs: "
          f"{''.join('+' if x > 0 else '-' for x in folds)}")
    o = oos_tr["pnl_r"].to_numpy()
    print(f"OOS expectancy: {o.mean() if len(o) else float('nan'):+.4f} R over {len(o)} trades")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--data-dir", default="data/parquet")
    ap.add_argument("--db", default="propfloor.sqlite")
    ap.add_argument("--cache", default=None, help="dir for per-symbol trade csvs (resume/partial runs)")
    ap.add_argument("--genome", default=None, help="genome json (default: gold near-miss)")
    ap.add_argument("--oos-frac", type=float, default=0.25)
    a = ap.parse_args()

    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True, timeout=30)
    n_trials = int(c.execute("SELECT value FROM meta WHERE key='n_trials'").fetchone()[0])
    c.close()
    g = StrategyGenome.from_dict(json.loads(a.genome) if a.genome else GOLD_NEAR_MISS)
    program = load_program(DEFAULT_PROGRAM)
    st = DataStore(a.data_dir)
    syms = a.symbols.split(",") if a.symbols else st.symbols()
    cache = Path(a.cache) if a.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    print(f"n_trials={n_trials} genome={g.key()} tf={g.tf} symbols={syms}", flush=True)

    is_all, oos_all, rows = [], [], []
    for sym in syms:  # one symbol at a time keeps memory low; this rule uses no peers
        if st.is_synthetic(sym):
            print(f"{sym}: synthetic, skipped")
            continue
        f = cache / f"{sym}_{g.key()}.csv" if cache else None
        if f and f.exists():
            tr = pd.read_csv(f, parse_dates=["entry_time", "exit_time"])
            meta = json.loads((f.with_suffix(".json")).read_text())
            cut = pd.Timestamp(meta["oos_start"])
        else:
            u = Universe({sym: st.load(sym)}, timeframes=[g.tf], oos_frac=a.oos_frac)
            ev = Evaluator(u, program)
            tr = ev.trades(g, sym)
            cut = u.oos_start[sym]
            del u, ev
            gc.collect()
            if f:
                tr.to_csv(f, index=False)
                f.with_suffix(".json").write_text(json.dumps({"oos_start": str(cut)}))
        et, xt = pd.to_datetime(tr.entry_time), pd.to_datetime(tr.exit_time)
        if getattr(cut, "tzinfo", None) is not None and et.dt.tz is None:
            cut = cut.tz_localize(None)
        is_tr, oos_tr = tr[xt < cut].assign(sym=sym), tr[et >= cut].assign(sym=sym)
        is_all.append(is_tr)
        oos_all.append(oos_tr)
        d = row(sym, is_tr["pnl_r"].to_numpy(), n_trials)
        o = oos_tr["pnl_r"].to_numpy()
        rows.append((d, len(o), o.mean() if len(o) else float("nan")))
        print(f"{fmt(d)}  | OOS n={len(o)} meanR={rows[-1][2]:+.4f}", flush=True)

    print("\n=== PER SYMBOL (IS) ===")
    print(HDR + "  | OOS n, meanR")
    for d, n_o, m_o in rows:
        print(f"{fmt(d)}  | {n_o:>4} {m_o:+.4f}")
    if not is_all:
        sys.exit("no trades")
    IS, OOS = pd.concat(is_all), pd.concat(oos_all)
    pooled("ALL", IS, OOS, n_trials)
    liq = [s for s in LIQUID if s in set(IS.sym)]
    pooled("LIQUID6", IS[IS.sym.isin(liq)], OOS[OOS.sym.isin(liq)], n_trials)


if __name__ == "__main__":
    main()
