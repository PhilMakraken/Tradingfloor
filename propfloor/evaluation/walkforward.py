"""In-sample / walk-forward / out-of-sample evaluation with rejection gates.

The data of every symbol is split at a fixed timestamp: everything before ``oos_start`` is in-sample
(the only thing the optimiser ever sees); the tail is an untouched holdout used once per candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from propfloor.config import ProgramConfig
from propfloor.data.clean import resample
from propfloor.evaluation.challenge import ChallengeSimulator, summarize
from propfloor.evaluation.stats import deflated_sharpe, trade_summary
from propfloor.strategies.blocks import TIMEFRAMES, Ctx
from propfloor.strategies.genome import StrategyGenome


@dataclass
class Gates:
    min_trades_is: int = 40
    min_expectancy_is: float = 0.05
    n_folds: int = 4
    min_positive_folds: float = 0.75
    min_trades_oos: int = 12
    min_expectancy_oos: float = 0.0
    min_dsr: float = 0.90
    min_p_pass_is: float = 0.25
    min_p_pass_oos: float = 0.20
    max_degradation: float = 0.5  # OOS p_pass must be >= this share of IS p_pass
    min_cross_symbol_positive: float = 0.5  # share of other symbols with positive expectancy


@dataclass
class ValidationResult:
    validated: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class Universe:
    """Contexts for every (symbol, timeframe), sharing the in-sample/OOS split."""

    def __init__(self, data: dict[str, pd.DataFrame], timeframes=TIMEFRAMES, oos_frac: float = 0.25,
                 synthetic: dict[str, bool] | None = None) -> None:
        self.ctx: dict[tuple[str, str], Ctx] = {}
        self.oos_start: dict[str, pd.Timestamp] = {}
        self.synthetic = synthetic or {}
        for sym, m1 in data.items():
            self.oos_start[sym] = m1.index[int(len(m1) * (1 - oos_frac))]
            for tf in timeframes:
                bars = m1 if tf == "M1" else resample(m1, tf)
                self.ctx[(sym, tf)] = Ctx(bars, sym, tf)
        self.symbols = list(data)
        self.timeframes = list(timeframes)


class Evaluator:
    def __init__(self, universe: Universe, program: ProgramConfig, gates: Gates | None = None,
                 quick_sims: int = 30, full_sims: int = 150, seed: int = 0) -> None:
        self.u = universe
        self.program = program
        self.gates = gates or Gates()
        self.quick_sims = quick_sims
        self.full_sims = full_sims
        self.seed = seed
        c = program.costs
        self.costs = dict(spread_pips=c.spread_pips, slippage_pips=c.slippage_pips, commission_per_lot=c.commission_per_lot)

    def trades(self, g: StrategyGenome, symbol: str | None = None) -> pd.DataFrame:
        return g.backtest(self.u.ctx[(symbol or g.symbol, g.tf)], **self.costs)

    def split(self, tr: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        cut = self.u.oos_start[symbol]
        return tr[tr.exit_time < cut], tr[tr.entry_time >= cut]

    def sim(self, g: StrategyGenome) -> ChallengeSimulator:
        return ChallengeSimulator(self.program, risk_per_trade=g.sizing.get("risk_per_trade", 0.005))

    # ---------------------------------------------------------------- quick (GA fitness, IS only)
    def quick(self, g: StrategyGenome) -> tuple[float, dict]:
        tr = self.trades(g)
        is_tr, _ = self.split(tr, g.symbol)
        r = is_tr["pnl_r"].to_numpy()
        ts = trade_summary(r)
        info = {"is": ts}
        if ts["n"] < self.gates.min_trades_is or ts["expectancy_r"] <= 0:
            fit = ts["expectancy_r"] * min(ts["n"], 50) / 50 - 1.0  # negative: guides search towards edge
            return fit, info
        res = self.sim(g).evaluate(is_tr, n_windows=self.quick_sims, n_mc=self.quick_sims, seed=self.seed)
        s = summarize(res)
        info["challenge_is"] = {k: s[k] for k in ("p_pass_all", "p_pass_stage1", "p_funded_survive", "expected_payout")}
        # shrink by sample size so tiny samples can't top the leaderboard
        fit = s["p_pass_all"] * (1 - 1 / np.sqrt(ts["n"])) + 0.1 * s["p_funded_survive"]
        return float(fit), info

    # ---------------------------------------------------------------- full validation
    def validate(self, g: StrategyGenome, n_trials: int) -> ValidationResult:
        G = self.gates
        reasons: list[str] = []
        tr = self.trades(g)
        is_tr, oos_tr = self.split(tr, g.symbol)
        r_is, r_oos = is_tr["pnl_r"].to_numpy(), oos_tr["pnl_r"].to_numpy()
        m: dict = {"is": trade_summary(r_is), "oos": trade_summary(r_oos)}
        if self.u.synthetic.get(g.symbol):
            reasons.append("synthetic data cannot validate")
        if m["is"]["n"] < G.min_trades_is:
            reasons.append(f"IS trades {m['is']['n']} < {G.min_trades_is}")
        if m["is"]["expectancy_r"] < G.min_expectancy_is:
            reasons.append("IS expectancy too low")
        # walk-forward folds over IS period: edge must persist through time
        folds = np.array_split(r_is, G.n_folds) if len(r_is) >= G.n_folds else []
        fold_exp = [float(f.mean()) if len(f) else 0.0 for f in folds]
        m["fold_expectancy"] = fold_exp
        if folds and np.mean([e > 0 for e in fold_exp]) < G.min_positive_folds:
            reasons.append("edge not stable across walk-forward folds")
        # deflated Sharpe vs number of strategies tried so far
        dsr = deflated_sharpe(r_is, n_trials) if len(r_is) > 2 else 0.0
        m["dsr"] = dsr
        m["n_trials"] = n_trials
        if dsr < G.min_dsr:
            reasons.append(f"deflated Sharpe {dsr:.2f} < {G.min_dsr}")
        # OOS holdout
        if m["oos"]["n"] < G.min_trades_oos:
            reasons.append(f"OOS trades {m['oos']['n']} < {G.min_trades_oos}")
        if m["oos"]["expectancy_r"] <= G.min_expectancy_oos:
            reasons.append("OOS expectancy <= 0")
        sim = self.sim(g)
        s_is = summarize(sim.evaluate(is_tr, n_windows=self.full_sims, n_mc=self.full_sims, seed=self.seed + 1))
        s_oos = summarize(sim.evaluate(oos_tr, n_windows=self.full_sims, n_mc=self.full_sims, seed=self.seed + 2))
        m["challenge_is"], m["challenge_oos"] = s_is, s_oos
        if s_is["p_pass_all"] < G.min_p_pass_is:
            reasons.append(f"IS P(pass) {s_is['p_pass_all']:.2f} < {G.min_p_pass_is}")
        if s_oos["p_pass_all"] < G.min_p_pass_oos:
            reasons.append(f"OOS P(pass) {s_oos['p_pass_all']:.2f} < {G.min_p_pass_oos}")
        if s_oos["p_pass_all"] < G.max_degradation * s_is["p_pass_all"]:
            reasons.append("OOS P(pass) degraded vs IS (overfit)")
        # cross-symbol robustness (same rules on other pairs, IS only)
        others = [s for s in self.u.symbols if s != g.symbol]
        if others:
            pos = []
            for s in others:
                o_is, _ = self.split(self.trades(g, s), s)
                pos.append(len(o_is) >= 10 and o_is["pnl_r"].mean() > 0)
            m["cross_symbol_positive"] = float(np.mean(pos))
            if m["cross_symbol_positive"] < G.min_cross_symbol_positive:
                reasons.append("does not generalise to other symbols")
        return ValidationResult(not reasons, reasons, m)
