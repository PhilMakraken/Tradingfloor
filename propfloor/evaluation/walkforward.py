"""In-sample / walk-forward / out-of-sample evaluation with rejection gates.

The data of every symbol is split at a fixed timestamp: everything before ``oos_start`` is in-sample
(the only thing the optimiser ever sees); the tail is an untouched holdout used once per candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from propfloor.config import ProgramConfig
from propfloor.data.clean import TF_RULES, resample
from propfloor.evaluation import robustness as rb
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
    allow_synthetic: bool = False  # DEV ONLY: lets synthetic data validate (pipeline smoke tests)
    # long-history requirement ("proven over N years") - checked on the full IS+OOS trade list
    min_history_years: float = 0.0
    min_positive_years: float = 0.0  # share of calendar years with positive total R
    # Monte Carlo anti-overfitting tests (evaluation/robustness.py), run only once every other gate passed
    robustness: bool = True
    mc_min_p_profit: float = 0.95  # bootstrap P(total R > 0)
    max_random_entry_p: float = 0.05  # permutation-test p-value vs random entries with the same exits
    min_neighbour_positive: float = 0.6  # share of +/-15% parameter neighbours with positive IS expectancy


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
            native = m1.index.to_series().diff().median() if len(m1) > 1 else pd.Timedelta(minutes=1)
            for tf in timeframes:
                if tf_length(tf) < native * 0.9:
                    continue  # e.g. daily-only stock history cannot make M15 bars
                bars = m1 if tf == "M1" else resample(m1, tf)
                self.ctx[(sym, tf)] = Ctx(bars, sym, tf)
                self.ctx[(sym, tf)].train_end = self.oos_start[sym]
        for (sym, tf), c in self.ctx.items():  # intermarket blocks see the other symbols on the same timeframe
            c.peers = {s: self.ctx[(s, tf)] for s in data if s != sym and (s, tf) in self.ctx}
        self.timeframes = list(timeframes)
        self.symbols = [s for s in data if self.tfs_for(s)]  # data coarser than every timeframe is unusable

    def tfs_for(self, sym: str) -> list[str]:
        return [tf for tf in self.timeframes if (sym, tf) in self.ctx]


def tf_length(tf: str) -> pd.Timedelta:
    rule = TF_RULES.get(tf.upper(), tf)
    return pd.Timedelta(days=1) if rule == "D1" else pd.Timedelta(rule)


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
        return ChallengeSimulator(self.program, risk_per_trade=g.sizing.get("risk_per_trade", 0.005), overlay=g.overlay)

    # ---------------------------------------------------------------- quick (GA fitness, IS only)
    def quick(self, g: StrategyGenome) -> tuple[float, dict]:
        tr = self.trades(g)
        is_tr, _ = self.split(tr, g.symbol)
        r = is_tr["pnl_r"].to_numpy()
        ts = trade_summary(r)
        info = {"is": ts}
        if ts["n"] < self.gates.min_trades_is or ts["expectancy_r"] <= 0:
            # always negative: guides search towards edge; expectancy capped so a tiny lucky sample
            # (e.g. 36 trades at +6R) cannot outrank strategies that clear the trade-count gate
            fit = min(ts["expectancy_r"], 1.0) * min(ts["n"], 50) / 50 - 1.0
            return fit, info
        res = self.sim(g).evaluate(is_tr, n_windows=self.quick_sims // 3, n_mc=self.quick_sims, seed=self.seed,
                                   mc_len=max(len(is_tr) * 2, 600))
        s = summarize(res, mc_only=True)
        info["challenge_is"] = {k: s[k] for k in ("p_pass_all", "p_pass_stage1", "p_funded_survive", "expected_payout")}
        # primary: P(pass both stages); secondary terms give the GA a gradient before anything passes.
        # Shrunk by sample size so tiny samples can't top the leaderboard.
        fit = (s["p_pass_all"] + 0.25 * s["p_pass_stage1"] + 0.1 * s["p_funded_survive"]
               + 0.1 * min(ts["expectancy_r"], 1.0)) * (1 - 1 / np.sqrt(ts["n"]))
        return float(fit), info

    # ---------------------------------------------------------------- full validation
    def validate(self, g: StrategyGenome, n_trials: int) -> ValidationResult:
        G = self.gates
        reasons: list[str] = []
        tr = self.trades(g)
        is_tr, oos_tr = self.split(tr, g.symbol)
        r_is, r_oos = is_tr["pnl_r"].to_numpy(), oos_tr["pnl_r"].to_numpy()
        m: dict = {"is": trade_summary(r_is), "oos": trade_summary(r_oos)}
        if self.u.synthetic.get(g.symbol) and not G.allow_synthetic:
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
        s_is = summarize(sim.evaluate(is_tr, n_windows=self.full_sims // 3, n_mc=self.full_sims, seed=self.seed + 1,
                                      mc_len=max(len(is_tr) * 2, 600)), mc_only=True)
        # holdout: Monte Carlo over long resampled sequences of the holdout trades ("if this behaviour continued,
        # would it pass?"). Historical windows inside a ~2-year holdout run out of data for slow strategies.
        oos_res = sim.evaluate(oos_tr, n_windows=self.full_sims // 3, n_mc=self.full_sims, seed=self.seed + 2,
                               mc_len=max(len(oos_tr) * 4, 600))
        s_oos = summarize(oos_res, mc_only=True)
        s_oos["windows_p_pass_all"] = oos_res["windows"].p_pass_all
        m["challenge_is"], m["challenge_oos"] = s_is, s_oos
        if s_is["p_pass_all"] < G.min_p_pass_is:
            reasons.append(f"IS P(pass) {s_is['p_pass_all']:.2f} < {G.min_p_pass_is}")
        if s_oos["p_pass_all"] < G.min_p_pass_oos:
            reasons.append(f"OOS P(pass) {s_oos['p_pass_all']:.2f} < {G.min_p_pass_oos}")
        if s_oos["p_pass_all"] < G.max_degradation * s_is["p_pass_all"]:
            reasons.append("OOS P(pass) degraded vs IS (overfit)")
        # cross-symbol robustness (same rules on other pairs, IS only). Symbols where the block cannot
        # trade (e.g. an intermarket block without its peers) are not applicable rather than failures.
        others = [s for s in self.u.symbols if s != g.symbol]
        if others:
            pos = []
            for s in others:
                if (s, g.tf) not in self.u.ctx:
                    continue
                o_is, _ = self.split(self.trades(g, s), s)
                if len(o_is) >= 10:
                    pos.append(o_is["pnl_r"].mean() > 0)
            m["cross_symbol_applicable"] = len(pos)
            m["cross_symbol_positive"] = float(np.mean(pos)) if pos else None
            if not pos:
                m["specialised"] = True  # only tradable on its own symbol: flagged, OOS gates still apply
            elif m["cross_symbol_positive"] < G.min_cross_symbol_positive:
                reasons.append("does not generalise to other symbols")
        ctx = self.u.ctx[(g.symbol, g.tf)]
        m["yearly"] = rb.yearly(tr, ctx.time[0], ctx.time[-1])
        if m["yearly"]["years"] < G.min_history_years:
            reasons.append(f"history {m['yearly']['years']:.1f}y < {G.min_history_years}y")
        if m["yearly"]["positive_years"] < G.min_positive_years:
            reasons.append(f"profitable in {m['yearly']['positive_years']:.0%} of years < {G.min_positive_years:.0%}")
        if G.robustness and not reasons:
            m["robustness"] = self.robustness(g, tr)
            reasons += self._robustness_reasons(m["robustness"])
        return ValidationResult(not reasons, reasons, m)

    def robustness(self, g: StrategyGenome, tr: pd.DataFrame | None = None, n_random: int = 100) -> dict:
        tr = self.trades(g) if tr is None else tr
        ctx = self.u.ctx[(g.symbol, g.tf)]
        return {"bootstrap": rb.bootstrap(tr["pnl_r"].to_numpy(), seed=self.seed),
                "random_entries": rb.random_entries(g, ctx, self.costs, n=n_random, seed=self.seed),
                "neighbourhood": rb.neighbourhood(g, ctx, self.costs, seed=self.seed)}

    def _robustness_reasons(self, r: dict) -> list[str]:
        G, out = self.gates, []
        if r["bootstrap"]["p_profit"] < G.mc_min_p_profit:
            out.append(f"MC bootstrap P(profit) {r['bootstrap']['p_profit']:.2f} < {G.mc_min_p_profit}")
        if r["random_entries"]["p_value"] > G.max_random_entry_p:
            out.append(f"not better than random entries (p={r['random_entries']['p_value']:.2f})")
        if r["neighbourhood"]["positive_share"] < G.min_neighbour_positive:
            out.append(f"parameter neighbourhood fragile ({r['neighbourhood']['positive_share']:.0%} positive)")
        return out
