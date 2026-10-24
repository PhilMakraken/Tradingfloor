"""Simulated prop challenges from a strategy's trade list (R units).

Two complementary samplers:
* historical windows: start the challenge at many random dates and replay the real trade sequence;
* Monte Carlo: block-bootstrap the trades into new sequences (re-timed by sampled inter-trade gaps).

Each simulated trade is pushed through the full Account (trailing + daily DD + rules) with sizing
from the RiskManager, including intratrade floating extremes (MAE/MFE in the order they occurred).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from propfloor.account import Account
from propfloor.config import ProgramConfig
from propfloor.risk import RiskManager, RiskProfile
from propfloor.rules.base import Position, build_rules

GATE_RULES = {"min_trading_days", "consistency_rule"}


@dataclass
class StageResult:
    status: str  # passed | violated | expired | exhausted
    reason: str | None
    days: float
    n_trades: int
    end_idx: int  # index in the trade arrays after which the stage ended
    end_time: pd.Timestamp | None
    payouts: float = 0.0
    min_buffer: float = 0.0


@dataclass
class ChallengeStats:
    n: int = 0
    p_stage: dict[str, float] = field(default_factory=dict)  # unconditional P(pass up to & incl. stage)
    p_pass_all: float = 0.0
    p_funded_survive: float = 0.0
    median_days: dict[str, float] = field(default_factory=dict)
    breach_causes: dict[str, int] = field(default_factory=dict)
    expected_payout: float = 0.0
    p_payout: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class TradeArrays:
    """Column arrays of a trade DataFrame for fast replay."""

    def __init__(self, tr: pd.DataFrame) -> None:
        self.n = len(tr)
        ns = lambda c: tr[c].dt.as_unit("ns").astype("int64").to_numpy() if self.n else np.zeros(0, np.int64)  # noqa: E731
        self.entry = ns("entry_time")
        self.exit = ns("exit_time")
        self.mae_t = ns("mae_time")
        self.mfe_t = ns("mfe_time")
        self.pnl = tr["pnl_r"].to_numpy(float) if self.n else np.zeros(0)
        self.mae = tr["mae_r"].to_numpy(float) if self.n else np.zeros(0)
        self.mfe = tr["mfe_r"].to_numpy(float) if self.n else np.zeros(0)
        self.post = tr["post_mfe_r"].to_numpy(float) if self.n else np.zeros(0)
        self.stop_pips = tr["stop_pips"].to_numpy(float) if self.n else np.zeros(0)
        self.side = tr["side"].to_numpy(int) if self.n else np.zeros(0, int)
        self.symbol = tr["symbol"].iloc[0] if self.n else "EURUSD"

    def equity_curve_mask(self, n: int) -> np.ndarray:
        """P3: trade i is taken only if the strategy's cumulative R before it is >= its n-trade moving average."""
        cache = self.__dict__.setdefault("_ec", {})
        if n not in cache:
            eq = np.concatenate([[0.0], np.cumsum(self.pnl)])[:-1]
            ma = pd.Series(eq).rolling(n, min_periods=n).mean().to_numpy()
            cache[n] = np.where(np.isnan(ma), True, eq >= ma)
        return cache[n]

    def resampled(self, rng: np.random.Generator, n_out: int, block: int = 5) -> "TradeArrays":
        """Block bootstrap of trades, re-timed from a start using sampled gaps between entries."""
        if self.n < 2:
            return self
        idx = []
        while len(idx) < n_out:
            s = int(rng.integers(0, self.n))
            idx.extend(range(s, min(s + block, self.n)))
        idx = np.array(idx[:n_out])
        gaps = np.diff(self.entry)
        gaps = gaps[gaps > 0]
        hold = np.maximum(self.exit - self.entry, 1)
        new = TradeArrays.__new__(TradeArrays)
        new.n = n_out
        new.symbol = self.symbol
        g = rng.choice(gaps, size=n_out) if len(gaps) else np.full(n_out, 86_400 * 10**9)
        g = np.maximum(g, hold[idx] + 1)  # never overlap: next entry after previous exit
        start = self.entry[0]
        entry = start + np.concatenate([[0], np.cumsum(g[:-1])])
        # sparse strategies x long MC sequences can run past pandas' Timestamp range (year 2262):
        # truncate there (conservative - the run just ends "exhausted")
        keep = int(np.searchsorted(entry + hold[idx], _TS_LIMIT))
        if keep < n_out:
            n_out, idx, entry = max(keep, 1), idx[:max(keep, 1)], entry[:max(keep, 1)]
            new.n = n_out
        new.entry = entry
        new.exit = entry + hold[idx]
        new.mae_t = entry + (self.mae_t[idx] - self.entry[idx])
        new.mfe_t = entry + (self.mfe_t[idx] - self.entry[idx])
        for a in ("pnl", "mae", "mfe", "post", "stop_pips", "side"):
            setattr(new, a, getattr(self, a)[idx])
        return new


_TS_LIMIT = pd.Timestamp("2200-01-01", tz="UTC").value


def _ts(v: int) -> pd.Timestamp:
    return pd.Timestamp(v, tz="UTC")


class ChallengeSimulator:
    def __init__(self, program: ProgramConfig, risk: RiskManager | None = None, risk_per_trade: float | None = None,
                 time_rules: bool = True, funded_days: int = 120, overlay: dict | None = None) -> None:
        self.program = program
        self.risk = risk or RiskManager(costs=program.costs)
        overlay = dict(overlay or {})
        self.equity_curve_n = int(overlay.pop("equity_curve_n", 0) or 0)  # P3, applied per trade below
        fields = set(RiskProfile.model_fields)
        ov = {k: v for k, v in overlay.items() if k in fields}
        if risk_per_trade is not None or ov:
            upd = dict(ov)
            if risk_per_trade is not None:
                upd["risk_per_trade"] = risk_per_trade
            prof = self.risk.profile.model_copy(update=upd)
            # funded: same tactics, half the risk, never looser than the conservative funded profile
            fupd = {**ov, "risk_per_trade": upd.get("risk_per_trade", prof.risk_per_trade) * 0.5, "fast_pass_mult": 1.0,
                    "soft_daily_stop": min(ov.get("soft_daily_stop", 1.0), self.risk.funded_profile.soft_daily_stop),
                    "max_buffer_use": self.risk.funded_profile.max_buffer_use}
            fprof = self.risk.funded_profile.model_copy(update=fupd)
            self.risk = RiskManager(prof, fprof, self.risk.costs, self.risk.pip_value)
        self.funded_days = funded_days
        self._rules = {}
        for st in program.stage_names():
            specs = program.rules_for(st)
            if not time_rules:  # re-timed MC sequences: calendar rules are meaningless, keep gates
                specs = [s for s in specs if s.name in GATE_RULES or s.name in ("max_lot", "max_open_trades")]
            self._rules[st] = build_rules(specs, program.plugins)

    def run_stage(self, ta: TradeArrays, start: int, stage: str, start_time: int | None = None) -> StageResult:
        acc = Account(self.program, stage, rules=self._rules[stage])
        t0 = ta.entry[start] if start < ta.n else 0
        if start_time is not None:
            t0 = start_time
        acc.tick(_ts(t0))
        funded = stage == "funded"
        period_ns = self.program.funded.payout_period_days * 86_400 * 10**9
        next_payout = t0 + period_ns
        hwm = acc.state.start_balance
        payouts = 0.0
        horizon = t0 + self.funded_days * 86_400 * 10**9 if funded else None
        i = start
        while i < ta.n:
            if horizon is not None and ta.entry[i] > horizon:
                break
            if funded and ta.entry[i] >= next_payout:
                payouts, hwm = self._payout(acc, payouts, hwm)
                next_payout += period_ns
            et = _ts(ta.entry[i])
            acc.tick(et)
            if acc.status != "active":
                break
            if self.equity_curve_n and not ta.equity_curve_mask(self.equity_curve_n)[i]:
                i += 1  # P3: strategy's own equity curve below its MA -> sit this trade out
                continue
            dec = self.risk.size(acc, float(ta.stop_pips[i]))
            if dec.ok:
                money_per_r = dec.lots * self.risk.pip_value * float(ta.stop_pips[i])
                pos = Position(f"t{i}", ta.symbol, int(ta.side[i]), dec.lots, et, risk_amount=dec.risk_amount)
                if self.risk.approve(acc, pos, float(ta.stop_pips[i]), et) is None:
                    self._replay(acc, pos, ta, i, money_per_r)
            i += 1
            if acc.status != "active":
                break
        if funded and acc.status == "active":
            payouts, hwm = self._payout(acc, payouts, hwm)
        s = acc.state
        end_t = s.last_time
        days = (end_t.value - int(t0)) / 86_400e9 if end_t is not None else 0.0
        status = acc.status if acc.status != "active" else ("survived" if funded else "exhausted")
        reason = acc.violation.rule if acc.violation else None
        return StageResult(status, reason, days, s.n_trades, i, end_t, payouts, s.min_equity_buffer)

    def _payout(self, acc: Account, payouts: float, hwm: float):
        if acc.status != "active":
            return payouts, hwm
        gain = acc.state.balance - hwm
        if gain > 0 and not acc.completion_blocks():
            payouts += gain * self.program.funded.profit_split
            hwm = acc.state.balance
            acc.state.period_start_balance = acc.state.balance
            acc.state.daily_pnl.clear()
        return payouts, hwm

    @staticmethod
    def _replay(acc: Account, pos: Position, ta: TradeArrays, i: int, money_per_r: float) -> None:
        if acc.open(pos) is not None:
            return
        marks = [(ta.mae_t[i], 1, ta.mae[i]), (ta.mfe_t[i], 0, ta.mfe[i])]  # same bar: MFE first (harsher)
        marks.sort()
        if ta.mae_t[i] < ta.mfe_t[i] and ta.post[i] < 0:
            mid = ta.mfe_t[i] + max((ta.exit[i] - ta.mfe_t[i]) // 2, 1)
            marks.append((min(mid, ta.exit[i] - 1), 2, ta.post[i]))
        for t, _, r in marks:
            if acc.mark(_ts(t), {pos.id: r * money_per_r}) is not None:
                return
        acc.close(pos.id, ta.pnl[i] * money_per_r, _ts(ta.exit[i]))

    # ------------------------------------------------------------------ campaigns
    def run_campaign(self, ta: TradeArrays, start: int, start_time: int | None = None) -> list[StageResult]:
        out = []
        idx, t = start, start_time
        for st in [s.name for s in self.program.stages] + ["funded"]:
            r = self.run_stage(ta, idx, st, t)
            out.append(r)
            if r.status != "passed":
                break
            idx = r.end_idx
            t = None
        return out

    def evaluate(self, tr: pd.DataFrame, n_windows: int = 100, n_mc: int = 100, seed: int = 0,
                 mc_len: int | None = None) -> dict[str, ChallengeStats]:
        rng = np.random.default_rng(seed)
        ta = TradeArrays(tr.sort_values("entry_time").reset_index(drop=True))
        res = {}
        if ta.n < 5:
            return {"windows": ChallengeStats(), "mc": ChallengeStats()}
        # historical random start dates (leave room: last start at ~70% of the timeline)
        span_end = ta.entry[0] + int(0.7 * (ta.entry[-1] - ta.entry[0]))
        starts = rng.integers(ta.entry[0], max(span_end, ta.entry[0] + 1), size=n_windows)
        camps = []
        for s in starts:
            k = int(np.searchsorted(ta.entry, s))
            camps.append(self.run_campaign(ta, k, int(s)))
        res["windows"] = self._aggregate(camps)
        # Monte Carlo resequencing (calendar-free rules)
        mc = ChallengeSimulator.__new__(ChallengeSimulator)
        mc.__dict__.update(self.__dict__)
        mc._rules = {k: [r for r in v if r.name in GATE_RULES or r.name in ("max_lot", "max_open_trades")]
                     for k, v in self._rules.items()}
        L = mc_len or max(ta.n, 300)
        camps = [mc.run_campaign(ta.resampled(rng, L), 0) for _ in range(n_mc)]
        res["mc"] = self._aggregate(camps)
        return res

    def _aggregate(self, camps: list[list[StageResult]]) -> ChallengeStats:
        names = [s.name for s in self.program.stages]
        st = ChallengeStats(n=len(camps))
        causes: Counter = Counter()
        for k, nm in enumerate(names):
            passed = [c[k] for c in camps if len(c) > k and c[k].status == "passed"]
            st.p_stage[nm] = len(passed) / max(len(camps), 1)
            st.median_days[nm] = float(np.median([r.days for r in passed])) if passed else float("nan")
        for c in camps:
            last = c[-1]
            if last.status in ("violated", "expired", "exhausted"):
                causes[f"{last.reason or last.status}@{(names + ['funded'])[len(c) - 1]}"] += 1
        funded = [c[-1] for c in camps if len(c) == len(names) + 1]
        st.p_pass_all = len(funded) / max(len(camps), 1)
        st.p_funded_survive = (sum(r.status == "survived" for r in funded) / len(funded)) if funded else 0.0
        st.expected_payout = float(np.mean([c[-1].payouts if len(c) == len(names) + 1 else 0.0 for c in camps])) if camps else 0.0
        st.p_payout = sum(r.payouts > 0 for r in funded) / max(len(camps), 1)
        st.breach_causes = dict(causes)
        return st


def summarize(res: dict[str, ChallengeStats], mc_only: bool = False) -> dict:
    """Conservative headline: the worse of the historical-window and Monte Carlo estimates.
    mc_only=True uses only the Monte Carlo estimate (for short holdouts, where historical windows run out of
    data before a slow strategy can finish both stages - "exhausted" is not a breach)."""
    w, m = res["windows"], res["mc"]
    if mc_only:
        w = m
    return {
        "p_pass_all": min(w.p_pass_all, m.p_pass_all),
        "p_pass_stage1": min(next(iter(w.p_stage.values()), 0), next(iter(m.p_stage.values()), 0)),
        "p_funded_survive": min(w.p_funded_survive, m.p_funded_survive),
        "expected_payout": min(w.expected_payout, m.expected_payout),
        "median_days": w.median_days,
        "breach_causes": {**{f"w:{k}": v for k, v in w.breach_causes.items()}, **{f"mc:{k}": v for k, v in m.breach_causes.items()}},
        "windows": w.to_dict(),
        "mc": m.to_dict(),
    }
