"""Virtual trading floor: many paper accounts moving Stage1 -> Stage2 -> Funded (or Violated).

Accounts trade baskets of validated strategies. Strategy signals are produced by the same
backtester on the most recent bars (paper trading); every order goes through the RiskManager
and a BrokerAdapter (PaperBroker by default), and every event is checked by the Account.
"""
from __future__ import annotations

import heapq
import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from propfloor.account import Account
from propfloor.config import ProgramConfig
from propfloor.db import ResultsDB, now
from propfloor.floor.allocator import allocate
from propfloor.floor.broker import BrokerAdapter, Order, PaperBroker
from propfloor.risk import RiskManager
from propfloor.rules.base import Position
from propfloor.strategies.genome import StrategyGenome

log = logging.getLogger(__name__)

LIFECYCLE = ["stage1", "stage2", "funded", "violated"]


@dataclass
class FloorAccount:
    id: str
    program: ProgramConfig
    strategies: list[str]
    stage: str = ""
    status: str = "active"  # active | violated | expired
    payouts: float = 0.0
    history: list[dict] = field(default_factory=list)
    account: Account | None = None
    _hwm: float = 0.0
    _next_payout: pd.Timestamp | None = None

    def __post_init__(self):
        if not self.stage:
            self.stage = self.program.stages[0].name
        self.account = Account(self.program, self.stage)
        self._hwm = self.account.state.start_balance

    @property
    def lifecycle_stage(self) -> str:
        return "violated" if self.status == "violated" else self.stage

    def advance(self, t: pd.Timestamp) -> None:
        names = [s.name for s in self.program.stages] + ["funded"]
        nxt = names[names.index(self.stage) + 1]
        self.history.append({"time": str(t), "event": "passed", "stage": self.stage, "balance": self.account.state.balance})
        self.stage = nxt
        self.account = Account(self.program, nxt)
        self.account.tick(t)
        self._hwm = self.account.state.start_balance
        self._next_payout = t + pd.Timedelta(days=self.program.funded.payout_period_days) if nxt == "funded" else None

    def maybe_payout(self, t: pd.Timestamp) -> float:
        if self.stage != "funded" or self._next_payout is None or t < self._next_payout or self.account.status != "active":
            return 0.0
        self._next_payout += pd.Timedelta(days=self.program.funded.payout_period_days)
        s = self.account.state
        gain = s.balance - self._hwm
        if gain <= 0 or self.account.completion_blocks():
            return 0.0
        amt = gain * self.program.funded.profit_split
        self.payouts += amt
        self._hwm = s.balance
        s.period_start_balance = s.balance
        s.daily_pnl.clear()
        self.history.append({"time": str(t), "event": "payout", "amount": amt})
        return amt


class TradingFloor:
    def __init__(self, program: ProgramConfig, db: ResultsDB, broker: BrokerAdapter | None = None,
                 risk: RiskManager | None = None) -> None:
        self.program = program
        self.db = db
        self.broker = broker or PaperBroker()
        self.risk = risk or RiskManager(costs=program.costs)
        self.accounts: dict[str, FloorAccount] = {}

    # ---------------------------------------------------------------- setup
    def open_accounts(self, n: int, per_account: int = 2, max_corr: float = 0.3) -> list[FloorAccount]:
        cands = self.db.validated()
        rets = {c["id"]: self.db.returns(c["id"]) for c in cands}
        baskets = allocate(cands, rets, n, per_account, max_corr)
        start = len(self.accounts)
        out = []
        for i, b in enumerate(baskets):
            fa = FloorAccount(f"acct-{start + i + 1:03d}", self.program, b)
            self.accounts[fa.id] = fa
            out.append(fa)
            self.db.log_event(fa.id, "opened", {"strategies": b})
        return out

    def genomes(self, ids: list[str]) -> dict[str, StrategyGenome]:
        out = {}
        for sid in ids:
            r = self.db.query("SELECT genome FROM strategies WHERE id=?", (sid,))
            if r:
                out[sid] = StrategyGenome.from_dict(json.loads(r[0]["genome"]))
        return out

    # ---------------------------------------------------------------- simulation
    def run(self, trades_by_strategy: dict[str, pd.DataFrame], start=None, end=None) -> None:
        """Replay strategy trades (from live/paper signals) through every account, chronologically."""
        for fa in self.accounts.values():
            if fa.status != "active":
                continue
            evs = []
            for sid in fa.strategies:
                tr = trades_by_strategy.get(sid)
                if tr is None or tr.empty:
                    continue
                if start is not None:
                    tr = tr[tr.entry_time >= pd.Timestamp(start, tz="UTC")]
                if end is not None:
                    tr = tr[tr.exit_time < pd.Timestamp(end, tz="UTC")]
                for k, row in enumerate(tr.itertuples(index=False)):
                    pid = f"{sid}:{k}"
                    evs.append((row.entry_time, 0, pid, "open", row))
                    a, b = (row.mfe_time, row.mae_time) if row.mfe_time <= row.mae_time else (row.mae_time, row.mfe_time)
                    evs.append((max(a, row.entry_time), 1, pid, "mark_a", row))
                    evs.append((max(b, row.entry_time), 2, pid, "mark_b", row))
                    evs.append((row.exit_time, 3, pid, "close", row))
            heapq.heapify(evs)
            self._replay(fa, sorted(evs, key=lambda e: (e[0], e[1], e[2])))
            self._persist(fa)

    def _replay(self, fa: FloorAccount, evs: list) -> None:
        live: dict[str, tuple[float, str]] = {}  # pid -> (money per R, broker id)
        for t, _, pid, kind, row in evs:
            if fa.status != "active":
                break
            fa.maybe_payout(t)
            acc = fa.account
            if kind == "open":
                acc.tick(t)
                if acc.status != "active":
                    self._on_end(fa, t)
                    continue
                dec = self.risk.size(acc, float(row.stop_pips))
                if not dec.ok:
                    continue
                pos = Position(pid, row.symbol, int(row.side), dec.lots, t, entry_price=float(row.entry))
                if self.risk.approve(acc, pos, float(row.stop_pips), t):
                    continue
                fill = self.broker.place(Order(fa.id, row.symbol, int(row.side), dec.lots), float(row.entry), t)
                live[pid] = (dec.lots * self.risk.pip_value * float(row.stop_pips), fill.broker_id)
                acc.open(pos)
            elif pid in live:
                mpr, bid = live[pid]
                if kind == "close":
                    live.pop(pid)
                    self.broker.close(bid, float(row.exit), t)
                    acc.close(pid, float(row.pnl_r) * mpr, t)
                else:
                    first_is_mfe = row.mfe_time <= row.mae_time
                    r = (row.mfe_r if first_is_mfe else row.mae_r) if kind == "mark_a" else (row.mae_r if first_is_mfe else row.mfe_r)
                    acc.mark(t, {pid: float(r) * mpr})
            if acc.status == "passed" and fa.stage != "funded":
                fa.advance(t)
                live.clear()
                self.db.log_event(fa.id, "stage", {"stage": fa.stage}, str(t))
            elif acc.status in ("violated", "expired"):
                self._on_end(fa, t)

    def _on_end(self, fa: FloorAccount, t) -> None:
        acc = fa.account
        fa.status = acc.status
        detail = {"stage": fa.stage, "status": acc.status,
                  "violation": acc.violation.__dict__ if acc.violation else None}
        fa.history.append({"time": str(t), "event": acc.status, **detail})
        self.db.log_event(fa.id, acc.status, detail, str(t))

    def _persist(self, fa: FloorAccount) -> None:
        a = fa.account
        s = a.state
        self.db.execute(
            "INSERT OR REPLACE INTO floor_accounts(id, program, stage, status, strategies, start_balance, balance, equity,"
            " floor, daily_floor, payouts, violation, created_at, updated_at, state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,"
            " COALESCE((SELECT created_at FROM floor_accounts WHERE id=?), ?), ?, ?)",
            (fa.id, self.program.name, fa.lifecycle_stage, fa.status, json.dumps(fa.strategies), s.start_balance,
             s.balance, s.equity, a.floor, a.daily_floor, fa.payouts,
             json.dumps(a.violation.__dict__, default=str) if a.violation else None, fa.id, now(), now(),
             json.dumps({"history": fa.history[-50:], "trading_days": len(s.trading_days), "n_trades": s.n_trades}, default=str)))

    def summary(self) -> pd.DataFrame:
        rows = []
        for fa in self.accounts.values():
            a = fa.account
            b = a.buffers()
            rows.append({"id": fa.id, "stage": fa.lifecycle_stage, "balance": a.state.balance, "equity": a.state.equity,
                         "trailing_buffer": b["trailing"], "daily_buffer": b["daily"], "payouts": fa.payouts,
                         "strategies": len(fa.strategies)})
        return pd.DataFrame(rows)
