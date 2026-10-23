"""Simulated prop account: balance/equity, trailing max-DD floor, daily DD and rule checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Literal

import pandas as pd

from propfloor.config import ProgramConfig
from propfloor.rules.base import Event, Position, Rule, Violation, build_rules

@lru_cache(maxsize=500_000)
def _day_key(minute: int, tz: str, reset_ns: int) -> date:
    return (pd.Timestamp(minute * 60_000_000_000, tz="UTC").tz_convert(tz) - pd.Timedelta(reset_ns)).date()


Status = Literal["active", "passed", "violated", "expired"]


@dataclass
class AccountState:
    start_balance: float
    balance: float
    equity: float
    peak: float
    period_start_balance: float
    day_key: date | None = None
    day_start_balance: float = 0.0
    day_start_equity: float = 0.0
    open_positions: dict[str, Position] = field(default_factory=dict)
    floating: dict[str, float] = field(default_factory=dict)
    trading_days: set[date] = field(default_factory=set)
    daily_pnl: dict[date, float] = field(default_factory=dict)
    n_trades: int = 0
    start_time: pd.Timestamp | None = None
    last_time: pd.Timestamp | None = None
    min_equity_buffer: float = float("inf")  # closest approach to the trailing floor


class Account:
    """One account in one stage. Feed it events; it closes permanently on any breach."""

    def __init__(
        self,
        program: ProgramConfig,
        stage: str = "stage1",
        start_balance: float | None = None,
        rules: list[Rule] | None = None,
    ) -> None:
        self.program = program
        self.stage = stage
        sb = float(start_balance if start_balance is not None else program.starting_balance)
        self.state = AccountState(sb, sb, sb, sb, sb, day_start_balance=sb, day_start_equity=sb)
        self.rules = rules if rules is not None else build_rules(program.rules_for(stage), program.plugins)
        self.status: Status = "active"
        self.violation: Violation | None = None
        self.passed_time: pd.Timestamp | None = None
        dd = program.daily_drawdown
        self._tz = dd.timezone
        self._reset = pd.Timedelta(minutes=dd.reset_minutes)
        st = program.stage(stage)
        self.target = st.profit_target
        self.max_days = st.max_calendar_days

    # ------------------------------------------------------------------ levels
    @property
    def floor(self) -> float:
        dd = self.program.drawdown
        s = self.state
        amt = dd.max_dd * s.start_balance
        if not dd.trailing:
            return s.start_balance - amt
        f = s.peak - amt
        if dd.lock_at_start:
            f = min(f, s.start_balance)
        return f

    @property
    def daily_floor(self) -> float:
        dd = self.program.daily_drawdown
        s = self.state
        basis = {
            "balance": s.day_start_balance,
            "equity": s.day_start_equity,
            "max_balance_equity": max(s.day_start_balance, s.day_start_equity),
        }[dd.basis]
        ref = s.start_balance if dd.limit_of == "starting_balance" else basis
        return basis - dd.limit * ref

    def buffers(self) -> dict[str, float]:
        """Money that can still be lost before each limit is touched."""
        e = self.state.equity
        return {"trailing": e - self.floor, "daily": e - self.daily_floor}

    @property
    def remaining_buffer(self) -> float:
        b = self.buffers()
        return min(b["trailing"], b["daily"])

    @property
    def target_balance(self) -> float | None:
        return None if self.target is None else self.state.start_balance * (1 + self.target)

    def day_key(self, t: pd.Timestamp) -> date:
        return _day_key(t.as_unit("ns").value // 60_000_000_000, self._tz, self._reset.value)

    # ------------------------------------------------------------------ events
    def _roll_day(self, t: pd.Timestamp) -> None:
        k = self.day_key(t)
        s = self.state
        if s.day_key != k:
            s.day_key = k
            s.day_start_balance = s.balance
            s.day_start_equity = s.equity

    def process(self, ev: Event) -> Violation | None:
        if self.status != "active":
            return self.violation
        t = pd.Timestamp(ev.time)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        ev.time = t
        s = self.state
        if s.start_time is None:
            s.start_time = t
        if self.max_days is not None and (t - s.start_time).days >= self.max_days:
            self.status = "expired"
            return None
        self._roll_day(t)
        s.last_time = t

        if ev.kind == "open":
            p = ev.position
            assert p is not None
            s.open_positions[p.id] = p
            s.floating[p.id] = 0.0
            s.trading_days.add(s.day_key)
        elif ev.kind == "close":
            p = ev.position
            assert p is not None
            s.open_positions.pop(p.id, None)
            s.floating.pop(p.id, None)
            s.balance += ev.pnl
            s.n_trades += 1
            s.daily_pnl[s.day_key] = s.daily_pnl.get(s.day_key, 0.0) + ev.pnl
            s.trading_days.add(s.day_key)
        s.equity = s.balance + sum(s.floating.values())

        v = self._check_limits(t)
        if v is None:
            for r in self.rules:
                v = r.check(s, ev)
                if v is not None:
                    break
        if v is not None:
            self.status = "violated"
            self.violation = v
            return v
        if ev.kind == "close" and not s.open_positions:
            self._check_pass(t)
        return None

    def _check_limits(self, t: pd.Timestamp) -> Violation | None:
        s = self.state
        dd = self.program.drawdown
        if dd.trail_basis == "equity":
            s.peak = max(s.peak, s.equity)
        else:
            s.peak = max(s.peak, s.balance)
        s.min_equity_buffer = min(s.min_equity_buffer, s.equity - self.floor)
        if s.equity <= self.floor:
            return Violation("max_drawdown", f"equity {s.equity:.2f} <= floor {self.floor:.2f}", t)
        if s.equity <= self.daily_floor:
            return Violation("daily_drawdown", f"equity {s.equity:.2f} <= daily floor {self.daily_floor:.2f}", t)
        return None

    def _check_pass(self, t: pd.Timestamp) -> None:
        tb = self.target_balance
        if tb is None or self.state.balance < tb:
            return
        if self.completion_blocks():
            return
        self.status = "passed"
        self.passed_time = t

    def completion_blocks(self) -> list[str]:
        return [m for r in self.rules if (m := r.completion_block(self.state))]

    # convenience API ------------------------------------------------------
    def open(self, pos: Position, time=None) -> Violation | None:
        return self.process(Event("open", time if time is not None else pos.entry_time, pos))

    def mark(self, time, floating: dict[str, float]) -> Violation | None:
        self.state.floating.update({k: v for k, v in floating.items() if k in self.state.open_positions})
        return self.process(Event("mark", time))

    def close(self, pos_id: str, pnl: float, time) -> Violation | None:
        pos = self.state.open_positions.get(pos_id)
        return self.process(Event("close", time, pos, pnl))

    def tick(self, time) -> Violation | None:
        """Time passes with no price change (lets day rollover / time rules fire)."""
        return self.process(Event("time", time))

    def check_pre_trade(self, pos: Position, time) -> list[str]:
        t = pd.Timestamp(time)
        return [m for r in self.rules if (m := r.pre_trade(self.state, pos, t))]
