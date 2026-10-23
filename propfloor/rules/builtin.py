"""Built-in rules. Each is independent; add new ones anywhere with @register_rule."""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from propfloor.rules.base import Event, Position, Rule, register_rule

log = logging.getLogger(__name__)


@register_rule("max_lot")
class MaxLot(Rule):
    def _bad(self, pos: Position) -> bool:
        return pos.lots > float(self.params.get("max_lots", 50))

    def pre_trade(self, state, position, time):
        return f"lots {position.lots:.2f} > max" if self._bad(position) else None

    def check(self, state, event):
        if event.kind == "open" and event.position and self._bad(event.position):
            return self.violation(f"lot size {event.position.lots:.2f} exceeds max", event.time)


@register_rule("max_open_trades")
class MaxOpenTrades(Rule):
    def pre_trade(self, state, position, time):
        if len(state.open_positions) + 1 > int(self.params.get("max", 10)):
            return "too many open trades"

    def check(self, state, event):
        if event.kind == "open" and len(state.open_positions) > int(self.params.get("max", 10)):
            return self.violation("max open trades exceeded", event.time)


@register_rule("news_blackout")
class NewsBlackout(Rule):
    """No opening/closing within [t-before, t+after] of a high-impact news time.

    Calendar CSV needs a ``time`` column (UTC) and optionally ``currency``. ``times`` may also be
    passed directly as a list of timestamps (handy in tests).
    """

    def __init__(self, **params):
        super().__init__(**params)
        times = params.get("times")
        if times is None:
            path = Path(params.get("calendar", "data/calendar/high_impact.csv"))
            if path.exists():
                cal = pd.read_csv(path)
                times = pd.to_datetime(cal["time"], utc=True)
            else:
                log.debug("news calendar %s not found; news_blackout inactive", path)
                times = []
        self.times = (
            np.sort(pd.to_datetime(pd.Series(list(times)), utc=True).dt.as_unit("ns").astype("int64").to_numpy())
            if len(times) else np.array([], dtype=np.int64)
        )
        self.before = int(params.get("minutes_before", 2)) * 60_000_000_000
        self.after = int(params.get("minutes_after", 2)) * 60_000_000_000

    def in_blackout(self, t: pd.Timestamp) -> bool:
        if not len(self.times):
            return False
        v = pd.Timestamp(t).as_unit("ns").value
        i = np.searchsorted(self.times, v - self.after)
        return i < len(self.times) and self.times[i] <= v + self.before

    def pre_trade(self, state, position, time):
        return "news blackout" if self.in_blackout(time) else None

    def check(self, state, event):
        if event.kind in ("open", "close") and self.in_blackout(event.time):
            return self.violation(f"{event.kind} during news blackout", event.time)


def _next_friday_cutoff(t: pd.Timestamp, tz: str, cutoff: str) -> pd.Timestamp:
    local = t.tz_convert(tz)
    h, m = map(int, cutoff.split(":"))
    days = (4 - local.weekday()) % 7
    cand = (local + timedelta(days=days)).replace(hour=h, minute=m, second=0, microsecond=0, nanosecond=0)
    if cand < local:
        # already past this week's cutoff (Fri evening / weekend) -> already "weekend"
        if local.weekday() >= 4:
            return local.tz_convert("UTC")
        cand += timedelta(days=7)
    return cand.tz_convert("UTC")


@register_rule("no_weekend_holding")
class NoWeekendHolding(Rule):
    def _tz(self):
        return self.params.get("timezone", "America/New_York"), self.params.get("friday_cutoff", "16:50")

    def pre_trade(self, state, position, time):
        tz, co = self._tz()
        if _next_friday_cutoff(pd.Timestamp(time), tz, co) <= pd.Timestamp(time):
            return "weekend: cannot open"

    def check(self, state, event):
        tz, co = self._tz()
        held = list(state.open_positions.values())
        if event.kind == "close" and event.position is not None:
            held.append(event.position)
        for p in held:
            if pd.Timestamp(event.time) > _next_friday_cutoff(p.entry_time, tz, co):
                return self.violation(f"position {p.id} held over weekend", event.time)


@register_rule("max_time_in_trade")
class MaxTimeInTrade(Rule):
    def check(self, state, event):
        limit = pd.Timedelta(hours=float(self.params.get("max_hours", 120)))
        held = list(state.open_positions.values())
        if event.kind == "close" and event.position is not None:
            held.append(event.position)
        for p in held:
            if pd.Timestamp(event.time) - p.entry_time > limit:
                return self.violation(f"position {p.id} exceeded max time in trade", event.time)


@register_rule("min_time_in_trade")
class MinTimeInTrade(Rule):
    """Anti-scalping rule: positions must be held at least N seconds (violation)."""

    def check(self, state, event):
        if event.kind == "close" and event.position is not None:
            if pd.Timestamp(event.time) - event.position.entry_time < pd.Timedelta(seconds=float(self.params.get("min_seconds", 60))):
                return self.violation("trade closed before minimum hold time", event.time)


@register_rule("min_trading_days")
class MinTradingDays(Rule):
    def completion_block(self, state):
        need = int(self.params.get("days", 4))
        if len(state.trading_days) < need:
            return f"only {len(state.trading_days)}/{need} trading days"


@register_rule("consistency_rule")
class ConsistencyRule(Rule):
    """Best single day's profit must not exceed ``max_day_share`` of total profit."""

    def completion_block(self, state):
        total = state.balance - state.period_start_balance
        if total <= 0:
            return None
        best = max([v for v in state.daily_pnl.values()] + [0.0])
        share = float(self.params.get("max_day_share", 0.5))
        if best > share * total:
            return f"consistency: best day {best:.0f} > {share:.0%} of {total:.0f}"
