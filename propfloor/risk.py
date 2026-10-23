"""Pre-trade risk manager.

Every order's *worst-case* loss (stop + spread + slippage + gap allowance + commission) must fit
inside the remaining daily AND trailing buffers (with a safety margin), else it is blocked.
Position size is derived from the remaining buffer, with a soft daily stop, de-risking near the
profit target and a conservative profile once funded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel

from propfloor.account import Account
from propfloor.config import CostConfig
from propfloor.rules.base import Position


class RiskProfile(BaseModel):
    risk_per_trade: float = 0.005  # fraction of starting balance risked to the stop
    max_buffer_use: float = 0.33  # a single trade's worst case may use at most this share of the min buffer
    safety_margin: float = 0.9  # never plan to use more than 90% of any buffer
    soft_daily_stop: float = 0.6  # stop for the day after losing this share of the daily limit
    derisk_near_target: bool = True
    target_reward_r: float = 2.0  # expected R of a winner, used for de-risking near target
    min_risk_per_trade: float = 0.001
    min_lot: float = 0.01
    lot_step: float = 0.01
    # ---- account-level overlays (catalogue J, P4, Q). 0 / False / 1.0 = off
    daily_profit_cap: float = 0.0  # Q4/J: stop for the day after +x of starting balance
    max_trades_per_day: int = 0  # Q5
    loss_streak_n: int = 0  # J: after n consecutive losses ...
    loss_streak_mult: float = 0.0  # ... pause for the day (0) or scale risk by this factor
    fast_pass_mult: float = 1.0  # Q1: risk multiplier in stage 1 ...
    fast_pass_until: float = 0.5  # ... until this share of the profit target is banked
    buffer_risk_frac: float = 0.0  # Q2: risk = frac x distance to the trailing floor (cap at risk_per_trade)
    derisk_within: float = 0.0  # J: halve risk within this distance (fraction of start) of the target
    target_banking: bool = True  # Q3: at/after target, only minimum-size trades (min trading days)
    anti_martingale_step: float = 0.0  # P4: +step risk per consecutive win, reset after a loss


CONSERVATIVE_FUNDED = RiskProfile(risk_per_trade=0.0025, max_buffer_use=0.2, soft_daily_stop=0.4)


@dataclass
class SizeDecision:
    lots: float
    risk_amount: float  # loss at the stop, excluding costs
    worst_case: float  # stop + spread + slippage + gap + commission
    blocked: str | None = None

    @property
    def ok(self) -> bool:
        return self.blocked is None and self.lots > 0


class RiskManager:
    def __init__(
        self,
        profile: RiskProfile | None = None,
        funded_profile: RiskProfile | None = None,
        costs: CostConfig | None = None,
        pip_value_per_lot: float = 10.0,
    ) -> None:
        self.profile = profile or RiskProfile()
        self.funded_profile = funded_profile or CONSERVATIVE_FUNDED
        self.costs = costs or CostConfig()
        self.pip_value = pip_value_per_lot

    def profile_for(self, account: Account) -> RiskProfile:
        return self.funded_profile if account.stage == "funded" else self.profile

    def worst_case_per_lot(self, stop_pips: float, costs: CostConfig | None = None) -> float:
        c = costs or self.costs
        return self.pip_value * (stop_pips + c.spread_pips + c.slippage_pips + c.gap_allowance_pips) + c.commission_per_lot

    def worst_case_loss(self, lots: float, stop_pips: float, costs: CostConfig | None = None) -> float:
        return lots * self.worst_case_per_lot(stop_pips, costs)

    def day_loss(self, account: Account) -> float:
        s = account.state
        return max(0.0, s.day_start_balance - s.equity)

    def size(self, account: Account, stop_pips: float, costs: CostConfig | None = None) -> SizeDecision:
        if account.status != "active":
            return SizeDecision(0, 0, 0, f"account {account.status}")
        if stop_pips <= 0:
            return SizeDecision(0, 0, 0, "stop distance must be > 0")
        prof = self.profile_for(account)
        s = account.state
        daily_amt = account.program.daily_drawdown.limit * s.start_balance
        if self.day_loss(account) >= prof.soft_daily_stop * daily_amt:
            return SizeDecision(0, 0, 0, "soft daily stop")
        if prof.daily_profit_cap > 0 and s.equity - s.day_start_balance >= prof.daily_profit_cap * s.start_balance:
            return SizeDecision(0, 0, 0, "daily profit cap")
        if prof.max_trades_per_day > 0 and s.day_trades.get(s.day_key, 0) >= prof.max_trades_per_day:
            return SizeDecision(0, 0, 0, "max trades per day")
        streak_loss = _streak(s.results, lambda r: r < 0)
        streak_win = _streak(s.results, lambda r: r > 0)
        today_closed = s.daily_pnl.get(s.day_key) is not None
        if prof.loss_streak_n > 0 and streak_loss >= prof.loss_streak_n and prof.loss_streak_mult == 0 and today_closed:
            return SizeDecision(0, 0, 0, "losing-streak cooldown")

        risk = prof.risk_per_trade * s.start_balance
        if prof.anti_martingale_step > 0:
            risk += min(streak_win, 4) * prof.anti_martingale_step * s.start_balance
        if prof.loss_streak_n > 0 and streak_loss >= prof.loss_streak_n and prof.loss_streak_mult > 0:
            risk *= prof.loss_streak_mult
        tb = account.target_balance
        if tb is not None and account.stage != "funded" and prof.fast_pass_mult != 1.0:
            target_amt = tb - s.start_balance
            if s.balance - s.start_balance < prof.fast_pass_until * target_amt and account.stage == account.program.stages[0].name:
                risk *= prof.fast_pass_mult
        if prof.buffer_risk_frac > 0:
            risk = min(risk, prof.buffer_risk_frac * max(s.equity - account.floor, 0.0))
        if prof.derisk_within > 0 and tb is not None and 0 < tb - s.balance <= prof.derisk_within * s.start_balance:
            risk *= 0.5
        if tb is not None and s.balance >= tb and prof.target_banking:
            risk = 0.0  # target banked: only minimum-size trades to satisfy min trading days
        elif prof.derisk_near_target and tb is not None:
            remaining = tb - s.balance
            if remaining <= 0:
                risk = prof.min_risk_per_trade * s.start_balance
            else:
                # only risk what is needed to finish with one winner (+10% for costs)
                risk = min(risk, max(1.1 * remaining / prof.target_reward_r, prof.min_risk_per_trade * s.start_balance))

        buf = account.buffers()
        allowed = min(buf["daily"], buf["trailing"]) * prof.safety_margin
        allowed = min(allowed, min(buf["daily"], buf["trailing"]) * prof.max_buffer_use)
        per_lot_wc = self.worst_case_per_lot(stop_pips, costs)
        lots = risk / (self.pip_value * stop_pips)
        lots = min(lots, allowed / per_lot_wc) if allowed > 0 else 0.0
        lots = math.floor(lots / prof.lot_step + 1e-9) * prof.lot_step
        if risk == 0.0 and allowed >= prof.min_lot * per_lot_wc:
            lots = prof.min_lot
        if lots < prof.min_lot:
            return SizeDecision(0, 0, 0, "insufficient buffer for minimum size")
        lots = round(lots, 8)
        return SizeDecision(lots, lots * self.pip_value * stop_pips, lots * per_lot_wc)

    def approve(self, account: Account, pos: Position, stop_pips: float, time: pd.Timestamp,
                costs: CostConfig | None = None) -> str | None:
        """Final gate for an already-sized order: rules' pre_trade + worst-case fit."""
        reasons = account.check_pre_trade(pos, time)
        if reasons:
            return "; ".join(reasons)
        wc = self.worst_case_loss(pos.lots, stop_pips, costs)
        buf = account.buffers()
        prof = self.profile_for(account)
        if wc > buf["daily"] * prof.safety_margin:
            return f"worst case {wc:.0f} exceeds daily buffer {buf['daily']:.0f}"
        if wc > buf["trailing"] * prof.safety_margin:
            return f"worst case {wc:.0f} exceeds trailing buffer {buf['trailing']:.0f}"
        return None


def _streak(results: list[float], cond) -> int:
    n = 0
    for r in reversed(results):
        if not cond(r):
            break
        n += 1
    return n
