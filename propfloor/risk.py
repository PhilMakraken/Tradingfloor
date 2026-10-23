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

        risk = prof.risk_per_trade * s.start_balance
        tb = account.target_balance
        if prof.derisk_near_target and tb is not None:
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
