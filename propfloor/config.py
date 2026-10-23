"""Program (prop-firm challenge) configuration, loaded from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class RuleSpec(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class DrawdownConfig(BaseModel):
    """Overall max drawdown. Fractions are of the starting balance."""

    max_dd: float = 0.05
    trailing: bool = True
    trail_basis: Literal["equity", "balance"] = "equity"
    lock_at_start: bool = False  # floor stops trailing once it reaches the starting balance


class DailyDrawdownConfig(BaseModel):
    limit: float = 0.03
    basis: Literal["balance", "equity", "max_balance_equity"] = "balance"
    reset_time: str = "00:00"  # HH:MM in `timezone`
    timezone: str = "UTC"
    limit_of: Literal["starting_balance", "day_start"] = "starting_balance"

    @field_validator("reset_time")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        h, m = v.split(":")
        assert 0 <= int(h) < 24 and 0 <= int(m) < 60
        return v

    @property
    def reset_minutes(self) -> int:
        h, m = self.reset_time.split(":")
        return int(h) * 60 + int(m)


class StageConfig(BaseModel):
    name: str
    profit_target: float | None = 0.08  # None => no target (funded)
    max_calendar_days: int | None = None
    rules: list[RuleSpec] = Field(default_factory=list)  # stage-specific, appended to global rules


class FundedConfig(BaseModel):
    profit_split: float = 0.8
    payout_period_days: int = 30
    min_payout: float = 0.0
    rules: list[RuleSpec] = Field(default_factory=list)


class CostConfig(BaseModel):
    """Default execution cost assumptions (in pips) used by the risk manager/backtests."""

    spread_pips: float = 1.0
    slippage_pips: float = 0.3
    commission_per_lot: float = 7.0  # round turn, account currency
    gap_allowance_pips: float = 5.0


class ProgramConfig(BaseModel):
    name: str
    currency: str = "USD"
    starting_balance: float = 100_000.0
    stages: list[StageConfig]
    funded: FundedConfig = Field(default_factory=FundedConfig)
    drawdown: DrawdownConfig = Field(default_factory=DrawdownConfig)
    daily_drawdown: DailyDrawdownConfig = Field(default_factory=DailyDrawdownConfig)
    rules: list[RuleSpec] = Field(default_factory=list)
    plugins: list[str] = Field(default_factory=list)  # modules imported to register extra rules
    costs: CostConfig = Field(default_factory=CostConfig)

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages] + ["funded"]

    def rules_for(self, stage: str) -> list[RuleSpec]:
        if stage == "funded":
            return list(self.rules) + list(self.funded.rules)
        st = next(s for s in self.stages if s.name == stage)
        return list(self.rules) + list(st.rules)

    def stage(self, name: str) -> StageConfig:
        if name == "funded":
            return StageConfig(name="funded", profit_target=None, rules=self.funded.rules)
        return next(s for s in self.stages if s.name == name)


def load_program(path: str | Path) -> ProgramConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return ProgramConfig.model_validate(raw)


DEFAULT_PROGRAM = Path(__file__).resolve().parent.parent / "programs" / "generic_2stage.yaml"
