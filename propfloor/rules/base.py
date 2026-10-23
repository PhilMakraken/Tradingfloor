"""Plugin rule system.

A rule is a class with ``@register_rule("name")`` implementing any of:

* ``check(state, event) -> Violation | None``   hard rule; a violation closes the account
* ``pre_trade(state, position, time) -> str | None``  used by the risk manager to block orders
* ``completion_block(state) -> str | None``     gate: blocks passing a stage / payout, never violates

Extra rules are loaded from modules listed in a program's ``plugins:`` key, or from the
``propfloor.rules`` entry-point group, so core code never needs to change.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from propfloor.account import AccountState
    from propfloor.config import RuleSpec


@dataclass
class Position:
    id: str
    symbol: str
    side: int  # +1 long / -1 short
    lots: float
    entry_time: pd.Timestamp
    entry_price: float = 0.0
    stop_price: float = 0.0
    risk_amount: float = 0.0  # money lost if stopped out at the stop (excl. gap)
    meta: dict[str, Any] = field(default_factory=dict)


EventKind = Literal["open", "close", "mark", "time"]


@dataclass
class Event:
    kind: EventKind
    time: pd.Timestamp
    position: Position | None = None
    pnl: float = 0.0  # realised pnl for "close"


@dataclass
class Violation:
    rule: str
    message: str
    time: pd.Timestamp | None = None


class Rule:
    name: str = "rule"

    def __init__(self, **params: Any) -> None:
        self.params = params

    def check(self, state: "AccountState", event: Event) -> Violation | None:
        return None

    def pre_trade(self, state: "AccountState", position: Position, time: pd.Timestamp) -> str | None:
        return None

    def completion_block(self, state: "AccountState") -> str | None:
        return None

    def violation(self, msg: str, time: pd.Timestamp | None = None) -> Violation:
        return Violation(self.name, msg, time)


RULE_REGISTRY: dict[str, type[Rule]] = {}


def register_rule(name: str):
    def deco(cls: type[Rule]) -> type[Rule]:
        cls.name = name
        RULE_REGISTRY[name] = cls
        return cls

    return deco


_EP_LOADED = False


def _load_entry_points() -> None:
    global _EP_LOADED
    if _EP_LOADED:
        return
    _EP_LOADED = True
    for ep in entry_points(group="propfloor.rules"):
        ep.load()


def build_rules(specs: "list[RuleSpec]", plugins: list[str] | None = None) -> list[Rule]:
    import propfloor.rules.builtin  # noqa: F401  registers built-ins

    for mod in plugins or []:
        importlib.import_module(mod)
    _load_entry_points()
    out = []
    for spec in specs:
        if spec.name not in RULE_REGISTRY:
            raise KeyError(f"Unknown rule '{spec.name}'. Registered: {sorted(RULE_REGISTRY)}")
        out.append(RULE_REGISTRY[spec.name](**spec.params))
    return out
