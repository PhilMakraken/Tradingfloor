import pandas as pd
import pytest

from propfloor.account import Account
from propfloor.config import DEFAULT_PROGRAM, ProgramConfig, RuleSpec, load_program
from propfloor.rules.base import RULE_REGISTRY, Position, Rule, build_rules, register_rule


def T(s):
    return pd.Timestamp(s, tz="UTC")


def prog(**over) -> ProgramConfig:
    p = load_program(DEFAULT_PROGRAM)
    data = p.model_dump()
    data["rules"] = []
    for st in data["stages"]:
        st["rules"] = []
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    return ProgramConfig.model_validate(data)


def pos(i="p1", t="2024-01-02 10:00", lots=1.0):
    return Position(id=i, symbol="EURUSD", side=1, lots=lots, entry_time=T(t))


def test_load_default_program():
    p = load_program(DEFAULT_PROGRAM)
    assert len(p.stages) == 2 and p.drawdown.max_dd == 0.05 and p.daily_drawdown.limit == 0.03
    assert all(isinstance(r, Rule) for r in build_rules(p.rules_for("stage1"), p.plugins))


def test_trailing_floor_follows_peak_equity_and_touch_violates():
    a = Account(prog(daily_drawdown={"limit": 0.5}))
    a.open(pos())
    assert a.floor == 95_000
    a.mark(T("2024-01-02 11:00"), {"p1": 2_000})  # peak equity 102k -> floor 97k (lock caps at 100k)
    assert a.floor == 97_000
    v = a.mark(T("2024-01-02 12:00"), {"p1": -3_000})  # equity 97k touches floor
    assert v and v.rule == "max_drawdown" and a.status == "violated"
    # permanently closed
    assert a.close("p1", 5_000, T("2024-01-02 13:00")) is v and a.state.balance == 100_000


def test_lock_at_start_and_balance_basis():
    a = Account(prog(daily_drawdown={"limit": 0.5}, drawdown={"lock_at_start": True}))
    a.open(pos())
    a.mark(T("2024-01-02 11:00"), {"p1": 9_000})
    assert a.floor == 100_000  # locked at start balance
    b = Account(prog(daily_drawdown={"limit": 0.5}, drawdown={"trail_basis": "balance", "lock_at_start": False}))
    b.open(pos())
    b.mark(T("2024-01-02 11:00"), {"p1": 9_000})
    assert b.floor == 95_000  # floating peak ignored with balance basis
    b.close("p1", 9_000, T("2024-01-02 12:00"))
    assert b.floor == 104_000


def test_static_drawdown():
    a = Account(prog(daily_drawdown={"limit": 0.5}, drawdown={"trailing": False}))
    a.open(pos())
    a.mark(T("2024-01-02 11:00"), {"p1": 4_000})
    assert a.floor == 95_000


def test_daily_drawdown_reset_in_timezone():
    a = Account(prog())  # 3% daily, reset 17:00 New York
    a.open(pos(t="2024-01-02 15:00"))
    a.close("p1", -2_500, T("2024-01-02 16:00"))  # 11:00 NY
    a.open(pos("p2", "2024-01-02 20:00"))  # 15:00 NY same trading day
    v = a.close("p2", -600, T("2024-01-02 21:00"))  # day loss 3.1k > 3k
    assert v and v.rule == "daily_drawdown"

    b = Account(prog())
    b.open(pos(t="2024-01-02 15:00"))
    b.close("p1", -2_500, T("2024-01-02 16:00"))
    b.open(pos("p2", "2024-01-02 23:00"))  # 18:00 NY -> new day
    assert b.close("p2", -600, T("2024-01-02 23:30")) is None
    assert b.daily_floor == pytest.approx(97_500 - 3_000)


def test_daily_equity_basis():
    a = Account(prog(daily_drawdown={"basis": "equity", "reset_time": "00:00", "timezone": "UTC"}))
    a.open(pos(t="2024-01-02 22:00"))
    a.mark(T("2024-01-02 23:00"), {"p1": 2_000})
    a.mark(T("2024-01-03 00:30"), {"p1": 2_000})  # new day, start equity 102k
    assert a.daily_floor == 99_000


def test_pass_requires_target_and_min_days():
    p = prog(stages=[{"name": "stage1", "profit_target": 0.08, "rules": [{"name": "min_trading_days", "params": {"days": 2}}]}],
             daily_drawdown={"limit": 0.5}, drawdown={"max_dd": 0.5})
    a = Account(p)
    a.open(pos(t="2024-01-02 10:00"))
    a.close("p1", 9_000, T("2024-01-02 11:00"))
    assert a.status == "active" and "trading days" in a.completion_blocks()[0]
    a.open(pos("p2", "2024-01-03 10:00"))
    a.close("p2", 10, T("2024-01-03 11:00"))
    assert a.status == "passed"


def test_rules_builtin():
    news = RULE_REGISTRY["news_blackout"](times=["2024-01-02 13:30"], minutes_before=2, minutes_after=2)
    assert news.in_blackout(T("2024-01-02 13:29")) and news.in_blackout(T("2024-01-02 13:32"))
    assert not news.in_blackout(T("2024-01-02 13:33"))

    p = prog(rules=[{"name": "no_weekend_holding", "params": {}}], daily_drawdown={"limit": 0.5})
    a = Account(p)
    a.open(pos(t="2024-01-05 15:00"))  # Friday
    v = a.close("p1", 10, T("2024-01-08 10:00"))  # Monday
    assert v and v.rule == "no_weekend_holding"

    p = prog(rules=[{"name": "max_time_in_trade", "params": {"max_hours": 2}}], daily_drawdown={"limit": 0.5})
    a = Account(p)
    a.open(pos())
    assert a.mark(T("2024-01-02 13:00"), {"p1": 0}).rule == "max_time_in_trade"

    p = prog(rules=[{"name": "max_lot", "params": {"max_lots": 5}}])
    a = Account(p)
    assert a.check_pre_trade(pos(lots=6), T("2024-01-02 10:00"))
    assert a.open(pos(lots=6)).rule == "max_lot"


def test_consistency_rule_gate():
    r = RULE_REGISTRY["consistency_rule"](max_day_share=0.5)
    a = Account(prog(daily_drawdown={"limit": 0.5}), rules=[r])
    a.open(pos(t="2024-01-02 10:00"))
    a.close("p1", 3_000, T("2024-01-02 11:00"))
    a.open(pos("p2", "2024-01-03 10:00"))
    a.close("p2", 1_000, T("2024-01-03 11:00"))
    assert r.completion_block(a.state)


def test_custom_plugin_rule_without_core_changes():
    @register_rule("no_trading_on_mondays_test")
    class NoMon(Rule):
        def check(self, state, event):
            if event.kind == "open" and event.time.weekday() == 0:
                return self.violation("monday", event.time)

    rules = build_rules([RuleSpec(name="no_trading_on_mondays_test")])
    a = Account(prog(), rules=rules)
    assert a.open(pos(t="2024-01-08 10:00")).rule == "no_trading_on_mondays_test"
    with pytest.raises(KeyError):
        build_rules([RuleSpec(name="does_not_exist")])
