import pandas as pd

from propfloor.account import Account
from propfloor.risk import RiskManager, RiskProfile
from propfloor.rules.base import Position
from tests.test_account_rules import T, pos, prog


def test_worst_case_includes_all_costs():
    rm = RiskManager()
    c = rm.costs
    assert rm.worst_case_loss(1.0, 20) == 10 * (20 + c.spread_pips + c.slippage_pips + c.gap_allowance_pips) + c.commission_per_lot


def test_size_basic_and_fits_buffers():
    a = Account(prog())
    rm = RiskManager(RiskProfile(risk_per_trade=0.005, derisk_near_target=False))
    d = rm.size(a, stop_pips=20)
    assert d.ok and abs(d.risk_amount - 500) < 10  # 0.5% of 100k
    assert d.worst_case <= min(a.buffers().values())


def test_size_shrinks_with_small_buffer_and_blocks():
    a = Account(prog(daily_drawdown={"limit": 0.5}))
    a.open(pos())
    a.close("p1", -4_800, T("2024-01-02 11:00"))  # 200 left to the trailing floor
    rm = RiskManager(RiskProfile(risk_per_trade=0.01, derisk_near_target=False))
    d = rm.size(a, stop_pips=20)
    assert d.ok and d.worst_case <= 200 * 0.33 + 1e-6
    a.open(pos("p2"))
    a.close("p2", -195, T("2024-01-02 12:00"))  # 5 left
    assert not rm.size(a, stop_pips=20).ok


def test_soft_daily_stop():
    a = Account(prog())
    a.open(pos())
    a.close("p1", -2_000, T("2024-01-02 11:00"))  # 2k of 3k daily limit used (>60%)
    assert RiskManager().size(a, 20).blocked == "soft daily stop"


def test_derisk_near_target_and_funded_profile():
    a = Account(prog(daily_drawdown={"limit": 0.5}, drawdown={"max_dd": 0.5}))
    a.open(pos())
    a.close("p1", 7_800, T("2024-01-02 11:00"))  # 200 from target
    rm = RiskManager(RiskProfile(risk_per_trade=0.01))
    assert rm.size(a, 20).risk_amount <= 200
    f = Account(prog(), stage="funded")
    assert rm.size(f, 20).risk_amount < rm.size(Account(prog()), 20).risk_amount


def test_approve_blocks_oversized():
    a = Account(prog())
    rm = RiskManager()
    big = Position("x", "EURUSD", 1, 30.0, T("2024-01-02 10:00"))
    assert "buffer" in rm.approve(a, big, 20, pd.Timestamp("2024-01-02 10:00", tz="UTC"))
    small = Position("y", "EURUSD", 1, 1.0, T("2024-01-02 10:00"))
    assert rm.approve(a, small, 20, T("2024-01-02 10:00")) is None
