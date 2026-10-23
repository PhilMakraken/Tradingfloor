"""Tests for the catalogue extensions: intermarket, ensembles/ML, exits, overlays, rotation."""
import numpy as np
import pandas as pd
import pytest

from propfloor.account import Account
from propfloor.config import DEFAULT_PROGRAM, load_program
from propfloor.data.synthetic import make_bars
from propfloor.evaluation.challenge import ChallengeSimulator, TradeArrays
from propfloor.evaluation.walkforward import Universe
from propfloor.floor.floor import rotate_monthly
from propfloor.risk import RiskManager, RiskProfile
from propfloor.strategies.backtest import run_backtest
from propfloor.strategies.blocks import ENTRIES, FILTERS, OVERLAY_SPACE, Ctx, sample_params
from propfloor.strategies.blocks_intermarket import currency_strength, log_rate
from propfloor.strategies.genome import random_genome, signal_decay_hold
from tests.test_account_rules import T, pos, prog
from tests.test_evaluation import mk
from tests.test_strategies import bars, flat_ctx, run

SYMS = {"EURUSD": 1.15, "GBPUSD": 1.3, "EURGBP": 0.88, "USDJPY": 110.0, "EURJPY": 125.0, "AUDJPY": 80.0,
        "USDCHF": 0.95, "XAUUSD": 1500.0, "AUDUSD": 0.7}


@pytest.fixture(scope="module")
def uni():
    data = {s: make_bars("2020-01-01", "2020-04-01", seed=i, price0=p, freq="15min") for i, (s, p) in enumerate(SYMS.items())}
    return Universe(data, timeframes=["H1"])


def test_intermarket_helpers(uni):
    x = uni.ctx[("EURUSD", "H1")]
    assert set(x.peers) == set(SYMS) - {"EURUSD"}
    inv = log_rate(x, "USD", "EUR")
    assert np.allclose(inv, -np.log(x.c))
    cs = currency_strength(x, 10)
    assert {"EUR", "USD", "JPY", "GBP"} <= set(cs)


def test_intermarket_blocks_causal_and_active(uni):
    rng = np.random.default_rng(0)
    im = [n for n, b in ENTRIES.items() if b.family == "intermarket"] + ["jpy_risk_sentiment", "safe_haven_confirm"]
    full = uni.ctx[("EURGBP", "H1")]
    cut = full.n - 40
    short = Ctx(full.df.iloc[:cut], "EURGBP", "H1")
    short.peers = {s: Ctx(c.df[c.df.index < full.time[cut]], s, "H1") for s, c in full.peers.items()}
    fired = 0
    for name in im:
        b = ENTRIES.get(name) or FILTERS[name]
        for _ in range(3):
            p = sample_params(b.space, rng)
            lo, sh = b.fn(full, p)
            lo2, sh2 = b.fn(short, p)
            assert (lo[: cut - 1] == lo2[: cut - 1]).all() and (sh[: cut - 1] == sh2[: cut - 1]).all(), name
            fired += lo.sum() + sh.sum()
    assert fired > 0


def test_ml_classifier_walk_forward_is_causal():
    df = make_bars("2020-01-01", "2021-06-01", seed=3, freq="1h")
    full = Ctx(df, "EURUSD", "H1")
    cut = 6000 + 1200  # inside the first predicted fold (H1 folds = 6000 bars)
    short = Ctx(df.iloc[:cut], "EURUSD", "H1")
    p = {"horizon": 8, "thr": 0.55}
    lo, sh = ENTRIES["ml_classifier"].fn(full, p)
    lo2, sh2 = ENTRIES["ml_classifier"].fn(short, p)
    assert (lo[: cut - 10] == lo2[: cut - 10]).all() and (sh[: cut - 10] == sh2[: cut - 10]).all()
    assert not lo[:6000].any() and not sh[:6000].any()  # first fold has no model


def test_vote_and_router_run(uni):
    x = uni.ctx[("EURUSD", "H1")]
    rng = np.random.default_rng(1)
    for name in ("vote_ensemble", "regime_router"):
        lo, sh = ENTRIES[name].fn(x, sample_params(ENTRIES[name].space, rng))
        assert lo.dtype == bool and not (lo & sh).any()


def test_partial_profit_banks_half_and_moves_stop():
    rows = flat_ctx(30)
    rows[21] = (1.1, 1.1035, 1.0995, 1.1005)  # +1R high (ATR 0.002 -> 1R = 0.003)
    rows[22] = (1.1005, 1.1006, 1.0980, 1.0985)  # back down through entry -> breakeven stop
    tr, _ = run(rows, [20], {"sl_atr": 1.5, "tp_r": 0, "partial_r": 1.0})
    t = tr.iloc[0]
    assert t.reason == "trail" and t.pnl_r == pytest.approx(0.5, abs=0.05)
    tr0, _ = run(rows, [20], {"sl_atr": 1.5, "tp_r": 0})
    assert tr0.iloc[0].pnl_r < t.pnl_r


def test_exit_modes_and_chandelier():
    rows = flat_ctx(40)
    for i in range(21, 30):
        p = 1.1 + 0.0004 * (i - 20)
        rows[i] = (p - 0.0004, p + 0.0002, p - 0.0005, p)
    rows[30] = (1.1036, 1.1037, 1.1010, 1.1012)
    tr, _ = run(rows, [20], {"sl_atr": 3, "tp_r": 0, "trail_atr": 1.0, "trail_type": "chandelier"})
    assert tr.iloc[0].reason == "trail" and tr.iloc[0].pnl_r > 0
    tr, _ = run(rows, [20], {"sl_atr": 3, "tp_r": 0, "exit_mode": "session_end", "exit_hour": 23})
    assert tr.iloc[0].reason == "opposite" and tr.iloc[0].exit_idx == 23


def test_signal_decay_hold():
    df = make_bars("2020-01-01", "2020-06-01", seed=4, freq="1h")
    x = Ctx(df, "EURUSD", "H1")
    lo = np.zeros(x.n, bool)
    lo[::50] = True
    h = signal_decay_hold(x, lo, np.zeros(x.n, bool))
    assert h in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)


def _acc():
    return Account(prog(daily_drawdown={"limit": 0.5}, drawdown={"max_dd": 0.5}))


def test_overlay_daily_cap_max_trades_and_banking():
    rm = RiskManager(RiskProfile(daily_profit_cap=0.01, max_trades_per_day=2))
    a = _acc()
    a.open(pos("a"))
    a.close("a", 600, T("2024-01-02 11:00"))
    assert rm.size(a, 20).ok
    a.open(pos("b", "2024-01-02 12:00"))
    a.close("b", 600, T("2024-01-02 13:00"))
    assert rm.size(a, 20).blocked == "daily profit cap"
    rm2 = RiskManager(RiskProfile(max_trades_per_day=2))
    assert rm2.size(a, 20).blocked == "max trades per day"
    b = Account(prog(daily_drawdown={"limit": 0.5}, drawdown={"max_dd": 0.5},
                     stages=[{"name": "stage1", "profit_target": 0.08,
                              "rules": [{"name": "min_trading_days", "params": {"days": 3}}]}]))
    b.open(pos())
    b.close("p1", 8_500, T("2024-01-02 11:00"))  # target reached (min days not yet)
    d = RiskManager(RiskProfile(target_banking=True)).size(b, 20)
    assert d.ok and d.lots == 0.01


def test_overlay_buffer_streak_and_anti_martingale():
    a = Account(prog(daily_drawdown={"limit": 0.5}))
    base = RiskManager(RiskProfile(risk_per_trade=0.01, derisk_near_target=False)).size(a, 20).risk_amount
    buf = RiskManager(RiskProfile(risk_per_trade=0.01, buffer_risk_frac=0.1, derisk_near_target=False)).size(a, 20).risk_amount
    assert buf <= 0.1 * 5_000 + 1 and buf < base
    for i, pnl in enumerate([-100, -100, -100]):
        a.open(pos(f"l{i}", f"2024-01-02 1{i}:00"))
        a.close(f"l{i}", pnl, T(f"2024-01-02 1{i}:30"))
    assert RiskManager(RiskProfile(loss_streak_n=3, loss_streak_mult=0.0)).size(a, 20).blocked == "losing-streak cooldown"
    half = RiskManager(RiskProfile(risk_per_trade=0.01, loss_streak_n=3, loss_streak_mult=0.5, derisk_near_target=False))
    assert half.size(a, 20).risk_amount < base * 0.6
    w = _acc()
    for i in range(2):
        w.open(pos(f"w{i}", f"2024-01-02 1{i}:00"))
        w.close(f"w{i}", 100, T(f"2024-01-02 1{i}:30"))
    am = RiskManager(RiskProfile(risk_per_trade=0.005, anti_martingale_step=0.001, derisk_near_target=False))
    assert am.size(w, 20).risk_amount > RiskManager(RiskProfile(risk_per_trade=0.005, derisk_near_target=False)).size(w, 20).risk_amount


def test_equity_curve_filter_and_overlay_sim():
    tr = mk([1.0] * 5 + [-1.0] * 20 + [1.0] * 20)
    ta = TradeArrays(tr)
    m = ta.equity_curve_mask(5)
    assert m[:5].all() and not m[10:20].all()
    prog_ = load_program(DEFAULT_PROGRAM)
    sim = ChallengeSimulator(prog_, risk_per_trade=0.005, overlay=sample_params(OVERLAY_SPACE, np.random.default_rng(0)))
    r = sim.run_stage(ta, 0, "stage1")
    assert r.status in ("passed", "violated", "exhausted", "expired")


def test_rotation_keeps_best_recent():
    good = mk([1.0] * 200)
    bad = mk([-1.0] * 200)
    out = rotate_monthly({"g": good, "b": bad}, top=1, lookback_days=60)
    assert len(out["g"]) > 0 and len(out["b"]) == 0


def test_random_genomes_with_overlays_backtest(uni):
    rng = np.random.default_rng(9)
    for _ in range(40):
        g = random_genome(rng, ["H1"], symbols=list(SYMS))
        assert g.overlay
        g.backtest(uni.ctx[(g.symbol, "H1")])
