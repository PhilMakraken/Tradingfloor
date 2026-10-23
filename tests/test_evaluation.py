import numpy as np
import pandas as pd
import pytest

from propfloor.config import DEFAULT_PROGRAM, load_program
from propfloor.data.synthetic import make_bars
from propfloor.evaluation.challenge import ChallengeSimulator, TradeArrays, summarize
from propfloor.evaluation.stats import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe, trade_summary
from propfloor.evaluation.walkforward import Evaluator, Gates, Universe
from propfloor.strategies.genome import random_genome


def mk(pnls, start="2022-01-03 08:00", mae=-0.3):
    rows = []
    t = pd.Timestamp(start, tz="UTC")
    for p in pnls:
        while t.weekday() >= 4:
            t += pd.Timedelta(days=1)
        rows.append(dict(entry_time=t, exit_time=t + pd.Timedelta(hours=3), mae_time=t + pd.Timedelta(hours=1),
                         mfe_time=t + pd.Timedelta(hours=2), pnl_r=p, mae_r=min(mae, p), mfe_r=max(0.3, p),
                         post_mfe_r=min(0, p), stop_pips=20.0, side=1, symbol="EURUSD"))
        t += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def program():
    return load_program(DEFAULT_PROGRAM)


def test_all_winners_pass_all_losers_fail(program):
    sim = ChallengeSimulator(program, risk_per_trade=0.01)
    s = summarize(sim.evaluate(mk([2.0] * 200), n_windows=10, n_mc=10))
    assert s["p_pass_all"] == 1.0 and s["p_funded_survive"] == 1.0 and s["expected_payout"] > 0
    s = summarize(sim.evaluate(mk([-1.0] * 200), n_windows=10, n_mc=10))
    assert s["p_pass_all"] == 0.0 and s["p_pass_stage1"] == 0.0


def test_mae_breach_counts_even_if_trade_wins(program):
    """A winner whose floating loss touched the floor must still violate the account."""
    sim = ChallengeSimulator(program, risk_per_trade=0.01)
    ta = TradeArrays(mk([1.0] * 20, mae=-30.0))  # deep floating drawdown before winning
    r = sim.run_stage(ta, 0, "stage1")
    assert r.status == "violated" and r.reason in ("max_drawdown", "daily_drawdown")


def test_min_trading_days_respected(program):
    sim = ChallengeSimulator(program, risk_per_trade=0.01)
    r = sim.run_stage(TradeArrays(mk([8.0] * 10)), 0, "stage1")
    assert r.status == "passed" and r.n_trades >= 4


def test_edge_monotonic(program):
    rng = np.random.default_rng(0)
    sim = ChallengeSimulator(program, risk_per_trade=0.005)
    good = summarize(sim.evaluate(mk(rng.choice([2.0, -1.0], p=[0.45, 0.55], size=400)), 30, 30))
    bad = summarize(sim.evaluate(mk(rng.choice([2.0, -1.0], p=[0.25, 0.75], size=400)), 30, 30))
    assert good["p_pass_all"] > bad["p_pass_all"]


def test_bootstrap_resample_no_overlap():
    ta = TradeArrays(mk(np.random.default_rng(1).normal(0.1, 1, 100)))
    r = ta.resampled(np.random.default_rng(2), 250)
    assert r.n == 250 and (r.entry[1:] > r.exit[:-1]).all() and (r.mae_t >= r.entry).all()


def test_stats():
    rng = np.random.default_rng(0)
    r = rng.normal(0.2, 1, 500)
    assert probabilistic_sharpe(r) > 0.99
    assert deflated_sharpe(r, 1000) < probabilistic_sharpe(r)
    assert expected_max_sharpe(1000, 0.01) > expected_max_sharpe(10, 0.01)
    noise = rng.normal(0, 1, 200)
    assert deflated_sharpe(noise, 500) < 0.5
    ts = trade_summary(np.array([1, -1, 2, -1.0]))
    assert ts["win_rate"] == 0.5 and ts["profit_factor"] == 1.5 and ts["max_dd_r"] == 1.0


def test_evaluator_runs_and_rejects_synthetic(program):
    data = {"EURUSD": make_bars("2019-01-01", "2020-06-01", seed=11)}
    u = Universe(data, timeframes=["H1"], synthetic={"EURUSD": True})
    ev = Evaluator(u, program, Gates(min_trades_is=5), quick_sims=5, full_sims=5)
    g = random_genome(np.random.default_rng(5), timeframes=["H1"])
    fit, info = ev.quick(g)
    assert np.isfinite(fit) and "is" in info
    v = ev.validate(g, n_trials=100)
    assert not v.validated and "synthetic data cannot validate" in v.reasons
    is_tr, oos_tr = ev.split(ev.trades(g), "EURUSD")
    if len(is_tr) and len(oos_tr):
        assert is_tr.exit_time.max() < oos_tr.entry_time.min()


def test_short_holdout_judged_by_monte_carlo(program):
    sim = ChallengeSimulator(program, risk_per_trade=0.005)
    rng = np.random.default_rng(3)
    short = mk(rng.choice([3.0, -1.0], p=[0.4, 0.6], size=40))  # good edge, too few trades to finish 2 stages
    res = sim.evaluate(short, 20, 20, mc_len=600)
    assert summarize(res)["p_pass_all"] < summarize(res, mc_only=True)["p_pass_all"]
