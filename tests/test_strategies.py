import numpy as np
import pandas as pd
import pytest

from propfloor.data.clean import resample
from propfloor.data.synthetic import make_bars
from propfloor.strategies import indicators as ind
from propfloor.strategies.backtest import run_backtest
from propfloor.strategies.blocks import ENTRIES, FILTERS, Ctx, sample_params
from propfloor.strategies.genome import StrategyGenome, crossover, mutate, random_genome


@pytest.fixture(scope="module")
def ctx():
    return Ctx(resample(make_bars("2020-01-01", "2020-07-01", seed=7), "H1"), "EURUSD", "H1")


def bars(rows, start="2024-01-02 00:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


def flat_ctx(n=40, p=1.1):
    rows = [(p, p + 0.001, p - 0.001, p)] * n
    return rows


def run(rows, long_at, exit_p, short_at=()):
    c = Ctx(bars(rows), "EURUSD", "H1")
    lo = np.zeros(c.n, bool)
    sh = np.zeros(c.n, bool)
    lo[list(long_at)] = True
    sh[list(short_at)] = True
    return run_backtest(c, lo, sh, exit_p, spread_pips=0, slippage_pips=0, commission_per_lot=0), c


def test_indicators_basic():
    x = np.arange(1, 31, dtype=float)
    assert np.isnan(ind.sma(x, 5)[3]) and ind.sma(x, 5)[4] == 3
    assert ind.rsi(x, 14)[-1] == 100
    sh, sl = ind.swing_levels(np.array([1, 2, 5, 2, 1, 2, 1.0]), np.array([0, 1, 4, 1, 0.5, 1, 0.0]), 2)
    assert sh[4] == 5 and np.isnan(sh[3])  # swing high known only k bars later


def test_entry_next_open_and_target():
    rows = flat_ctx(30)
    rows[21] = (1.1, 1.1060, 1.0995, 1.105)  # target bar
    tr, c = run(rows, [20], {"sl_type": "atr", "sl_atr": 1.5, "tp_r": 2.0})
    t = tr.iloc[0]
    assert t.entry_idx == 21 and t.entry == pytest.approx(1.1)
    assert t.reason == "target" and t.pnl_r == pytest.approx(2.0)


def test_stop_wins_when_both_hit_and_gap_through_stop():
    rows = flat_ctx(30)
    rows[21] = (1.1, 1.12, 1.08, 1.1)
    tr, _ = run(rows, [20], {"sl_type": "atr", "sl_atr": 1.5, "tp_r": 2.0})
    assert tr.iloc[0].reason == "stop" and tr.iloc[0].pnl_r == pytest.approx(-1.0)
    rows = flat_ctx(30)
    rows[22] = (1.09, 1.09, 1.085, 1.087)  # opens below the stop
    tr, _ = run(rows, [20], {"sl_type": "atr", "sl_atr": 1.5, "tp_r": 0})
    t = tr.iloc[0]
    assert t.reason == "stop" and t.pnl_r < -1.0 and t.mae_r <= t.pnl_r


def test_mae_mfe_and_time_stop():
    rows = flat_ctx(40)
    rows[22] = (1.1, 1.1025, 1.0990, 1.1)
    rows[23] = (1.1, 1.1005, 1.0985, 1.1)
    tr, _ = run(rows, [20], {"sl_type": "atr", "sl_atr": 1.5, "tp_r": 0, "time_stop": 5})
    t = tr.iloc[0]
    assert t.reason == "time" and t.exit_idx == 25
    assert t.mfe_idx == 22 and t.mae_idx == 23 and t.mfe_r > 0.5 and t.mae_r < -0.4
    assert t.post_mfe_r == pytest.approx(t.mae_r)


def test_costs_reduce_pnl(ctx):
    g = StrategyGenome.from_dict(random_genome(np.random.default_rng(1)).to_dict())
    lo, sh = g.signals(ctx)
    a = run_backtest(ctx, lo, sh, g.exit, spread_pips=0, slippage_pips=0, commission_per_lot=0)
    b = run_backtest(ctx, lo, sh, g.exit, spread_pips=2, slippage_pips=0.5, commission_per_lot=7)
    if len(a) and len(b):
        assert b.pnl_r.sum() < a.pnl_r.sum()


def test_force_flat_before_weekend(ctx):
    lo = np.ones(ctx.n, bool)
    tr = run_backtest(ctx, lo, np.zeros(ctx.n, bool), {"sl_atr": 50, "tp_r": 0})
    ny_entry = tr.entry_time.dt.tz_convert("America/New_York")
    assert not ((ny_entry.dt.weekday == 4) & (ny_entry.dt.hour >= 16)).any()
    from propfloor.rules.builtin import _next_friday_cutoff
    for e, x in zip(tr.entry_time, tr.exit_time):
        assert x <= _next_friday_cutoff(e, "America/New_York", "16:50")


def test_all_blocks_run_without_lookahead(ctx):
    rng = np.random.default_rng(0)
    cut = ctx.n - 50
    short = Ctx(ctx.df.iloc[:cut], "EURUSD", "H1")
    for name, b in {**ENTRIES, **FILTERS}.items():
        p = sample_params(b.space, rng)
        lo, sh = b.fn(ctx, p)
        assert lo.dtype == bool and len(lo) == ctx.n, name
        lo2, sh2 = b.fn(short, p)
        # values on truncated history must equal the full-history values (no future data used)
        assert (lo[: cut - 1] == lo2[: cut - 1]).all() and (sh[: cut - 1] == sh2[: cut - 1]).all(), name


def test_genome_roundtrip_mutation_crossover(ctx):
    rng = np.random.default_rng(3)
    a, b = random_genome(rng), random_genome(rng)
    assert StrategyGenome.from_dict(a.to_dict()).key() == a.key()
    for _ in range(50):
        m = mutate(a, rng)
        m.backtest(ctx)
    c = crossover(a, b, rng)
    assert c.tf in (a.tf, b.tf)
    assert isinstance(a.describe(), str)
