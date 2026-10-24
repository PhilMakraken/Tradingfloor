"""cBot 6/7 entry blocks (blocks_cbot): registration, boolean output, causality, and agreement with the bot replays."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propfloor.data.clean import resample
from propfloor.data.synthetic import make_bars
from propfloor.strategies.blocks import ENTRIES, Ctx, sample_params

NAMES = ["breakout_runner", "macd_dema"]
USER = {"breakout_runner": {"lookback": 24, "fast": 50, "slow": 200, "daily_gate": True},
        "macd_dema": {"fast": 12, "slow": 26, "signal": 9, "dema_n": 100, "htf_mult": 3}}
PARQUET = Path(__file__).resolve().parents[1] / "data" / "parquet" / "XAUUSD_M1.parquet"


@pytest.fixture(scope="module")
def m15():
    return make_bars("2016-01-01", "2019-01-01", seed=5, price0=1200, annual_vol=0.15, freq="15min")


def test_registered_with_small_spaces_containing_user_values():
    for n in NAMES:
        b = ENTRIES[n]
        assert b.family
        for k, v in USER[n].items():
            spec = b.space[k]
            assert (spec[1] <= v <= spec[2]) if spec[0] in ("int", "float") else v in spec[1], (n, k)


@pytest.mark.parametrize("tf", ["H1", "H4"])
@pytest.mark.parametrize("name", NAMES)
def test_boolean_signals_and_causal(m15, name, tf):
    df = resample(m15, tf)
    x = Ctx(df, "XAUUSD", tf)
    rng = np.random.default_rng(1)
    for p in [USER[name]] + [sample_params(ENTRIES[name].space, rng) for _ in range(4)]:
        lo, sh = ENTRIES[name].fn(x, p)
        assert lo.dtype == bool and sh.dtype == bool and len(lo) == len(sh) == x.n
        assert not (lo & sh).any()
        for cut in (x.n - 37, x.n - 200):
            lo2, sh2 = ENTRIES[name].fn(Ctx(df.iloc[:cut], "XAUUSD", tf), p)
            assert (lo[:cut] == lo2).all() and (sh[:cut] == sh2).all(), (name, tf, p, cut)
    lo, sh = ENTRIES[name].fn(x, USER[name])
    assert lo.sum() > 0 and sh.sum() > 0  # both directions fire


def test_one_macd_dema_signal_per_leg(m15):
    from propfloor.strategies.blocks_cbot import _macd_state
    x = Ctx(resample(m15, "H4"), "XAUUSD", "H4")
    lo, sh = ENTRIES["macd_dema"].fn(x, USER["macd_dema"])
    st = _macd_state(x.c, 12, 26, 9)
    leg = np.cumsum(np.r_[0, st[1:] != st[:-1]])
    assert (st[lo] == 1).all() and (st[sh] == -1).all()
    assert pd.Series(leg[lo | sh]).is_unique


@pytest.fixture(scope="module")
def gold():
    if not PARQUET.exists():
        pytest.skip("data/parquet/XAUUSD_M1.parquet missing")
    from propfloor.data.store import DataStore
    df = DataStore(PARQUET.parent).load("XAUUSD", "M15", "2015-01-01", "2021-01-01")
    if df.empty:
        pytest.skip("no XAUUSD bars in 2015-2020")
    return df


def test_breakout_runner_matches_bot6(gold):
    from propfloor.ports import cbots as cb
    tr = cb.run_bot6(gold)
    assert len(tr) > 50
    x = Ctx(resample(gold, "H1"), "XAUUSD", "H1")
    lo, _ = ENTRIES["breakout_runner"].fn(x, USER["breakout_runner"])
    k = x.time.get_indexer(pd.to_datetime(tr.entry_time).dt.floor("1h"))  # the bot buys inside the breaking bar
    assert (k >= 0).all()
    assert lo[k].mean() >= 0.7, lo[k].mean()


def test_macd_dema_matches_bot7(gold):
    from propfloor.ports import cbots as cb
    tr = cb.run_bot7(gold)
    assert len(tr) > 50
    fr = cb.Frames(gold)
    b, codes = fr.bars("H4")  # broker-clock H4 bars as the bot sees them
    first = pd.Series(np.arange(fr.n)).groupby(codes).first().to_numpy()
    h4 = b.copy()
    h4.index = fr.t[first]
    x = Ctx(h4, "XAUUSD", "H4")
    lo, _ = ENTRIES["macd_dema"].fn(x, USER["macd_dema"])
    k = x.time.get_indexer(pd.to_datetime(tr.entry_time)) - 1  # entry at the open after the signal bar
    assert (k >= 0).all()
    assert lo[k].mean() >= 0.7, lo[k].mean()
