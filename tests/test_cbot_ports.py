import numpy as np
import pandas as pd
import pytest

from propfloor.data.synthetic import make_bars
from propfloor.ports import cbots as cb

COLS = ["entry_time", "exit_time", "side", "entry", "exit", "pnl_r", "net"]


@pytest.fixture(scope="module")
def gold():
    return make_bars("2019-01-01", "2020-07-01", seed=3, price0=1300, annual_vol=0.15, freq="15min")


@pytest.fixture(scope="module")
def runs(gold):
    return {name: fn(gold) for name, fn in cb.BOTS.items()}


def test_all_bots_run_and_return_trades(runs):
    for name, tr in runs.items():
        assert list(tr.columns[:7]) == COLS, name
        assert len(tr) > 0, name
        assert (tr.side == "long").all(), name  # all four are long-only with the user's parameters
        assert (pd.to_datetime(tr.exit_time) >= pd.to_datetime(tr.entry_time)).all()
        assert np.isfinite(tr.pnl_r).all()
        # compounding: balance column is the running sum of net from 10,000
        assert np.allclose(tr.balance.to_numpy(), 10_000 + tr.net.cumsum().to_numpy())
        # one position at a time
        et, xt = pd.to_datetime(tr.entry_time).to_numpy(), pd.to_datetime(tr.exit_time).to_numpy()
        assert (et[1:] >= xt[:-1]).all(), name
        s = cb.summarize(tr)
        assert s["trades"] == len(tr) and 0 <= s["win_rate"] <= 1


@pytest.mark.parametrize("name", list(cb.BOTS))
def test_no_lookahead_truncation(gold, runs, name):
    cut = pd.Timestamp("2020-02-13 13:15", tz="UTC")
    full = runs[name]
    part = cb.BOTS[name](gold[gold.index < cut])
    a = full[pd.to_datetime(full.exit_time) < cut - pd.Timedelta(minutes=15)].drop(columns=["info"])
    b = part[part.reason != "end_of_data"].drop(columns=["info"])
    b = b[pd.to_datetime(b.exit_time) < cut - pd.Timedelta(minutes=15)]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    ea, eb = full.attrs["equity"], part.attrs["equity"]
    assert np.allclose(ea[ea.index < cut - pd.Timedelta(hours=1)], eb[eb.index < cut - pd.Timedelta(hours=1)])


def test_trade_start_blocks_earlier_entries(gold):
    tr = cb.run_bot7(gold, trade_start="2020-01-01")
    assert (pd.to_datetime(tr.entry_time) >= pd.Timestamp("2020-01-01", tz="UTC")).all()
    assert tr.attrs["equity"].index[0] >= pd.Timestamp("2020-01-01", tz="UTC")


def test_params_match_user_json_values():
    assert cb.BOT4_PARAMS["Partial Close %"] == 33 and cb.BOT4_PARAMS["Trailing Stop (x ATR, 0 = off)"] == 10
    assert cb.BOT5_PARAMS["EMA Period"] == 40 and cb.BOT6_PARAMS["Stop Buffer (x ATR14)"] == 0.4
    assert cb.BOT7_PARAMS["DEMA Period"] == 100
    assert all(p.get("Risk %", p.get("Risk % per trade", p.get("Risk % per trade (0 = fixed lots)"))) == 1.0
               for p in cb.DEFAULT_PARAMS.values())


# ---------------------------------------------------------------------------------------------------------------
# mechanics on the position engine
# ---------------------------------------------------------------------------------------------------------------
T = pd.Timestamp("2024-01-02 10:00", tz="UTC")
RULES = {"Use Portfolio Stop Rules": True, "Half Risk At Account Drawdown %": 12.0,
         "Pause New Trades At Account Drawdown %": 20.0}


def test_partial_then_breakeven_exit():
    bk = cb.Book(cb.ZERO_COSTS, 10_000, RULES, partial_pct=33, breakeven=True)
    units = bk.units_for(1.0, 10.0)
    assert units == 10  # 100 risk / 10 stop
    bk.open(1, T, 2000.0, units, 10.0, None, 0, partial_r=2.0)
    bk.segment(2000.0, 2019.0, T)  # not yet +2R
    assert bk.pos.partial_done is False and bk.pos.units == 10
    bk.segment(2019.0, 2025.0, T)  # crosses +2R at 2020
    assert bk.pos.units == 7 and bk.pos.sl == 2000.0  # floor(33% of 10) = 3 closed, stop to breakeven
    assert bk.balance == pytest.approx(10_000 + 3 * 20.0)
    bk.segment(2025.0, 1990.0, T)  # back through breakeven
    assert bk.pos is None
    t = bk.trades[-1]
    assert t["n_closes"] == 2 and t["reason"] == "sl_after_partial"
    assert t["net"] == pytest.approx(60.0) and t["pnl_r"] == pytest.approx(0.6)


def test_partial_skipped_when_too_small_but_breakeven_still_moves():
    bk = cb.Book(cb.ZERO_COSTS, 10_000, RULES, partial_pct=33, breakeven=True)
    bk.open(1, T, 2000.0, 2.0, 10.0, None, 0, partial_r=2.0)
    bk.segment(2000.0, 2030.0, T)
    assert bk.pos.units == 2 and bk.pos.sl == 2000.0  # floor(0.66) = 0 -> no close, still breakeven


def test_stop_target_and_gap_fills_with_costs():
    c = cb.Costs(spread=0.2, slippage=0.05, commission_per_unit=0.07)
    bk = cb.Book(c, 10_000, RULES)
    bk.open(1, T, 2000.0, 10, 5.0, 10.0, 0)
    assert bk.pos.entry == pytest.approx(2000.25) and bk.pos.sl == pytest.approx(1995.25)
    assert bk.pos.tp == pytest.approx(2010.25)
    bk.segment(2000.0, 2009.0, T)
    assert bk.pos is not None
    bk.segment(2009.0, 2012.0, T, gap=True)  # gap through the target fills at the gap price
    t = bk.trades[-1]
    assert t["reason"] == "tp" and t["exit"] == pytest.approx(2012.0 - 0.05)
    assert t["net"] == pytest.approx((2011.95 - 2000.25) * 10 - 0.7)
    bk.open(1, T, 2000.0, 10, 5.0, None, 0)
    bk.segment(2000.0, 1990.0, T)  # continuous move fills at the stop level
    assert bk.trades[-1]["exit"] == pytest.approx(1995.25 - 0.05)


def test_short_uses_ask_for_stops():
    c = cb.Costs(spread=0.5, slippage=0.0, commission_per_unit=0.0)
    bk = cb.Book(c, 10_000, RULES)
    bk.open(-1, T, 2000.0, 10, 5.0, None, 0)  # sell at bid 2000, stop 2005 (ask)
    bk.segment(2000.0, 2004.6, T)  # ask 2005.1 >= 2005 -> stopped
    assert bk.pos is None and bk.trades[-1]["exit"] == pytest.approx(2005.0)


def test_portfolio_rules_half_and_pause():
    bk = cb.Book(cb.ZERO_COSTS, 10_000, RULES)
    bk.update_state(0)
    assert bk.eff_risk(1.0, 0) == 1.0
    bk.balance = 8_700  # 13% DD
    assert bk.eff_risk(1.0, 0) == 0.5 and not bk.paused(0)
    bk.balance = 9_000  # still below the peak -> half risk persists
    assert bk.eff_risk(1.0, 0) == 0.5
    bk.balance = 10_100  # new high resets half risk
    assert bk.eff_risk(1.0, 0) == 1.0
    bk.balance = 8_000
    assert bk.paused(0)
    bk.balance = 11_000  # pause is permanent until a manual restart
    assert bk.paused(0)
    off = cb.Book(cb.ZERO_COSTS, 10_000, {**RULES, "Use Portfolio Stop Rules": False})
    off.balance = 5_000
    assert off.eff_risk(1.0, 0) == 1.0 and not off.paused(0)


def test_swing_higher_low_confirmed_k_bars_later():
    low = np.array([10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 9, 8, 7.5, 8, 9, 10, 11, 12, 13.0])
    high = low + 1
    hl, lh, piv, prev = cb.swing_signals(high, low, 2)
    # swing lows at 4 (6.0) and 12 (7.5): the second is a higher low, confirmed at bar 12 + 2
    assert np.flatnonzero(~np.isnan(piv)).tolist() == [6, 14]
    assert not hl[6] and hl[14] and piv[14] == 7.5 and prev[14] == 6.0


def _m15_from_h4(rows, start="2024-01-08 22:00"):
    """M15 bars (16 per H4 bar, broker-clock aligned: 22:00 UTC = 00:00 server in winter) from H4 OHLC rows."""
    out = []
    for o, h, l, c in rows:
        path = np.r_[np.linspace(o, l, 6), np.linspace(l, h, 6)[1:], np.linspace(h, c, 6)[1:]]
        seg = np.interp(np.linspace(0, len(path) - 1, 17), np.arange(len(path)), path)
        for k in range(16):
            a, b = seg[k], seg[k + 1]
            out.append((a, max(a, b), min(a, b), b))
    idx = pd.date_range(start, periods=len(out), freq="15min", tz="UTC")
    df = pd.DataFrame(out, columns=["open", "high", "low", "close"], index=idx)
    df["high"] = np.maximum(df.high, df[["open", "close"]].max(axis=1))
    df["low"] = np.minimum(df.low, df[["open", "close"]].min(axis=1))
    return df.assign(volume=1.0)


def test_gold_wick_signal_fills_next_open_with_2r_target():
    rows = [(100 + 0.1 * i, 100.6 + 0.1 * i, 99.6 + 0.1 * i, 100.1 + 0.1 * i) for i in range(70)]
    p = rows[-1]
    sig = (p[3], p[1] + 0.5, p[2] - 0.5, p[1] + 0.3)  # sweeps the previous low, closes above its high
    rows += [sig, (sig[3], sig[3] + 30, sig[3] - 0.1, sig[3] + 29)]
    m = _m15_from_h4(rows)
    tr = cb.run_bot5(m, params={"Use Daily EMA200 Trend Filter": False}, costs=cb.ZERO_COSTS)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t.entry_time == m.index[16 * 71]  # first M15 of the bar after the signal bar
    assert t.entry == pytest.approx(sig[3], abs=1e-6)
    assert t.stop_dist == pytest.approx(t.entry - m.low.iloc[16 * 70:16 * 71].min(), abs=1e-6)  # stop at signal low
    assert t.reason == "tp" and t.pnl_r == pytest.approx(2.0, abs=0.05)
