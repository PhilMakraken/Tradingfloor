"""ICC / session-reclaim blocks, Monte Carlo robustness gates, research notes, agents, live forward testing."""
import json
import lzma
import struct

import numpy as np
import pandas as pd
import pytest

from propfloor.data import dukascopy, yahoo
from propfloor.data.store import DataStore, pip_size
from propfloor.data.synthetic import make_bars
from propfloor.db import ResultsDB
from propfloor.evaluation import robustness as rb
from propfloor.evaluation.walkforward import Universe
from propfloor.factory.agents import AgentTeam
from propfloor.factory.notes import report_found, write_notes
from propfloor.live.forward import ForwardTester, longevity, verdict
from propfloor.strategies.blocks import ENTRIES, Ctx
from propfloor.strategies.genome import Component, StrategyGenome


def mk(closes, start="2024-01-02 00:00"):
    c = np.array(closes, float)
    pc = np.r_[c[0], c[:-1]]
    o = pc + 0.3 * (c - pc)
    idx = pd.date_range(start, periods=len(c), freq="1h", tz="UTC")
    return pd.DataFrame(dict(open=o, high=np.maximum(o, c) + 0.0002, low=np.minimum(o, c) - 0.0002, close=c,
                             volume=1.0), index=idx)


ICC_PATH = (list(1.100 + 0.0003 * np.sin(np.arange(20)))
            + [1.098, 1.096, 1.095, 1.097, 1.100, 1.103, 1.105, 1.103, 1.101, 1.100, 1.102, 1.104, 1.106, 1.109,
               1.112, 1.111, 1.110, 1.108, 1.107, 1.108, 1.110, 1.113, 1.115] + [1.115] * 10)
ICC_P = dict(k=2, min_retr=0.2, max_retr=0.8, max_bars=30, trig_n=2, min_leg_atr=0.5)


def test_icc_indication_correction_continuation():
    lo, sh = ENTRIES["icc"].fn(Ctx(mk(ICC_PATH), "EURUSD", "H1"), ICC_P)
    # bar 40: the 1.105 swing high broke (indication), price pulled back ~45% (correction) and closed above the
    # previous two bars' highs (continuation)
    assert 40 in np.where(lo)[0] and not sh.any()
    lo2, sh2 = ENTRIES["icc"].fn(Ctx(mk([2.2 - v for v in ICC_PATH]), "EURUSD", "H1"), ICC_P)
    assert (sh2 == lo).all() and not lo2.any()  # shorts mirror longs exactly
    # a correction must reach min_retr: demanding a 62% pullback removes both setups
    lo3, _ = ENTRIES["icc"].fn(Ctx(mk(ICC_PATH), "EURUSD", "H1"), {**ICC_P, "min_retr": 0.62, "max_retr": 0.65})
    assert not lo3.any()


PRE = list(1.105 + 0.002 * np.sin(np.arange(24)))  # previous day (ATR warm-up)
RANGE = [1.101, 1.108, 1.103, 1.109, 1.102, 1.107, 1.104]  # 00-06 UTC
RP = dict(range_start=0, range_end=7, latest=16, zone_a=0.70, zone_b=0.886, anchor="level", leg_n=8, min_ext_atr=0.3)


@pytest.mark.parametrize("pullback, fires", [(1.1115, True), (1.109, False), (1.115, False)])
def test_session_reclaim(pullback, fires):
    """07:00 breaks the range high, 08:00 extends to 1.120, 09:00 pulls back, 10:00 closes bullish back above the
    0.70 level. Fires only when the pullback reached the 0.70-0.886 zone without closing through 0.886."""
    day = PRE + RANGE + [1.112, 1.120, pullback, 1.1145, 1.118, 1.119] + [1.119] * 6
    lo, sh = ENTRIES["session_reclaim"].fn(Ctx(mk(day), "EURUSD", "H1"), RP)
    assert list(np.where(lo)[0]) == ([34] if fires else []) and not sh.any()
    lo2, sh2 = ENTRIES["session_reclaim"].fn(Ctx(mk([2.2 - v for v in day]), "EURUSD", "H1"), RP)
    assert (sh2 == lo).all() and not lo2.any()


def test_bootstrap_and_yearly():
    b = rb.bootstrap(np.full(50, 0.5))
    assert b["p_profit"] == 1.0 and b["maxdd_p95"] == 0.0
    assert rb.bootstrap(np.r_[np.full(20, -1.0), np.full(20, 0.9)])["p_profit"] < 0.5
    t = pd.Timestamp
    tr = pd.DataFrame({"exit_time": [t("2015-03-01", tz="UTC"), t("2016-03-01", tz="UTC"), t("2017-03-01", tz="UTC")],
                       "pnl_r": [1.0, -0.5, 2.0]})
    y = rb.yearly(tr, t("2014-01-01", tz="UTC"), t("2025-01-01", tz="UTC"))
    assert y["years"] > 10.9 and y["positive_years"] == pytest.approx(2 / 3) and y["by_year"][2016] == -0.5


def ma_genome(symbol="EURUSD"):
    return StrategyGenome(tf="H1", entry=Component("ma_cross", {"fast": 5, "slow": 20, "kind": "ema"}), filters=[],
                          exit={"sl_type": "atr", "sl_atr": 1.5, "tp_r": 2.0, "time_stop": 24, "trail_atr": 0.0,
                                "trail_type": "close", "breakeven_r": 0.0, "partial_r": 0.0, "exit_opposite": False,
                                "exit_mode": "none", "exit_n": 10, "exit_hour": 20},
                          sizing={"risk_per_trade": 0.005}, symbol=symbol)


def test_random_entry_and_neighbourhood_tests_run():
    from propfloor.data.clean import resample

    ctx = Ctx(resample(make_bars("2020-01-01", "2020-07-01", seed=3), "H1"), "EURUSD", "H1")
    r = rb.random_entries(ma_genome(), ctx, {}, n=20)
    assert 1 / 21 <= r["p_value"] <= 1.0
    nb = rb.neighbourhood(ma_genome(), ctx, {}, n=4)
    assert 0.0 <= nb["positive_share"] <= 1.0


def test_universe_skips_timeframes_finer_than_the_data():
    idx = pd.date_range("2010-01-01", periods=3000, freq="1D", tz="UTC")
    daily = pd.DataFrame({"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0}, index=idx)
    u = Universe({"AAPL": daily}, timeframes=["M15", "H1", "D1"])
    assert u.tfs_for("AAPL") == ["D1"]


def test_pip_sizes_for_new_markets():
    assert pip_size("EURUSD") == 0.0001 and pip_size("AAPL") == 0.01 and pip_size("WTI") == 0.01
    assert pip_size("XAGUSD") == 0.01 and pip_size("USDJPY") == 0.01


def test_yahoo_parse_and_dukascopy_ticks():
    js = {"chart": {"result": [{"timestamp": [1704153600, 1704240000],
                                "indicators": {"quote": [{"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5],
                                                          "close": [1.5, None], "volume": [10, 20]}]}}]}}
    df = yahoo.parse(js)
    assert len(df) == 1 and df["close"].iloc[0] == 1.5 and str(df.index.tz) == "UTC"
    assert yahoo.ticker("WTI") == "CL=F" and yahoo.ticker("aapl") == "AAPL"
    h = pd.Timestamp("2024-01-02 10:00", tz="UTC")
    recs = [(1_000, 110_010, 110_000, 1.0, 2.0), (30_000, 110_060, 110_050, 1.0, 1.0), (61_000, 109_990, 109_980, 1.0, 3.0)]
    raw = lzma.compress(b"".join(struct.pack(">IIIff", *r) for r in recs), format=lzma.FORMAT_ALONE)
    m1 = dukascopy.decode_ticks(raw, h, "EURUSD")
    assert len(m1) == 2 and m1["open"].iloc[0] == pytest.approx(1.1) and m1["high"].iloc[0] == pytest.approx(1.1005)
    assert m1["volume"].iloc[1] == 3.0


def test_verdict_and_longevity():
    assert verdict(np.full(10, 0.3), 0.3, 10.0) == "probation"
    assert verdict(np.full(25, 0.3) + np.r_[0.1, -0.1] .repeat(13)[:25], 0.3, 10.0) == "confirmed"
    assert verdict(np.full(25, -0.1), 0.3, 10.0) == "degraded"
    assert verdict(np.full(12, -1.0), 0.3, 5.0) == "failed"  # deeper than the Monte Carlo 95% drawdown
    assert longevity(0, 10, 0.0, 0.3, 0, 5) == 0.0
    assert longevity(50, 200, 0.45, 0.3, 1.0, 5.0) == 100.0
    assert 0 < longevity(10, 30, 0.3, 0.3, 1.0, 5.0) < 50


def test_forward_tester_counts_only_post_enrolment_trades(tmp_path):
    hist, live = DataStore(tmp_path / "hist"), DataStore(tmp_path / "live")
    bars = make_bars("2020-01-01", "2020-05-01", seed=5)
    hist.write("EURUSD", bars[bars.index < "2020-03-01"], source="dukascopy")
    db = ResultsDB(tmp_path / "r.sqlite")
    g = ma_genome()
    db.save_candidate(g.key(), g.to_dict(), g.describe(), 0, 1.0, {})
    db.execute("UPDATE strategies SET status='validated', validation=? WHERE id=?",
               (json.dumps({"oos": {"expectancy_r": 0.2}}), g.key()))
    fetched = []

    def fetch(sym, since):
        fetched.append(since)
        return bars[bars.index > since]

    ft = ForwardTester(db, hist, live, fetchers={"EURUSD": fetch}, found_path=str(tmp_path / "found.md"))
    ft.warmup_days = 10_000  # "now" is years after this fixture's history
    assert ft.enroll_validated() == 1
    db.execute("UPDATE forward_tests SET started='2020-03-15T00:00:00+00:00'")
    ft.refresh(["EURUSD"])
    assert live.meta("EURUSD")["end"] >= "2020-04-30" and hist.meta("EURUSD")["end"] < "2020-03-01"  # history untouched
    res = ft.evaluate()
    assert len(res) == 1 and res[0]["n_trades"] > 0
    full = g.backtest(Ctx(bars.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                                   "volume": "sum"}).dropna(), "EURUSD", "H1"))
    assert res[0]["n_trades"] == (full.entry_time >= "2020-03-15").sum()
    row = db.query("SELECT * FROM forward_tests")[0]
    assert row["status"] in ("probation", "confirmed", "degraded", "failed") and row["n_trades"] == res[0]["n_trades"]
    ft.refresh(["EURUSD"])  # incremental: resumes from the last stored live bar
    assert fetched[-1] == pd.Timestamp(live.meta("EURUSD")["end"])


def test_notes_and_found_report(tmp_path):
    db = ResultsDB(tmp_path / "r.sqlite")
    g = ma_genome()
    db.save_candidate(g.key(), g.to_dict(), g.describe(), 0, 0.4, {})
    db.save_validation(g.key(), False, ["IS trades 12 < 40", "OOS expectancy <= 0"],
                       {"is": {"n": 12, "expectancy_r": 0.1}, "oos": {"n": 3, "expectancy_r": -0.1}})
    p = write_notes(db, tmp_path / "notes.md")
    txt = p.read_text()
    assert "ma_cross | EURUSD | H1 | 1 | 0.400 | 0 |" in txt and len(txt) < 5000
    m = {"is": {"n": 80, "expectancy_r": 0.3}, "oos": {"n": 30, "expectancy_r": 0.2}, "dsr": 0.95,
         "yearly": {"years": 10.5, "positive_years": 0.8, "by_year": {2015: 3.0}},
         "robustness": {"bootstrap": {"p_profit": 0.99}, "random_entries": {"p_value": 0.01},
                        "neighbourhood": {"positive_share": 0.8}}, "challenge_oos": {"p_pass_all": 0.4}}
    report_found("abc", g.describe(), m, tmp_path / "found.md")
    assert "random-entry p=0.010" in (tmp_path / "found.md").read_text()


def test_agent_ideas_are_queued_for_the_factory(tmp_path, monkeypatch):
    team = AgentTeam(["EURUSD"], ["H1"], idea_dir=tmp_path / "ideas", notes_path=tmp_path / "none.md",
                     log_dir=tmp_path / "agents", backend="api")
    reply = 'Ideas: [{"symbol":"EURUSD","tf":"H1","entry":{"name":"icc","params":{"k":3}},"exit":{"sl_type":"swing"}},' \
            '{"entry":{"name":"not_a_block"}}]'
    monkeypatch.setattr(team, "ask", lambda role: reply)
    assert "session_reclaim" in team.context("price_action") and "ICC" in team.context("price_action")
    assert team.run_next() == 1
    files = list((tmp_path / "ideas").glob("*.json"))
    assert len(files) == 1 and files[0].name.startswith("technical_analyst-")
    item = json.loads(files[0].read_text())[0]
    assert item["entry"]["name"] == "icc" and item["exit"]["sl_type"] == "swing"
    assert (tmp_path / "agents" / "technical_analyst.md").read_text() == reply
