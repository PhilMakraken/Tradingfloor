import json

import numpy as np
import pandas as pd

from propfloor.config import DEFAULT_PROGRAM, load_program
from propfloor.db import ResultsDB
from propfloor.floor.allocator import allocate, select_diversified
from propfloor.floor.broker import PaperBroker
from propfloor.floor.floor import TradingFloor
from tests.test_evaluation import mk


def _seed_db(db, ids, scores):
    for sid, sc in zip(ids, scores):
        db.save_candidate(sid, {"tf": "H1", "symbol": "EURUSD", "origin": "test", "entry": {"name": "x", "params": {}},
                                "filters": [], "exit": {}, "sizing": {}}, sid, 0, sc, {})
        m = {"is": {"expectancy_r": 0.2, "n": 100}, "oos": {"expectancy_r": 0.2, "n": 30},
             "challenge_is": {"p_pass_all": sc}, "challenge_oos": {"p_pass_all": sc}}
        db.save_validation(sid, True, [], m)


def test_select_diversified_skips_correlated():
    days = [f"2022-01-{d:02d}" for d in range(1, 29)]
    rng = np.random.default_rng(0)
    base = rng.normal(size=len(days))
    rets = {"a": dict(zip(days, base)), "b": dict(zip(days, base * 2 + 0.01)), "c": dict(zip(days, rng.normal(size=len(days))))}
    cands = [{"id": "a", "p_pass_oos": 0.5}, {"id": "b", "p_pass_oos": 0.4}, {"id": "c", "p_pass_oos": 0.3}]
    assert select_diversified(cands, rets, k=3) == ["a", "c"]
    baskets = allocate(cands, rets, n_accounts=3, per_account=1)
    assert baskets == [["a"], ["c"], ["a"]]


def test_floor_lifecycle(tmp_path):
    db = ResultsDB(tmp_path / "f.sqlite")
    _seed_db(db, ["good", "bad"], [0.6, 0.5])
    prog = load_program(DEFAULT_PROGRAM)
    floor = TradingFloor(prog, db, PaperBroker())
    floor.accounts.clear()
    from propfloor.floor.floor import FloorAccount

    floor.accounts["A"] = FloorAccount("A", prog, ["good"])
    floor.accounts["B"] = FloorAccount("B", prog, ["gap"])
    floor.accounts["C"] = FloorAccount("C", prog, ["loser"])
    trades = {"good": mk([2.0] * 300).assign(entry=1.1, exit=1.1),
              "gap": mk([1.0] * 300, mae=-40.0).assign(entry=1.1, exit=1.1),  # floating loss far beyond the stop
              "loser": mk([-1.0] * 300).assign(entry=1.1, exit=1.1)}
    floor.run(trades)
    s = floor.summary().set_index("id")
    assert s.loc["A", "stage"] == "funded" and s.loc["A", "payouts"] > 0
    assert s.loc["B", "stage"] == "violated"
    # plain losers are throttled by the risk manager instead of breaching
    assert s.loc["C", "stage"] == "stage1" and s.loc["C", "balance"] < 100_000 and s.loc["C", "trailing_buffer"] > 0
    rows = {r["id"]: r for r in db.query("SELECT * FROM floor_accounts")}
    assert rows["B"]["status"] == "violated" and json.loads(rows["B"]["violation"])["rule"] in ("max_drawdown", "daily_drawdown")
    kinds = [r["kind"] for r in db.query("SELECT kind FROM floor_events WHERE account_id='A'")]
    assert kinds.count("stage") == 2
    assert len(floor.broker.fills) > 0 and not floor.broker.positions("A")


def test_open_accounts_from_db(tmp_path):
    db = ResultsDB(tmp_path / "g.sqlite")
    _seed_db(db, ["s1", "s2", "s3"], [0.5, 0.4, 0.3])
    fl = TradingFloor(load_program(DEFAULT_PROGRAM), db)
    accts = fl.open_accounts(4, per_account=2)
    assert len(accts) == 4 and all(len(a.strategies) == 2 for a in accts)
