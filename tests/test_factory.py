import json

import numpy as np

from propfloor.config import DEFAULT_PROGRAM, load_program
from propfloor.data.synthetic import make_bars
from propfloor.db import ResultsDB
from propfloor.evaluation.walkforward import Evaluator, Gates, Universe
from propfloor.factory.engine import Factory
from propfloor.factory.llm import coerce, parse_genomes
from propfloor.strategies.genome import StrategyGenome


def make_factory(tmp_path, **kw):
    data = {"EURUSD": make_bars("2019-01-01", "2020-01-01", seed=21)}
    u = Universe(data, timeframes=["H1", "H4"], synthetic={"EURUSD": True})
    ev = Evaluator(u, load_program(DEFAULT_PROGRAM), Gates(min_trades_is=10), quick_sims=4, full_sims=4)
    return Factory(ev, ResultsDB(tmp_path / "r.sqlite"), seed=1, pop_size=8, **kw)


def test_factory_runs_logs_and_resumes(tmp_path):
    f = make_factory(tmp_path, candidate_threshold=-0.5)
    f.run(max_generations=2)
    db = f.db
    n = db.n_trials
    assert n >= 8 and len(db.top(100)) == n
    assert db.query("SELECT generations FROM runs")[0]["generations"] == 2
    # candidates that crossed the threshold were validated - and synthetic data never validates
    assert not db.validated()
    # resume: generation counter persists, no duplicate evaluations
    f2 = Factory(f.ev, ResultsDB(tmp_path / "r.sqlite"), seed=2, pop_size=8)
    assert f2.generation == 2
    f2.step()
    ids = [r["id"] for r in db.query("SELECT id FROM strategies")]
    assert len(ids) == len(set(ids))


def test_llm_parsing_is_defensive():
    text = 'Here: [{"symbol":"EURUSD","tf":"H1","entry":{"name":"pin_bar","params":{"wick_ratio":99}},' \
           '"filters":[{"name":"session","params":{"start":7}},{"name":"bogus"}],"exit":{"sl_atr":2},' \
           '"sizing":{"risk_per_trade":0.005},"direction":"long"}, {"entry":{"name":"nope"}}]'
    gs = parse_genomes(text)
    assert len(gs) == 1
    g = gs[0]
    assert g.entry.params["wick_ratio"] == 4.0 and [c.name for c in g.filters] == ["session"]
    assert set(g.exit) >= {"sl_type", "tp_r", "time_stop"} and g.origin == "llm"
    assert parse_genomes("no json") == []


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def propose(self, n, *a):
        self.calls += 1
        return [coerce({"entry": {"name": "donchian_breakout", "params": {"n": 20}}, "tf": "H4"})]

    def critique(self, f):
        return "ok"


def test_factory_uses_llm(tmp_path):
    llm = FakeLLM()
    f = make_factory(tmp_path, llm=llm, llm_every=1)
    f.step()
    assert llm.calls == 1
    assert any(json.loads(r["genome"])["origin"] == "llm" for r in f.db.top(100))


def test_elites_are_diverse(tmp_path):
    f = make_factory(tmp_path)
    g = f.population()[0]
    for i in range(6):  # six near-copies of one entry block with top fitness
        c = StrategyGenome.from_dict(g.to_dict())
        c.exit["sl_atr"] = 1.0 + i / 10
        f.db.save_candidate(c.key(), c.to_dict(), c.describe(), 0, 1.0 - i / 100, {})
    el = f._elites()
    assert sum(e.entry.name == g.entry.name for e in el) <= 2
