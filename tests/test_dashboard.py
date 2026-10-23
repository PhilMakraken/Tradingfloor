import sys

import pytest

from propfloor.db import ResultsDB


def test_dashboard_renders(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    db = tmp_path / "d.sqlite"
    ResultsDB(db).add_trials(3)
    monkeypatch.setattr(sys, "argv", ["app", "--db", str(db)])
    at = testing.AppTest.from_file("propfloor/dashboard/app.py", default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].value == "3"
