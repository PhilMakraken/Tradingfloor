"""SQLite results database (strategies, trials, validation, floor state)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY,
    genome TEXT NOT NULL,
    description TEXT,
    origin TEXT,
    symbol TEXT,
    tf TEXT,
    generation INTEGER,
    created_at TEXT,
    fitness REAL,
    quick_info TEXT,
    status TEXT DEFAULT 'candidate',     -- candidate | validated | rejected
    validation TEXT,
    reasons TEXT,
    p_pass_is REAL, p_pass_oos REAL, dsr REAL,
    exp_is REAL, exp_oos REAL, n_is INTEGER, n_oos INTEGER,
    funded_survive REAL, expected_payout REAL,
    validated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_strat_fit ON strategies(fitness DESC);
CREATE INDEX IF NOT EXISTS ix_strat_status ON strategies(status);
CREATE TABLE IF NOT EXISTS strategy_returns (
    id TEXT PRIMARY KEY, daily TEXT  -- json {date: R}
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started TEXT, ended TEXT, generations INTEGER, evaluated INTEGER, notes TEXT
);
CREATE TABLE IF NOT EXISTS llm_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, kind TEXT, prompt TEXT, response TEXT
);
CREATE TABLE IF NOT EXISTS floor_accounts (
    id TEXT PRIMARY KEY, program TEXT, stage TEXT, status TEXT, strategies TEXT,
    start_balance REAL, balance REAL, equity REAL, floor REAL, daily_floor REAL,
    payouts REAL DEFAULT 0, violation TEXT, created_at TEXT, updated_at TEXT, state TEXT
);
CREATE TABLE IF NOT EXISTS forward_tests (
    id TEXT PRIMARY KEY, started TEXT, updated TEXT, status TEXT,  -- probation | confirmed | degraded | failed
    n_trades INTEGER DEFAULT 0, expectancy REAL DEFAULT 0, total_r REAL DEFAULT 0, max_dd_r REAL DEFAULT 0,
    longevity REAL DEFAULT 0, bt_expectancy REAL, bt_maxdd_p95 REAL, train_end TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS floor_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, account_id TEXT, kind TEXT, detail TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResultsDB:
    def __init__(self, path: str | Path = "propfloor.sqlite") -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def execute(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    def query(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # ------------------------------------------------------------- meta / trials
    def get_meta(self, key: str, default: Any = None) -> Any:
        r = self.query("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(r[0]["value"]) if r else default

    def set_meta(self, key: str, value: Any) -> None:
        self.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value)))

    def add_trials(self, n: int = 1) -> int:
        """Atomic increment (several processes may share the database)."""
        with self._lock:
            self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('n_trials', '0')")
            self.conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + ? WHERE key = 'n_trials'", (n,))
            self.conn.commit()
            return int(self.conn.execute("SELECT value FROM meta WHERE key='n_trials'").fetchone()[0])

    @property
    def n_trials(self) -> int:
        return int(self.get_meta("n_trials", 0))

    # ------------------------------------------------------------- strategies
    def has(self, sid: str) -> bool:
        return bool(self.query("SELECT 1 FROM strategies WHERE id=?", (sid,)))

    def save_candidate(self, sid: str, genome: dict, description: str, generation: int, fitness: float, info: dict) -> None:
        self.execute(
            "INSERT OR IGNORE INTO strategies(id, genome, description, origin, symbol, tf, generation, created_at, fitness, quick_info)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sid, json.dumps(genome), description, genome.get("origin"), genome.get("symbol"), genome.get("tf"),
             generation, now(), fitness, json.dumps(info, default=float)))

    def save_validation(self, sid: str, validated: bool, reasons: list[str], m: dict) -> None:
        ci, co = m.get("challenge_is", {}), m.get("challenge_oos", {})
        self.execute(
            "UPDATE strategies SET status=?, validation=?, reasons=?, p_pass_is=?, p_pass_oos=?, dsr=?, exp_is=?, exp_oos=?,"
            " n_is=?, n_oos=?, funded_survive=?, expected_payout=?, validated_at=? WHERE id=?",
            ("validated" if validated else "rejected", json.dumps(m, default=_jsonable), json.dumps(reasons),
             ci.get("p_pass_all"), co.get("p_pass_all"), m.get("dsr"), m["is"]["expectancy_r"], m["oos"]["expectancy_r"],
             m["is"]["n"], m["oos"]["n"], co.get("p_funded_survive"), co.get("expected_payout"), now(), sid))

    def save_returns(self, sid: str, daily: dict[str, float]) -> None:
        self.execute("INSERT OR REPLACE INTO strategy_returns(id, daily) VALUES (?, ?)", (sid, json.dumps(daily)))

    def returns(self, sid: str) -> dict[str, float]:
        r = self.query("SELECT daily FROM strategy_returns WHERE id=?", (sid,))
        return json.loads(r[0]["daily"]) if r else {}

    def top(self, n: int = 20, status: str | None = None) -> list[dict]:
        if status:
            return self.query("SELECT * FROM strategies WHERE status=? ORDER BY fitness DESC LIMIT ?", (status, n))
        return self.query("SELECT * FROM strategies ORDER BY fitness DESC LIMIT ?", (n,))

    def validated(self) -> list[dict]:
        return self.query("SELECT * FROM strategies WHERE status='validated' ORDER BY p_pass_oos DESC")

    def log_llm(self, kind: str, prompt: str, response: str) -> None:
        self.execute("INSERT INTO llm_log(time, kind, prompt, response) VALUES (?,?,?,?)", (now(), kind, prompt, response))

    def log_event(self, account_id: str, kind: str, detail: dict, time: str | None = None) -> None:
        self.execute("INSERT INTO floor_events(time, account_id, kind, detail) VALUES (?,?,?,?)",
                     (time or now(), account_id, kind, json.dumps(detail, default=str)))


def _jsonable(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)
