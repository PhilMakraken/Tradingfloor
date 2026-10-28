"""Live forward testing ("paper trading on real-time data") of strategies that passed every backtest gate.

The research history (data/parquet) is never touched. Fresh real bars are fetched into a separate live store
(data/live): Dukascopy hourly tick files for FX/metals (same BID feed as the history), Yahoo for symbols whose
history came from Yahoo. Each poll re-runs the strategy's unchanged rules over history + live bars and keeps only
trades ENTERED after the strategy was enrolled, i.e. on data that did not exist when it was found.

Status:  probation -> confirmed  (>= min_trades forward trades, expectancy > 0 with ~90% confidence and at least
                                  half the backtest expectancy)
                   -> degraded   (>= min_trades and forward expectancy <= 0)
                   -> failed     (forward drawdown deeper than the 95th percentile Monte Carlo backtest drawdown)
Longevity score (0-100) = agreement with the backtest x sqrt(trade-count maturity x time maturity) x drawdown health.
"""
from __future__ import annotations

import json
import logging
from typing import Callable

import numpy as np
import pandas as pd

from propfloor.data.clean import resample
from propfloor.data.store import DataStore
from propfloor.db import ResultsDB, now
from propfloor.evaluation.stats import max_drawdown_r
from propfloor.strategies.blocks import Ctx
from propfloor.strategies.genome import StrategyGenome

log = logging.getLogger(__name__)

WARMUP_DAYS = 120  # live bars fetched before enrolment when the history is older (indicator warm-up)
Fetcher = Callable[[str, pd.Timestamp], pd.DataFrame]


def default_fetcher(symbol: str, sources: list[str]) -> Fetcher:
    if "yahoo" in sources:
        from propfloor.data import yahoo
        return lambda sym, since: yahoo.recent(sym, since, tf="D1")
    from propfloor.data import dukascopy
    return dukascopy.recent


def longevity(n: int, days: float, exp: float, bt_exp: float, dd: float, bt_dd95: float) -> float:
    if n == 0 or bt_exp <= 0:
        return 0.0
    agree = float(np.clip(exp / bt_exp, 0.0, 1.5)) / 1.5
    mature = np.sqrt(min(1.0, n / 50) * min(1.0, days / 180))
    health = 1.0 if not bt_dd95 or dd <= bt_dd95 else 0.5
    return round(100 * agree * mature * health, 1)


def verdict(r: np.ndarray, bt_exp: float, bt_dd95: float, min_trades: int = 20) -> str:
    n = len(r)
    dd = max_drawdown_r(r) if n else 0.0
    if bt_dd95 and dd > bt_dd95:
        return "failed"
    if n < min_trades:
        return "probation"
    exp, sd = float(r.mean()), float(r.std(ddof=1)) if n > 1 else 0.0
    if exp <= 0:
        return "degraded"
    if exp - 1.28 * sd / np.sqrt(n) > 0 and exp >= 0.5 * bt_exp:
        return "confirmed"
    return "probation"


class ForwardTester:
    def __init__(self, db: ResultsDB, history: DataStore, live: DataStore | None = None, costs: dict | None = None,
                 fetchers: dict[str, Fetcher] | None = None, min_trades: int = 20,
                 found_path: str = "data/FOUND_STRATEGIES.md") -> None:
        self.db = db
        self.history = history
        self.live = live or DataStore("data/live")
        self.costs = costs or {}
        self.fetchers = fetchers or {}
        self.min_trades = min_trades
        self.found_path = found_path
        self.warmup_days = WARMUP_DAYS

    # ------------------------------------------------------------------ enrolment
    def enroll(self, sid: str, metrics: dict, train_end: pd.Timestamp | None) -> None:
        bt = metrics.get("robustness", {}).get("bootstrap", {})
        self.db.execute(
            "INSERT OR IGNORE INTO forward_tests(id, started, updated, status, bt_expectancy, bt_maxdd_p95, train_end)"
            " VALUES (?,?,?,?,?,?,?)",
            (sid, now(), now(), "probation", metrics.get("oos", {}).get("expectancy_r", 0.0), bt.get("maxdd_p95"),
             str(train_end) if train_end is not None else None))

    def enroll_validated(self) -> int:
        """Enrol every validated strategy not yet forward-tested (e.g. found before live testing existed)."""
        k = 0
        for r in self.db.query("SELECT s.id, s.validation FROM strategies s LEFT JOIN forward_tests f ON f.id = s.id "
                               "WHERE s.status='validated' AND f.id IS NULL"):
            self.enroll(r["id"], json.loads(r["validation"] or "{}"), None)
            k += 1
        return k

    # ------------------------------------------------------------------ data
    def refresh(self, symbols: list[str]) -> dict[str, int]:
        out = {}
        for sym in symbols:
            sources = self.history.meta(sym).get("sources", []) if self.history.path(sym).exists() else []
            fetch = self.fetchers.get(sym) or default_fetcher(sym, sources)
            since = self._last_bar(sym)
            if since is None:  # first poll: continue the history, or warm indicators up when it is old
                since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=self.warmup_days)
                if self.history.path(sym).exists():
                    since = max(since, pd.Timestamp(self.history.meta(sym)["end"]))
            try:
                df = fetch(sym, since)
            except Exception as e:  # noqa: BLE001 - one dead feed must not stop the others
                log.warning("live fetch %s failed: %s", sym, e)
                continue
            if len(df):
                self.live.write(sym, df, source="live", prefer_new=True)
            out[sym] = len(df)
        return out

    def _last_bar(self, sym: str) -> pd.Timestamp | None:
        if self.live.path(sym).exists():
            return pd.Timestamp(self.live.meta(sym)["end"])
        return None

    def bars(self, sym: str) -> pd.DataFrame:
        parts = []
        if self.history.path(sym).exists():
            parts.append(self.history.load(sym))
        if self.live.path(sym).exists():
            lv = self.live.load(sym)
            if parts:
                lv = lv[lv.index > parts[0].index[-1]]  # history wins where both exist
            parts.append(lv)
        return pd.concat(parts) if parts else pd.DataFrame()

    # ------------------------------------------------------------------ evaluation
    def _ctx(self, sym: str, tf: str, train_end, cache: dict) -> Ctx | None:
        key = (sym, tf, train_end)  # train_mask is memoised inside Ctx, so one Ctx per in-sample cut
        if key not in cache:
            if (sym, tf, None) not in cache:
                m1 = self.bars(sym)
                cache[(sym, tf, None)] = m1 if tf == "M1" or m1.empty else resample(m1, tf)
            if cache[(sym, tf, None)].empty:
                return None
            cache[key] = Ctx(cache[(sym, tf, None)], sym, tf)
            cache[key].train_end = pd.Timestamp(train_end) if train_end else None
        return cache[key]

    def evaluate(self, sids: list[str] | None = None) -> list[dict]:
        rows = self.db.query("SELECT f.*, s.genome FROM forward_tests f JOIN strategies s ON s.id = f.id")
        if sids is not None:
            rows = [r for r in rows if r["id"] in sids]
        cache: dict = {}  # (sym, tf, None) -> bars, (sym, tf, train_end) -> Ctx
        out = []
        for r in rows:
            g = StrategyGenome.from_dict(json.loads(r["genome"]))
            if r["status"] == "failed":
                continue
            ctx = self._ctx(g.symbol, g.tf, r["train_end"], cache)
            if ctx is None:
                continue
            peers = {s: self._ctx(s, g.tf, r["train_end"], cache) for s in self.history.symbols() if s != g.symbol}
            ctx.peers = {s: p for s, p in peers.items() if p is not None}
            started = pd.Timestamp(r["started"])
            tr = g.backtest(ctx, **self.costs)
            fwd = tr[tr.entry_time >= started]
            pnl = fwd["pnl_r"].to_numpy()
            days = (pd.Timestamp.now(tz="UTC") - started).total_seconds() / 86400
            dd = max_drawdown_r(pnl) if len(pnl) else 0.0
            exp = float(pnl.mean()) if len(pnl) else 0.0
            status = verdict(pnl, r["bt_expectancy"] or 0.0, r["bt_maxdd_p95"] or 0.0, self.min_trades)
            score = longevity(len(pnl), days, exp, r["bt_expectancy"] or 0.0, dd, r["bt_maxdd_p95"] or 0.0)
            last_bar = str(ctx.time[-1])
            self.db.execute("UPDATE forward_tests SET updated=?, status=?, n_trades=?, expectancy=?, total_r=?, "
                            "max_dd_r=?, longevity=?, detail=? WHERE id=?",
                            (now(), status, len(pnl), exp, float(pnl.sum()), dd, score,
                             json.dumps({"last_bar": last_bar, "days": round(days, 1)}), r["id"]))
            res = {"id": r["id"], "status": status, "n_trades": len(pnl), "expectancy": round(exp, 3),
                   "total_r": round(float(pnl.sum()), 2), "max_dd_r": round(dd, 2), "longevity": score,
                   "last_bar": last_bar}
            if status != r["status"]:
                from propfloor.factory.notes import report_forward
                report_forward(r["id"], status, res, self.found_path)
                log.info("FORWARD %s: %s -> %s %s", r["id"], r["status"], status, res)
            out.append(res)
        return out

    def run_once(self) -> list[dict]:
        self.enroll_validated()
        if not self.db.query("SELECT 1 FROM forward_tests WHERE status != 'failed' LIMIT 1"):
            return []
        self.refresh(self.history.symbols())  # all of them: intermarket blocks read their peers' live bars too
        return self.evaluate()
