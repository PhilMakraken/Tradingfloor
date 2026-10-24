"""Unattended strategy factory: random search + genetic evolution (+ optional Claude proposals).

Each generation:
  1. population = elites (best IS fitness so far) + mutations + crossovers + fresh randoms (+ LLM ideas)
  2. unseen genomes get a quick in-sample evaluation (every one counts as a trial for the DSR)
  3. genomes above the candidate threshold get full validation (walk-forward, OOS, DSR, cross-symbol)
Everything is persisted to SQLite so the factory can be stopped and resumed at any time.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from propfloor.db import ResultsDB, now
from propfloor.evaluation.walkforward import Evaluator
from propfloor.strategies.genome import StrategyGenome, crossover, mutate, random_genome

log = logging.getLogger(__name__)


class Factory:
    def __init__(self, evaluator: Evaluator, db: ResultsDB, seed: int | None = None, pop_size: int = 40,
                 elite: int = 12, random_frac: float = 0.5, candidate_threshold: float = 0.15, per_family: int = 2,
                 llm=None, llm_every: int = 10, llm_n: int = 8,
                 idea_dir: str | None = "data/ideas") -> None:
        self.ev = evaluator
        self.db = db
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.elite = elite
        self.random_frac = random_frac
        self.per_family = per_family
        self.threshold = candidate_threshold
        self.llm = llm
        self.llm_every = llm_every
        self.llm_n = llm_n
        self.generation = int(db.get_meta("generation", 0))
        self.idea_dir: Path | None = Path(idea_dir) if idea_dir else None
        self._pending_files: list[Path] = []
        self.symbols = evaluator.u.symbols
        self.tfs = evaluator.u.timeframes

    # ------------------------------------------------------------------ population
    def _elites(self) -> list[StrategyGenome]:
        """Best genomes, at most `per_family` per entry block (and one per entry+symbol+tf niche) so the
        search can't collapse onto near-copies of a single idea."""
        # best row per (entry block, symbol, timeframe) niche over the whole history, best niches first
        rows = self.db.query(
            "SELECT genome FROM (SELECT genome, fitness, ROW_NUMBER() OVER (PARTITION BY "
            "json_extract(genome,'$.entry.name'), symbol, tf ORDER BY fitness DESC) rn FROM strategies "
            "WHERE fitness > -5) WHERE rn = 1 ORDER BY fitness DESC LIMIT ?", (self.elite * 20,))
        out, fam = [], {}
        for r in rows:
            g = StrategyGenome.from_dict(json.loads(r["genome"]))
            if fam.get(g.entry.name, 0) >= self.per_family:
                continue
            fam[g.entry.name] = fam.get(g.entry.name, 0) + 1
            out.append(g)
            if len(out) >= self.elite:
                break
        return out

    def _tournament(self, elites):
        k = min(3, len(elites))
        idx = self.rng.choice(len(elites), size=k, replace=False)
        return elites[int(min(idx))]  # elites are sorted by fitness

    def population(self) -> list[StrategyGenome]:
        elites = self._elites()
        pop: list[StrategyGenome] = []
        n_rand = self.pop_size if len(elites) < 3 else int(self.pop_size * self.random_frac)
        pop += [random_genome(self.rng, self.tfs, symbols=self.symbols) for _ in range(n_rand)]
        while len(pop) < self.pop_size and elites:
            if self.rng.random() < 0.7:
                pop.append(mutate(self._tournament(elites), self.rng, self.tfs, self.symbols))
            else:
                pop.append(crossover(self._tournament(elites), self._tournament(elites), self.rng))
        if self.llm is not None and self.generation % self.llm_every == 0:
            try:
                pop += self.llm.propose(self.llm_n, self._leaderboard(), self._failures(), self.symbols, self.tfs)
            except Exception as e:  # noqa: BLE001
                log.warning("LLM proposal failed: %s", e)
        pop += self._queued_ideas()
        return pop

    def _queued_ideas(self, max_files: int = 20) -> list[StrategyGenome]:
        """Strategy ideas dropped as JSON arrays into `idea_dir` (e.g. by Claude research agents).
        Each file is consumed once (moved to idea_dir/done/); malformed items are skipped."""
        from propfloor.factory.llm import coerce

        if self.idea_dir is None or not self.idea_dir.exists():
            return []
        out: list[StrategyGenome] = []
        done = self.idea_dir / "done"
        done.mkdir(exist_ok=True)
        for f in sorted(self.idea_dir.glob("*.json"))[:max_files]:
            try:
                items = json.loads(f.read_text())
                tag = f.stem
                for it in items if isinstance(items, list) else []:
                    try:
                        g = coerce(it)
                    except Exception:  # noqa: BLE001
                        g = None
                    if g is not None:
                        g.origin = f"claude:{tag}"[:60]
                        out.append(g)
            except Exception as e:  # noqa: BLE001
                log.warning("bad idea file %s: %s", f.name, e)
            self._pending_files.append(f)  # moved to done/ only after the generation is scored (survives restarts)
        if out:
            log.info("loaded %d queued strategy ideas", len(out))
        return out

    def _leaderboard(self) -> list[dict]:
        return [{"desc": r["description"], "fitness": round(r["fitness"], 3), "status": r["status"]} for r in self.db.top(15)]

    def _failures(self) -> list[dict]:
        rows = self.db.query("SELECT description, reasons FROM strategies WHERE status='rejected' ORDER BY validated_at DESC LIMIT 15")
        return [{"desc": r["description"], "reasons": json.loads(r["reasons"] or "[]")} for r in rows]

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, g: StrategyGenome) -> float | None:
        sid = g.key()
        if self.db.has(sid):
            return None
        g.symbol = g.symbol if g.symbol in self.symbols else self.symbols[0]
        g.tf = g.tf if g.tf in self.tfs else self.tfs[0]
        try:
            fit, info = self.ev.quick(g)
        except Exception as e:  # noqa: BLE001 - a broken genome must never stop the factory
            log.debug("genome %s failed: %s", sid, e)
            fit, info = -10.0, {"error": str(e)}
        self.db.add_trials(1)
        self.db.save_candidate(sid, g.to_dict(), g.describe(), self.generation, float(fit), info)
        if fit >= self.threshold:
            self.validate(g)
        return fit

    def validate(self, g: StrategyGenome) -> bool:
        sid = g.key()
        res = self.ev.validate(g, n_trials=max(self.db.n_trials, 2))
        self.db.save_validation(sid, res.validated, res.reasons, res.metrics)
        if res.validated:
            tr = self.ev.trades(g)
            daily = tr.groupby(tr.exit_time.dt.strftime("%Y-%m-%d"))["pnl_r"].sum().to_dict()
            self.db.save_returns(sid, {k: float(v) for k, v in daily.items()})
            log.info("VALIDATED %s  OOS P(pass)=%.2f  %s", sid, res.metrics["challenge_oos"]["p_pass_all"], g.describe())
        return res.validated

    def step(self) -> dict:
        t0 = time.time()
        self._pending_files = []
        pop = self.population()
        fits = [f for g in pop if (f := self.evaluate(g)) is not None]
        if self._pending_files:
            done = self.idea_dir / "done"
            for f in self._pending_files:
                if f.exists():
                    f.rename(done / f.name)
        self.generation += 1
        self.db.set_meta("generation", self.generation)
        stats = {"generation": self.generation, "evaluated": len(fits), "best": max(fits, default=None),
                 "trials": self.db.n_trials, "validated": len(self.db.validated()), "secs": round(time.time() - t0, 1)}
        log.info("gen %(generation)d: evaluated %(evaluated)d best=%(best)s trials=%(trials)d validated=%(validated)d (%(secs)ss)", stats)
        return stats

    def run(self, max_generations: int | None = None, max_hours: float | None = None) -> None:
        """Runs indefinitely unless limits are given."""
        run_id = self.db.execute("INSERT INTO runs(started, generations, evaluated) VALUES (?,0,0)", (now(),)).lastrowid
        start = time.time()
        gens = evaluated = 0
        try:
            while True:
                if max_generations is not None and gens >= max_generations:
                    break
                if max_hours is not None and time.time() - start > max_hours * 3600:
                    break
                try:
                    s = self.step()
                except Exception:  # noqa: BLE001
                    log.exception("generation failed; continuing")
                    continue
                gens += 1
                evaluated += s["evaluated"]
                if self.llm is not None and gens % (self.llm_every * 5) == 0:
                    try:
                        self.db.set_meta("llm_critique", self.llm.critique(self._failures()))
                    except Exception as e:  # noqa: BLE001
                        log.warning("LLM critique failed: %s", e)
        finally:
            self.db.execute("UPDATE runs SET ended=?, generations=?, evaluated=? WHERE id=?", (now(), gens, evaluated, run_id))
