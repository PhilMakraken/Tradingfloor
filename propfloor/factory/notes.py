"""Small human/agent-readable research notes, regenerated from the results DB.

data/RESEARCH_NOTES.md   - one line per (entry block, symbol, timeframe) niche that has been tried: how often, best
                           fitness, how many validated, the most common rejection reason. Aggregated so the file
                           stays a few tens of KB however long the factory runs; research agents read it to avoid
                           re-proposing what already failed.
data/FOUND_STRATEGIES.md - append-only report of every strategy that passed every gate (and forward-test verdicts).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from propfloor.db import ResultsDB, now

NOTES = Path("data/RESEARCH_NOTES.md")
FOUND = Path("data/FOUND_STRATEGIES.md")


def _reason_key(r: str) -> str:
    """Strip the numbers so 'IS trades 12 < 40' and 'IS trades 30 < 40' count as one reason."""
    return "".join(ch for ch in r if not (ch.isdigit() or ch == ".")).split("<")[0].strip()[:50]


def write_notes(db: ResultsDB, path: str | Path = NOTES, max_rows: int = 150) -> Path:
    rows = db.query(
        "SELECT json_extract(genome,'$.entry.name') e, symbol, tf, COUNT(*) n, MAX(fitness) best, "
        "SUM(status='validated') v FROM strategies GROUP BY e, symbol, tf ORDER BY best DESC")
    reasons: dict[tuple, Counter] = defaultdict(Counter)
    for r in db.query("SELECT json_extract(genome,'$.entry.name') e, symbol, tf, reasons FROM strategies "
                      "WHERE status='rejected'"):
        for x in json.loads(r["reasons"] or "[]"):
            reasons[(r["e"], r["symbol"], r["tf"])][_reason_key(x)] += 1
    fams = Counter()
    for r in rows:
        fams[r["e"]] += r["n"]
    lines = [f"# Research notes (auto-generated {now()})", "",
             f"trials: {db.n_trials} | generation: {db.get_meta('generation', 0)} | niches tried: {len(rows)} | "
             f"validated: {len(db.validated())}", "",
             "## Most promising niches (entry | symbol | tf | tries | best fitness | validated | top rejection)"]
    for r in rows[:max_rows]:
        top = reasons.get((r["e"], r["symbol"], r["tf"]))
        why = top.most_common(1)[0][0] if top else "-"
        lines.append(f"{r['e']} | {r['symbol']} | {r['tf']} | {r['n']} | {r['best']:.3f} | {r['v']} | {why}")
    dead = [e for e, n in fams.items() if n >= 200 and not any(r["e"] == e and r["best"] > 0 for r in rows)]
    if dead:
        lines += ["", "## Exhausted (>=200 tries, never positive fitness)", ", ".join(sorted(dead))]
    fwd = db.query("SELECT id, status, n_trades, expectancy, longevity FROM forward_tests ORDER BY longevity DESC LIMIT 30")
    if fwd:
        lines += ["", "## Live forward tests (id | status | trades | expectancy R | longevity)"]
        lines += [f"{f['id']} | {f['status']} | {f['n_trades']} | {f['expectancy']:.3f} | {f['longevity']:.2f}" for f in fwd]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


def report_found(sid: str, description: str, m: dict, path: str | Path = FOUND) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    y = m.get("yearly", {})
    rb = m.get("robustness", {})
    co = m.get("challenge_oos", {})
    txt = (f"\n## {now()}  {sid}\n{description}\n\n"
           f"- years tested: {y.get('years')}  profitable years: {y.get('positive_years', 0):.0%}  R by year: {y.get('by_year')}\n"
           f"- IS: {m['is']['n']} trades, {m['is']['expectancy_r']:.3f} R/trade | OOS: {m['oos']['n']} trades, "
           f"{m['oos']['expectancy_r']:.3f} R/trade | deflated Sharpe {m.get('dsr', 0):.2f}\n"
           f"- Monte Carlo: P(profit) {rb.get('bootstrap', {}).get('p_profit', 0):.2f}, "
           f"random-entry p={rb.get('random_entries', {}).get('p_value', 1):.3f}, "
           f"parameter neighbours positive {rb.get('neighbourhood', {}).get('positive_share', 0):.0%}\n"
           f"- prop challenge OOS P(pass both stages) {co.get('p_pass_all', 0):.2f}\n"
           f"- status: backtest-validated, now in live forward test (`python -m propfloor live status`)\n")
    if not p.exists():
        txt = "# Strategies that passed every gate\n\nBacktests are not guarantees; see forward-test status.\n" + txt
    with p.open("a") as fh:
        fh.write(txt)


def report_forward(sid: str, status: str, detail: dict, path: str | Path = FOUND) -> None:
    with Path(path).open("a") as fh:
        fh.write(f"\n- {now()} forward test {sid}: **{status}** {json.dumps(detail)}\n")
