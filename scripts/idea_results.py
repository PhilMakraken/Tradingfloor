"""Read-only report of how queued (Claude-proposed) ideas scored.  usage: python scripts/idea_results.py [origin_prefix] [limit]"""
import json
import sqlite3
import sys

prefix = sys.argv[1] if len(sys.argv) > 1 else "claude:"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
c = sqlite3.connect("file:propfloor.sqlite?mode=ro", uri=True, timeout=30)
c.row_factory = sqlite3.Row
rows = c.execute("""SELECT origin, fitness, status, quick_info, validation, reasons, description FROM strategies
                    WHERE origin LIKE ? ORDER BY fitness DESC LIMIT ?""", (prefix + "%", limit)).fetchall()
n_all = c.execute("SELECT COUNT(*) FROM strategies WHERE origin LIKE ?", (prefix + "%",)).fetchone()[0]
trials = c.execute("SELECT value FROM meta WHERE key='n_trials'").fetchone()[0]
print(f"{n_all} scored ideas matching '{prefix}' (trials so far: {trials})")
print("fit | IS trades | IS expR | IS pass | IS stage1 | IS funded surv | IS payout | OOS expR | OOS pass | DSR | status | failed gates | rules")
for r in rows:
    q = json.loads(r["quick_info"] or "{}")
    v = json.loads(r["validation"] or "{}")
    isn = (q.get("is") or {})
    oos = v.get("oos") or {}
    f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    print(f"{r['fitness']:.2f} | {isn.get('n')} | {f(isn.get('expectancy_r'), 3)} | {f((q.get('challenge_is') or {}).get('p_pass_all'))}"
          f" | {f((q.get('challenge_is') or {}).get('p_pass_stage1'))} | {f((q.get('challenge_is') or {}).get('p_funded_survive'))}"
          f" | {f((q.get('challenge_is') or {}).get('expected_payout'), 0)}"
          f" | {f(oos.get('expectancy_r'), 3)} | {f((v.get('challenge_oos') or {}).get('p_pass_all'))} | {f(v.get('dsr'), 3)}"
          f" | {r['status']} | {'; '.join(json.loads(r['reasons'] or '[]'))[:90]} | [{r['origin']}] {r['description'][:110]}")
