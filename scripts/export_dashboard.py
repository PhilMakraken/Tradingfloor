"""Export factory results from propfloor.sqlite into small JSON files for the web dashboard."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main(db_path: str = "propfloor.sqlite", out: str = "dashboard_export") -> None:
    out_dir = Path(out)
    out_dir.mkdir(exist_ok=True)
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    q = lambda sql, a=(): [dict(r) for r in c.execute(sql, a)]  # noqa: E731
    meta = {r["key"]: json.loads(r["value"]) for r in q("SELECT key, value FROM meta")}
    status = {r["status"]: r["n"] for r in q("SELECT status, COUNT(*) n FROM strategies GROUP BY status")}
    summary = [{
        "generation": int(meta.get("generation", 0)),
        "trials": int(meta.get("n_trials", 0)),
        "fully_checked": status.get("validated", 0) + status.get("rejected", 0),
        "validated": status.get("validated", 0),
        "best_fitness": q("SELECT MAX(fitness) m FROM strategies")[0]["m"],
        "best_holdout_pass": q("SELECT MAX(p_pass_oos) m FROM strategies")[0]["m"],
        "best_dsr": q("SELECT MAX(dsr) m FROM strategies")[0]["m"],
        "entry_blocks_tried": q("SELECT COUNT(DISTINCT json_extract(genome,'$.entry.name')) n FROM strategies")[0]["n"],
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }]
    progress = q("SELECT generation, COUNT(*) AS tried, MAX(fitness) AS best FROM strategies "
                 "WHERE fitness > -5 GROUP BY generation ORDER BY generation")
    run_best = -1e9
    for p in progress:
        run_best = max(run_best, p["best"])
        p["best_so_far"] = run_best

    def row(r):
        info = json.loads(r["quick_info"] or "{}")
        g = json.loads(r["genome"])
        return {"id": r["id"], "symbol": r["symbol"], "tf": r["tf"], "entry": g["entry"]["name"],
                "description": r["description"], "status": r["status"], "fitness": round(r["fitness"], 3),
                "is_pass": (info.get("challenge_is") or {}).get("p_pass_all"),
                "holdout_pass": r["p_pass_oos"], "dsr": r["dsr"],
                "trades_is": (info.get("is") or {}).get("n"), "avg_r": (info.get("is") or {}).get("expectancy_r"),
                "failed_checks": len(json.loads(r["reasons"] or "[]"))}

    # leaderboard: best per entry block so near-copies don't fill it
    seen, board = set(), []
    for r in q("SELECT * FROM strategies ORDER BY fitness DESC LIMIT 2000"):
        g = json.loads(r["genome"])["entry"]["name"]
        if g in seen:
            continue
        seen.add(g)
        board.append(row(r))
        if len(board) >= 20:
            break
    holdout = [row(r) for r in q("SELECT * FROM strategies WHERE p_pass_oos IS NOT NULL "
                                 "ORDER BY p_pass_oos DESC, dsr DESC LIMIT 12")]
    cnt = Counter()
    labels = {"deflated Sharpe": "Deflated Sharpe below 0.9", "OOS P(pass) degraded": "Pass rate collapsed on holdout",
              "OOS P(pass)": "Holdout pass rate below 20%", "does not generalise": "Loses on other symbols",
              "OOS expectancy": "Loses money on holdout", "IS P(pass)": "In-sample pass rate below 25%",
              "IS expectancy": "In-sample edge too small", "edge not stable": "Edge unstable across time folds",
              "OOS trades": "Too few holdout trades", "IS trades": "Too few in-sample trades"}
    for (rs,) in c.execute("SELECT reasons FROM strategies WHERE status='rejected'"):
        for x in json.loads(rs or "[]"):
            lab = next((v for k, v in labels.items() if x.startswith(k)), re.sub(r"[\d.]+", "#", x))
            cnt[lab] += 1
    reasons = [{"check": k, "rejected": v} for k, v in cnt.most_common()]
    for name, data in {"summary": summary, "progress": progress, "leaderboard": board, "holdout": holdout,
                       "reasons": reasons}.items():
        (out_dir / f"{name}.json").write_text(json.dumps(data, default=float))
    print(f"exported gen {summary[0]['generation']}, {summary[0]['trials']} trials -> {out_dir}/")


if __name__ == "__main__":
    main(*sys.argv[1:])
