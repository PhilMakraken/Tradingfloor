"""Write the briefing pack for strategy-research agents: block catalogue + current factory state."""
import json
import sqlite3
from collections import Counter
from pathlib import Path

from propfloor.factory.llm import catalogue

out = Path("research")
out.mkdir(exist_ok=True)
(out / "catalogue.json").write_text(json.dumps(json.loads(catalogue()), indent=1))
c = sqlite3.connect("propfloor.sqlite")
c.row_factory = sqlite3.Row
best = c.execute("""SELECT description, fitness, p_pass_is, p_pass_oos, dsr, reasons FROM (SELECT *, ROW_NUMBER() OVER
  (PARTITION BY json_extract(genome,'$.entry.name') ORDER BY fitness DESC) rn FROM strategies) WHERE rn=1
  ORDER BY fitness DESC LIMIT 25""").fetchall()
fam = c.execute("""SELECT json_extract(genome,'$.entry.name') e, COUNT(*) n, MAX(fitness) f FROM strategies
  GROUP BY e ORDER BY f DESC""").fetchall()
sym = c.execute("SELECT symbol, COUNT(*) n, MAX(fitness) f, MAX(p_pass_oos) o FROM strategies GROUP BY symbol ORDER BY f DESC").fetchall()
tf = c.execute("SELECT tf, COUNT(*) n, MAX(fitness) f, MAX(p_pass_oos) o FROM strategies GROUP BY tf ORDER BY f DESC").fetchall()
reasons = Counter()
for (r,) in c.execute("SELECT reasons FROM strategies WHERE status='rejected'"):
    for x in json.loads(r or "[]"):
        reasons[x.split(" ")[0] + " " + " ".join(x.split(" ")[1:3])] += 1
L = ["# Factory state (strategy research brief)", "",
     f"Trials so far: {c.execute('SELECT value FROM meta WHERE key=\"n_trials\"').fetchone()[0]}. Validated: 0.", "",
     "Program: 2 stages, 8% target each, 5% TRAILING max drawdown (equity, locks at start), 3% daily DD, min 4 trading days.",
     "Data: 12 symbols (EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF EURGBP EURJPY GBPJPY AUDJPY EURCHF XAUUSD), M15 bars",
     "Nov-2012..Mar-2022; timeframes M15 M30 H1 H4 D1. In-sample = first 75%, holdout = last 25% (Dec-2019..Mar-2022).",
     "Costs: 1.0 pip spread + 0.3 pip slippage each side + $7/lot commission. Flat before the weekend (Fri 15:00 NY).",
     "", "Validation gates (ALL required): IS>=40 trades, IS expectancy>=0.05R, >=75% of 4 walk-forward folds positive,",
     "deflated Sharpe>=0.90 (per-trade Sharpe, deflated by number of trials), OOS>=12 trades, OOS expectancy>0,",
     "IS P(pass both stages)>=0.25, OOS P(pass)>=0.20 and >=50% of IS, positive on >=50% of other symbols where it trades.",
     "", "## Best strategy per entry block", "| fitness | IS pass | OOS pass | DSR | rules |", "|---|---|---|---|---|"]
for r in best:
    f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    L.append(f"| {r['fitness']:.2f} | {f(r['p_pass_is'])} | {f(r['p_pass_oos'])} | {f(r['dsr'])} | {r['description']} |")
L += ["", "## Effort and best fitness by entry block", ", ".join(f"{r['e']}:{r['n']}/{r['f']:.2f}" for r in fam), "",
      "## By symbol (tried / best fitness / best OOS pass)", ", ".join(f"{r['symbol']}:{r['n']}/{r['f']:.2f}/{(r['o'] or 0):.2f}" for r in sym),
      "", "## By timeframe", ", ".join(f"{r['tf']}:{r['n']}/{r['f']:.2f}/{(r['o'] or 0):.2f}" for r in tf),
      "", "## Rejection reasons (counts)", *[f"- {k}: {v}" for k, v in reasons.most_common(10)]]
(out / "state.md").write_text("\n".join(L))
print("wrote research/catalogue.json and research/state.md")
