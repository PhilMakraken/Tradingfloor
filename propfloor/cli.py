"""Command line: python -m propfloor <group> <command>."""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from propfloor.config import DEFAULT_PROGRAM, load_program

log = logging.getLogger("propfloor")

# shorthand symbol lists for `data download --symbols <name>`
UNIVERSES = {
    "fx-majors": "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,NZDUSD",
    "metals": "XAUUSD,XAGUSD",
    "stocks": "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,JPM,XOM,BRK-B",
    "commodities": "WTI,BRENT,NATGAS,COPPER",
    "indices": "US500,US100,US30,GER40,UK100,JPN225",
}


def _store(args):
    from propfloor.data.store import DataStore

    return DataStore(args.data_dir)


def cmd_ingest(args):
    from propfloor.data.store import ingest_raw

    reps = ingest_raw(args.raw_dir, _store(args))
    for k, r in reps.items():
        print(f"{k}: {r.n_out} bars (dup {r.duplicates}, invalid {r.invalid}, spikes {r.spikes}, gaps {len(r.gaps)})")
    if not reps:
        print(f"no parsable files in {args.raw_dir}")


def cmd_download(args):
    from propfloor.data import dukascopy, github_mirror, histdata, yahoo

    st = _store(args)
    mod = {"dukascopy": dukascopy, "histdata": histdata, "github": github_mirror, "yahoo": yahoo}[args.source]
    if args.symbols in UNIVERSES:
        args.symbols = UNIVERSES[args.symbols]
    for sym in args.symbols.split(","):
        try:
            df = mod.download(sym, args.start, args.end)
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: download failed: {e}")
            continue
        if len(df):
            # user-provided data (already stored) wins on overlapping timestamps
            st.write(sym, df, source=args.source, prefer_new=False)
            print(f"{sym}: {len(df)} bars from {args.source}")


def cmd_synth(args):
    from propfloor.data.synthetic import make_bars

    st = _store(args)
    for i, sym in enumerate(args.symbols.split(",")):
        st.write(sym, make_bars(args.start, args.end, seed=args.seed + i, price0=150.0 if sym.endswith("JPY") else 1.2),
                 source="synthetic", synthetic=True)
        print(f"{sym}: synthetic bars written (NOT eligible for validation)")


def _evaluator(args, program):
    from propfloor.data.store import DataStore
    from propfloor.evaluation.walkforward import Evaluator, Gates, Universe

    st = DataStore(args.data_dir)
    syms = args.symbols.split(",") if args.symbols else st.symbols()
    if not syms:
        sys.exit("no data: run `propfloor data ingest`, `data download` or `data synth` first")
    fake = [s for s in syms if st.is_synthetic(s)]
    if fake and not args.allow_synthetic:
        print(f"skipping synthetic (generated) data: {','.join(fake)} - only real market data is researched")
        syms = [s for s in syms if s not in fake]
        if not syms:
            sys.exit("no real market data: run `propfloor data download` first")
    data = {s: st.load(s, start=args.start) for s in syms}
    tfs = args.timeframes.split(",")
    u = Universe(data, timeframes=tfs, oos_frac=args.oos_frac, synthetic={s: st.is_synthetic(s) for s in syms})
    gates = Gates(allow_synthetic=args.allow_synthetic, min_history_years=args.min_years,
                  min_positive_years=args.min_positive_years, robustness=not args.no_robustness)
    return Evaluator(u, program, gates, quick_sims=args.quick_sims, full_sims=args.full_sims)


def _forward(args, program, db):
    from propfloor.data.store import DataStore
    from propfloor.live.forward import ForwardTester

    c = program.costs
    return ForwardTester(db, DataStore(args.data_dir), DataStore(args.live_dir),
                         costs=dict(spread_pips=c.spread_pips, slippage_pips=c.slippage_pips,
                                    commission_per_lot=c.commission_per_lot))


def cmd_factory(args):
    from propfloor.db import ResultsDB
    from propfloor.factory.engine import Factory
    from propfloor.factory.llm import ClaudeResearcher

    program = load_program(args.program)
    db = ResultsDB(args.db)
    ev = _evaluator(args, program)
    llm = ClaudeResearcher.maybe(db) if args.llm else None
    if args.llm and llm is None:
        print("Claude researcher unavailable (set ANTHROPIC_API_KEY and `pip install anthropic`); continuing without it")
    fwd = _forward(args, program, db) if args.live else None
    f = Factory(ev, db, seed=args.seed, pop_size=args.pop, candidate_threshold=args.threshold, llm=llm,
                idea_dir=args.idea_dir, forward=fwd, forward_every_min=args.live_every)
    stop = None
    if args.agents:
        from propfloor.factory.agents import AgentTeam

        team = AgentTeam(ev.u.symbols, ev.u.timeframes, idea_dir=args.idea_dir, backend=args.agent_backend, db=db)
        print(f"research agents: {', '.join(team.roles)} via {team.backend}, one every {args.agent_every} min")
        stop = team.start(args.agent_every)
    try:
        f.run(max_generations=args.generations, max_hours=args.hours)
    finally:
        if stop is not None:
            stop.set()


def cmd_agents(args):
    from propfloor.data.store import DataStore
    from propfloor.db import ResultsDB
    from propfloor.factory.agents import AgentTeam
    from propfloor.factory.notes import write_notes

    db = ResultsDB(args.db)
    write_notes(db)
    syms = args.symbols.split(",") if args.symbols else DataStore(args.data_dir).symbols()
    team = AgentTeam(syms, args.timeframes.split(","), idea_dir=args.idea_dir, backend=args.agent_backend, db=db,
                     roles=args.roles.split(",") if args.roles else None)
    for role in team.roles:
        print(f"{role}: {team.run_role(role)} ideas queued in {args.idea_dir}")


def cmd_live(args):
    import time

    from propfloor.db import ResultsDB

    program = load_program(args.program)
    db = ResultsDB(args.db)
    fwd = _forward(args, program, db)
    while True:
        res = fwd.run_once() if args.cmd == "run" else fwd.evaluate()
        if res:
            print(pd.DataFrame(res).sort_values("longevity", ascending=False).to_string(index=False))
        else:
            print("no strategies in forward test yet (they are enrolled when the factory validates them)")
        if args.cmd != "run" or args.once:
            break
        time.sleep(args.every * 60)


def cmd_notes(args):
    from propfloor.db import ResultsDB
    from propfloor.factory.notes import write_notes

    print(f"wrote {write_notes(ResultsDB(args.db))}")


def cmd_floor(args):
    from propfloor.db import ResultsDB
    from propfloor.floor.floor import TradingFloor

    program = load_program(args.program)
    db = ResultsDB(args.db)
    if not db.validated():
        sys.exit("no validated strategies yet - run the factory first")
    db.execute("DELETE FROM floor_accounts")
    db.execute("DELETE FROM floor_events")
    fl = TradingFloor(program, db)
    fl.open_accounts(args.accounts, per_account=args.per_account)
    ev = _evaluator(args, program)
    ids = {s for a in fl.accounts.values() for s in a.strategies}
    trades = {sid: ev.trades(g) for sid, g in fl.genomes(list(ids)).items()}
    fl.run(trades, start=args.since, rotate_top=args.rotate_top)
    print(fl.summary().to_string(index=False))


def cmd_status(args):
    from propfloor.db import ResultsDB

    db = ResultsDB(args.db)
    print(f"trials: {db.n_trials}  generation: {db.get_meta('generation', 0)}  validated: {len(db.validated())}")
    rows = db.top(args.n)
    if rows:
        df = pd.DataFrame(rows)[["id", "status", "fitness", "p_pass_is", "p_pass_oos", "dsr", "description"]]
        print(df.to_string(index=False, max_colwidth=90))


def main(argv=None):
    p = argparse.ArgumentParser(prog="propfloor")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--db", default="propfloor.sqlite")
    p.add_argument("--data-dir", default="data/parquet")
    p.add_argument("--program", default=str(DEFAULT_PROGRAM))
    sub = p.add_subparsers(dest="group", required=True)

    d = sub.add_parser("data").add_subparsers(dest="cmd", required=True)
    x = d.add_parser("ingest")
    x.add_argument("--raw-dir", default="data/raw")
    x.set_defaults(fn=cmd_ingest)
    x = d.add_parser("download")
    x.add_argument("--source", choices=["dukascopy", "histdata", "github", "yahoo"], default="dukascopy")
    x.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF,NZDUSD,EURJPY,GBPJPY,EURGBP")
    x.add_argument("--start", default="2015-01-01")
    x.add_argument("--end", default=None)
    x.set_defaults(fn=cmd_download)
    x = d.add_parser("synth")
    x.add_argument("--symbols", default="EURUSD,GBPUSD")
    x.add_argument("--start", default="2018-01-01")
    x.add_argument("--end", default="2022-01-01")
    x.add_argument("--seed", type=int, default=0)
    x.set_defaults(fn=cmd_synth)

    def eval_opts(x):
        x.add_argument("--symbols", default=None)
        x.add_argument("--timeframes", default="M15,M30,H1,H4,D1")
        x.add_argument("--start", default=None)
        x.add_argument("--oos-frac", type=float, default=0.25)
        x.add_argument("--quick-sims", type=int, default=30)
        x.add_argument("--full-sims", type=int, default=150)
        x.add_argument("--allow-synthetic", action="store_true", help="DEV ONLY: validate on synthetic data")
        x.add_argument("--min-years", type=float, default=10.0, help="required years of real history")
        x.add_argument("--min-positive-years", type=float, default=0.7, help="share of calendar years with profit")
        x.add_argument("--no-robustness", action="store_true", help="skip the Monte Carlo anti-overfitting tests")

    f = sub.add_parser("factory").add_subparsers(dest="cmd", required=True)
    x = f.add_parser("run")
    eval_opts(x)
    x.add_argument("--hours", type=float, default=None, help="stop after N hours (default: run forever)")
    x.add_argument("--generations", type=int, default=None)
    x.add_argument("--pop", type=int, default=40)
    x.add_argument("--threshold", type=float, default=0.15)
    x.add_argument("--seed", type=int, default=None)
    x.add_argument("--llm", action="store_true", help="use Claude for idea generation/critique")
    x.add_argument("--idea-dir", default="data/ideas", help="folder of queued strategy-idea JSON files")
    x.add_argument("--agents", action="store_true", help="run the Claude research-agent team alongside")
    x.add_argument("--agent-every", type=float, default=30.0, help="minutes between agent research rounds")
    x.add_argument("--agent-backend", choices=["auto", "claude-code", "api"], default="auto")
    x.add_argument("--live", action="store_true", help="forward-test validated strategies on live market data")
    x.add_argument("--live-every", type=float, default=60.0, help="minutes between live data polls")
    x.add_argument("--live-dir", default="data/live")
    x.set_defaults(fn=cmd_factory)

    x = sub.add_parser("agents").add_subparsers(dest="cmd", required=True).add_parser("run")
    x.add_argument("--symbols", default=None)
    x.add_argument("--timeframes", default="M15,M30,H1,H4,D1")
    x.add_argument("--roles", default=None, help="comma list (default: all)")
    x.add_argument("--idea-dir", default="data/ideas")
    x.add_argument("--agent-backend", choices=["auto", "claude-code", "api"], default="auto")
    x.set_defaults(fn=cmd_agents)

    lv = sub.add_parser("live").add_subparsers(dest="cmd", required=True)
    for name in ("run", "status"):
        x = lv.add_parser(name)
        x.add_argument("--live-dir", default="data/live")
        x.add_argument("--every", type=float, default=60.0, help="minutes between polls (run)")
        x.add_argument("--once", action="store_true")
        x.set_defaults(fn=cmd_live)

    x = sub.add_parser("notes")
    x.set_defaults(fn=cmd_notes)

    fl = sub.add_parser("floor").add_subparsers(dest="cmd", required=True)
    x = fl.add_parser("run")
    eval_opts(x)
    x.add_argument("--accounts", type=int, default=10)
    x.add_argument("--per-account", type=int, default=2)
    x.add_argument("--since", default=None, help="paper-trade from this date")
    x.add_argument("--rotate-top", type=int, default=0, help="P7: monthly rotation to the top-N strategies per account")
    x.set_defaults(fn=cmd_floor)

    x = sub.add_parser("status")
    x.add_argument("-n", type=int, default=20)
    x.set_defaults(fn=cmd_status)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
