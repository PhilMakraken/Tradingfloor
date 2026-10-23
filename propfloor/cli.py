"""Command line: python -m propfloor <group> <command>."""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from propfloor.config import DEFAULT_PROGRAM, load_program

log = logging.getLogger("propfloor")


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
    from propfloor.data import dukascopy, github_mirror, histdata

    st = _store(args)
    mod = {"dukascopy": dukascopy, "histdata": histdata, "github": github_mirror}[args.source]
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
    data = {s: st.load(s, start=args.start) for s in syms}
    tfs = args.timeframes.split(",")
    u = Universe(data, timeframes=tfs, oos_frac=args.oos_frac, synthetic={s: st.is_synthetic(s) for s in syms})
    return Evaluator(u, program, Gates(allow_synthetic=args.allow_synthetic), quick_sims=args.quick_sims, full_sims=args.full_sims)


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
    f = Factory(ev, db, seed=args.seed, pop_size=args.pop, candidate_threshold=args.threshold, llm=llm)
    f.run(max_generations=args.generations, max_hours=args.hours)


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
    x.add_argument("--source", choices=["dukascopy", "histdata", "github"], default="dukascopy")
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

    f = sub.add_parser("factory").add_subparsers(dest="cmd", required=True)
    x = f.add_parser("run")
    eval_opts(x)
    x.add_argument("--hours", type=float, default=None, help="stop after N hours (default: run forever)")
    x.add_argument("--generations", type=int, default=None)
    x.add_argument("--pop", type=int, default=40)
    x.add_argument("--threshold", type=float, default=0.15)
    x.add_argument("--seed", type=int, default=None)
    x.add_argument("--llm", action="store_true", help="use Claude for idea generation/critique")
    x.set_defaults(fn=cmd_factory)

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
