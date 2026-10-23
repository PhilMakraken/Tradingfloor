"""Strategy genome: entry x filters x exit x sizing, plus random generation / mutation / crossover."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from propfloor.strategies.backtest import run_backtest
from propfloor.strategies.blocks import (ENTRIES, EXIT_SPACE, FILTERS, SIZING_SPACE, TIMEFRAMES, Ctx, sample_param,
                                         sample_params)


@dataclass
class Component:
    name: str
    params: dict = field(default_factory=dict)


@dataclass
class StrategyGenome:
    tf: str
    entry: Component
    filters: list[Component]
    exit: dict
    sizing: dict
    direction: str = "both"  # both | long | short
    symbol: str = "EURUSD"
    origin: str = "random"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyGenome":
        d = copy.deepcopy(d)
        d["entry"] = Component(**d["entry"])
        d["filters"] = [Component(**f) for f in d["filters"]]
        return cls(**d)

    def key(self) -> str:
        d = self.to_dict()
        d.pop("origin", None)
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def describe(self) -> str:
        f = " & ".join(f"{c.name}" for c in self.filters if c.name != "none") or "no filter"
        e = self.exit
        return (f"{self.symbol} {self.tf} {self.entry.name}({_fmt(self.entry.params)}) | {f} | SL {e['sl_type']} {e['sl_atr']:.2f}ATR "
                f"TP {e['tp_r']}R ts{e['time_stop']} trail{e['trail_atr']} | {self.direction}")

    # ------------------------------------------------------------- signals
    def signals(self, ctx: Ctx) -> tuple[np.ndarray, np.ndarray]:
        eb = ENTRIES[self.entry.name]
        lo, sh = eb.fn(ctx, self.entry.params)
        lo, sh = lo.copy(), sh.copy()
        for f in self.filters:
            al, as_ = FILTERS[f.name].fn(ctx, f.params)
            lo &= al
            sh &= as_
        if self.direction == "long":
            sh[:] = False
        elif self.direction == "short":
            lo[:] = False
        return lo, sh

    def backtest(self, ctx: Ctx, **costs) -> pd.DataFrame:
        lo, sh = self.signals(ctx)
        return run_backtest(ctx, lo, sh, self.exit, **costs)


def _fmt(p: dict) -> str:
    return ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def random_genome(rng: np.random.Generator, timeframes=TIMEFRAMES, max_filters: int = 2,
                  symbols=("EURUSD",)) -> StrategyGenome:
    en = list(ENTRIES)[int(rng.integers(len(ENTRIES)))]
    nf = int(rng.integers(0, max_filters + 1))
    fnames = [n for n in FILTERS if n != "none"]
    chosen = list(rng.choice(fnames, size=nf, replace=False)) if nf else []
    return StrategyGenome(
        tf=str(timeframes[int(rng.integers(len(timeframes)))]),
        entry=Component(en, sample_params(ENTRIES[en].space, rng)),
        filters=[Component(str(f), sample_params(FILTERS[str(f)].space, rng)) for f in chosen],
        exit=sample_params(EXIT_SPACE, rng),
        sizing=sample_params(SIZING_SPACE, rng),
        direction=str(rng.choice(["both", "both", "long", "short"])),
        symbol=str(symbols[int(rng.integers(len(symbols)))]),
    )


def _mutate_params(params: dict, space: dict, rng, scale=0.2) -> dict:
    p = dict(params)
    if not space:
        return p
    k = list(space)[int(rng.integers(len(space)))]
    spec = space[k]
    if spec[0] in ("int", "float") and k in p and rng.random() < 0.7:
        lo, hi = spec[1], spec[2]
        v = p[k] + rng.normal(0, scale * (hi - lo))
        v = min(max(v, lo), hi)
        p[k] = int(round(v)) if spec[0] == "int" else round(float(v), 4)
    else:
        p[k] = sample_param(spec, rng)
    return p


def mutate(g: StrategyGenome, rng: np.random.Generator, timeframes=TIMEFRAMES, symbols=None) -> StrategyGenome:
    g = StrategyGenome.from_dict(g.to_dict())
    g.origin = "mutation"
    r = rng.random()
    if r < 0.3:
        g.entry.params = _mutate_params(g.entry.params, ENTRIES[g.entry.name].space, rng)
    elif r < 0.5:
        g.exit = _mutate_params(g.exit, EXIT_SPACE, rng)
    elif r < 0.62 and g.filters:
        i = int(rng.integers(len(g.filters)))
        g.filters[i].params = _mutate_params(g.filters[i].params, FILTERS[g.filters[i].name].space, rng)
    elif r < 0.74:
        fnames = [n for n in FILTERS if n != "none" and n not in {f.name for f in g.filters}]
        if len(g.filters) < 3 and fnames and rng.random() < 0.6:
            n = str(rng.choice(fnames))
            g.filters.append(Component(n, sample_params(FILTERS[n].space, rng)))
        elif g.filters:
            g.filters.pop(int(rng.integers(len(g.filters))))
    elif r < 0.82:
        g.tf = str(timeframes[int(rng.integers(len(timeframes)))])
    elif r < 0.86 and symbols:
        g.symbol = str(symbols[int(rng.integers(len(symbols)))])
    elif r < 0.9:
        g.direction = str(rng.choice(["both", "long", "short"]))
    elif r < 0.95:
        g.sizing = sample_params(SIZING_SPACE, rng)
    else:
        en = list(ENTRIES)[int(rng.integers(len(ENTRIES)))]
        g.entry = Component(en, sample_params(ENTRIES[en].space, rng))
    return g


def crossover(a: StrategyGenome, b: StrategyGenome, rng: np.random.Generator) -> StrategyGenome:
    pick = lambda x, y: copy.deepcopy(x if rng.random() < 0.5 else y)  # noqa: E731
    return StrategyGenome(tf=pick(a.tf, b.tf), entry=pick(a.entry, b.entry), filters=pick(a.filters, b.filters),
                          exit=pick(a.exit, b.exit), sizing=pick(a.sizing, b.sizing),
                          direction=pick(a.direction, b.direction), symbol=pick(a.symbol, b.symbol),
                          origin="crossover")
