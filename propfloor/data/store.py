"""Parquet store: data/parquet/{SYMBOL}_M1.parquet + provenance sidecar json."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from propfloor.data.clean import QualityReport, clean_bars, resample

log = logging.getLogger(__name__)

DATA_DIR = Path("data/parquet")


CCY = {"EUR", "GBP", "USD", "JPY", "AUD", "NZD", "CAD", "CHF", "SEK", "NOK", "DKK", "SGD", "HKD", "MXN", "ZAR",
       "TRY", "PLN", "CNH", "XAU", "XAG"}


def pip_size(symbol: str) -> float:
    s = symbol.upper()
    if s.endswith("JPY"):
        return 0.01
    if s.startswith("XAU"):
        return 0.1
    if s.startswith("XAG"):
        return 0.01
    if len(s) == 6 and s.isalpha() and s[:3] in CCY and s[3:] in CCY:
        return 0.0001
    return 0.01  # stocks, indices, oil: one cent / one index hundredth


class DataStore:
    def __init__(self, root: str | Path = DATA_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, symbol: str) -> Path:
        return self.root / f"{symbol.upper()}_M1.parquet"

    def meta_path(self, symbol: str) -> Path:
        return self.root / f"{symbol.upper()}_M1.meta.json"

    def symbols(self) -> list[str]:
        return sorted(p.name.split("_")[0] for p in self.root.glob("*_M1.parquet"))

    def meta(self, symbol: str) -> dict:
        p = self.meta_path(symbol)
        return json.loads(p.read_text()) if p.exists() else {"sources": [], "synthetic": False}

    def is_synthetic(self, symbol: str) -> bool:
        return bool(self.meta(symbol).get("synthetic"))

    def write(self, symbol: str, df: pd.DataFrame, source: str, synthetic: bool = False,
              prefer_new: bool = False) -> QualityReport:
        """Clean and merge bars into the store. Existing (user-provided) bars win unless prefer_new."""
        df, rep = clean_bars(df)
        p = self.path(symbol)
        meta = self.meta(symbol)
        if p.exists():
            old = pd.read_parquet(p)
            if meta.get("synthetic") != synthetic:
                raise ValueError(f"refusing to mix synthetic and real data for {symbol}")
            both = pd.concat([df, old] if not prefer_new else [old, df])
            df = both[~both.index.duplicated(keep="last")].sort_index()
        df.to_parquet(p)
        meta["sources"] = sorted(set(meta.get("sources", [])) | {source})
        meta["synthetic"] = synthetic
        meta.update(start=str(df.index.min()), end=str(df.index.max()), bars=len(df))
        meta.setdefault("quality", []).append({"source": source, **{k: v for k, v in rep.__dict__.items() if k != "gaps"},
                                               "n_gaps": len(rep.gaps)})
        self.meta_path(symbol).write_text(json.dumps(meta, indent=1, default=str))
        log.info("%s: stored %d bars from %s (%d dup, %d spikes, %d gaps)", symbol, len(df), source,
                 rep.duplicates, rep.spikes, len(rep.gaps))
        return rep

    def load(self, symbol: str, tf: str = "M1", start=None, end=None) -> pd.DataFrame:
        df = pd.read_parquet(self.path(symbol))
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index < pd.Timestamp(end, tz="UTC")]
        return df if tf.upper() == "M1" else resample(df, tf)


def ingest_raw(raw_dir: str | Path = "data/raw", store: DataStore | None = None) -> dict[str, QualityReport]:
    from propfloor.data.loader import load_raw_dir

    store = store or DataStore()
    out = {}
    for sym, items in load_raw_dir(raw_dir).items():
        for d in items:
            out[f"{sym}:{d.fmt}"] = store.write(sym, d.df, source=f"user:{d.fmt}", prefer_new=True)
    return out
