"""Optional Claude assistant: proposes novel genomes and critiques failures.

Enabled when the `anthropic` package is installed and credentials are available
(ANTHROPIC_API_KEY). Every proposal is validated against the block registry before use, and every
proposal still pays the full multiple-testing cost (it counts as a trial like any other genome).
"""
from __future__ import annotations

import json
import logging
import os
import re

from propfloor.strategies.blocks import ENTRIES, EXIT_SPACE, FILTERS, OVERLAY_SPACE, SIZING_SPACE
from propfloor.strategies.genome import Component, StrategyGenome

log = logging.getLogger(__name__)

MODEL = os.environ.get("PROPFLOOR_LLM_MODEL", "claude-opus-5-5")


def catalogue() -> str:
    spec = lambda s: {k: list(v[1]) if v[0] == "choice" else [v[0], v[1], v[2]] for k, v in s.items()}  # noqa: E731
    return json.dumps({
        "entries": {n: {"family": b.family, "params": spec(b.space)} for n, b in ENTRIES.items()},
        "filters": {n: spec(b.space) for n, b in FILTERS.items()},
        "exit": spec(EXIT_SPACE), "sizing": spec(SIZING_SPACE), "overlay": spec(OVERLAY_SPACE),
    })


SYSTEM = (
    "You are a quantitative FX researcher designing rule-based strategies whose goal is to PASS prop-firm "
    "challenges (profit target with a trailing max drawdown and a daily drawdown limit) and then survive funded. "
    "You may only use the building blocks in the catalogue provided. Prefer ideas with a plausible market "
    "rationale (session effects, volatility regimes, structure) over curve-fit parameter tweaks."
)


class ClaudeResearcher:
    def __init__(self, db=None, model: str = MODEL) -> None:
        import anthropic  # noqa: F401  (raises ImportError if missing)

        self.client = anthropic.Anthropic()
        self.model = model
        self.db = db

    @classmethod
    def maybe(cls, db=None) -> "ClaudeResearcher | None":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        try:
            return cls(db)
        except Exception as e:  # noqa: BLE001
            log.warning("Claude researcher disabled: %s", e)
            return None

    def _ask(self, prompt: str, kind: str) -> str:
        import anthropic

        try:
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=16000, system=SYSTEM,
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": prompt}],
            )
        except (anthropic.BadRequestError, TypeError):  # older SDKs / platforms without fallbacks
            resp = self.client.messages.create(model=self.model, max_tokens=16000, system=SYSTEM,
                                               messages=[{"role": "user", "content": prompt}])
        if resp.stop_reason == "refusal":
            return ""
        text = "".join(b.text for b in resp.content if b.type == "text")
        if self.db is not None:
            self.db.log_llm(kind, prompt[-4000:], text)
        return text

    def propose(self, n: int, leaderboard: list[dict], failures: list[dict], symbols: list[str],
                timeframes: list[str]) -> list[StrategyGenome]:
        prompt = (
            f"Catalogue of building blocks (param ranges are [type, lo, hi] or a list of choices):\n{catalogue()}\n\n"
            f"Symbols: {symbols}. Timeframes: {timeframes}.\n"
            f"Current leaderboard (in-sample fitness = P(pass both stages)):\n{json.dumps(leaderboard)[:6000]}\n\n"
            f"Recent rejected candidates and why:\n{json.dumps(failures)[:4000]}\n\n"
            f"Propose {n} NEW, diverse strategies. Reply with ONLY a JSON array; each item: "
            '{"symbol":..,"tf":..,"entry":{"name":..,"params":{..}},"filters":[{"name":..,"params":{..}}],'
            '"exit":{..all exit keys..},"sizing":{"risk_per_trade":..},"overlay":{..account tactics..},'
            '"direction":"both|long|short","rationale":".."}'
        )
        return parse_genomes(self._ask(prompt, "propose"))

    def critique(self, failures: list[dict]) -> str:
        return self._ask("Briefly diagnose the common failure modes in these rejected prop-challenge strategies "
                         f"and suggest research directions:\n{json.dumps(failures)[:8000]}", "critique")


def parse_genomes(text: str) -> list[StrategyGenome]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        try:
            g = coerce(it)
            if g is not None:
                out.append(g)
        except Exception:  # noqa: BLE001 - discard malformed proposals
            continue
    return out


def _clip(params: dict, space: dict) -> dict:
    from propfloor.strategies.blocks import sample_param
    import numpy as np

    rng = np.random.default_rng()
    out = {}
    for k, spec in space.items():
        v = params.get(k)
        if spec[0] == "int":
            out[k] = int(min(max(int(v), spec[1]), spec[2])) if isinstance(v, (int, float)) else sample_param(spec, rng)
        elif spec[0] == "float":
            out[k] = float(min(max(float(v), spec[1]), spec[2])) if isinstance(v, (int, float)) else sample_param(spec, rng)
        else:
            choices = spec[1]
            nums = [c for c in choices if isinstance(c, (int, float)) and not isinstance(c, bool)]
            if v in choices:
                out[k] = v
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and nums:
                out[k] = min(nums, key=lambda c: abs(c - v))  # snap to the nearest allowed value, never random
            else:
                out[k] = sample_param(spec, rng)
    return out


def coerce(d: dict) -> StrategyGenome | None:
    e = d["entry"]
    if e["name"] not in ENTRIES:
        return None
    filters = [Component(f["name"], _clip(f.get("params", {}), FILTERS[f["name"]].space))
               for f in d.get("filters", []) if f.get("name") in FILTERS and f["name"] != "none"]
    return StrategyGenome(
        tf=d.get("tf", "H1"), entry=Component(e["name"], _clip(e.get("params", {}), ENTRIES[e["name"]].space)),
        filters=filters[:3], exit=_clip(d.get("exit", {}), EXIT_SPACE), sizing=_clip(d.get("sizing", {}), SIZING_SPACE),
        direction=d.get("direction", "both") if d.get("direction") in ("both", "long", "short") else "both",
        symbol=d.get("symbol", "EURUSD"), origin="llm", overlay=_clip(d.get("overlay", {}), OVERLAY_SPACE))
