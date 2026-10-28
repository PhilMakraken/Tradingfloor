"""Team of Claude research agents that run alongside the factory and feed it strategy ideas.

Each agent has a specialist brief, reads the compact research notes (what has been tried and why it failed), may
research online (Claude Code backend: WebSearch/WebFetch), and replies with strategy genomes built ONLY from the
block catalogue. Ideas are written to `idea_dir` as JSON; the factory consumes them next generation, where they pay
the same multiple-testing cost and pass the same gates as everything else. An agent's opinion never validates
anything.

Backends:  "claude-code" -> the `claude` CLI in print mode (Claude Code agents; uses your Claude Code login)
           "api"         -> the Anthropic API (ANTHROPIC_API_KEY), no web access
           "auto"        -> claude-code if the CLI is on PATH, else api
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

from propfloor.factory.llm import SYSTEM, catalogue, parse_genomes

log = logging.getLogger(__name__)

ROLES: dict[str, str] = {
    "technical_analyst": (
        "You are the TECHNICAL ANALYSIS researcher. Combine classic indicators (moving averages, RSI, MACD, "
        "Bollinger, ADX, stochastics, Donchian, candlestick patterns) with trend/volatility filters and sensible "
        "multi-timeframe context. Favour simple rules that should hold across many markets and a decade of data."),
    "price_action": (
        "You are the PRICE-ACTION / SMART-MONEY researcher. Automate discretionary concepts: the ICC concept "
        "(Indication = break of structure, Correction = pullback, Continuation = resumption; block `icc`), the "
        "session reclaim continuation (session breakout, retrace into the ~0.70-0.886 Fibonacci zone, reclaim and "
        "continue; block `session_reclaim`), opening-range and previous-day breaks, liquidity sweeps, acceptance "
        "breaks, swing structure. Vary sessions, fib zones, anchors, swing sizes and exits to find a robust edge."),
    "quant": (
        "You are the QUANT researcher. Use statistical, regime, intermarket and ML blocks (stat_feature, "
        "stat_regime, hurst/choppiness/autocorrelation filters, intermarket and basket blocks, regime_router, "
        "ml_classifier). Look for effects with an economic rationale (risk-on/off, dollar strength, volatility "
        "clustering, session drift) rather than parameter tweaks."),
    "risk_critic": (
        "You are the RISK & ROBUSTNESS critic. Study the near-misses in the notes and their rejection reasons "
        "(Monte Carlo, random-entry test, parameter fragility, yearly consistency, OOS degradation). Propose "
        "repaired variants: better exits, stops, regime filters and account overlays that cut drawdown and make "
        "the edge survive every year rather than one lucky period."),
}

CONTRACT = (
    "Reply with ONLY a JSON array of {n} strategies (no prose outside it). Each item: "
    '{{"symbol":..,"tf":..,"entry":{{"name":..,"params":{{..}}}},"filters":[{{"name":..,"params":{{..}}}}],'
    '"exit":{{..all exit keys..}},"sizing":{{"risk_per_trade":..}},"overlay":{{..}},'
    '"direction":"both|long|short","rationale":".."}}. Use only block names and parameter ranges from the catalogue; '
    "do not repeat niches the notes mark as exhausted."
)


class AgentTeam:
    def __init__(self, symbols: list[str], timeframes: list[str], idea_dir: str | Path = "data/ideas",
                 notes_path: str | Path = "data/RESEARCH_NOTES.md", log_dir: str | Path = "data/agents",
                 backend: str = "auto", roles: list[str] | None = None, n_ideas: int = 6, timeout: int = 1200,
                 model: str | None = None, db=None) -> None:
        self.symbols, self.timeframes = symbols, timeframes
        self.idea_dir, self.notes_path, self.log_dir = Path(idea_dir), Path(notes_path), Path(log_dir)
        self.roles = roles or list(ROLES)
        self.n_ideas, self.timeout, self.model, self.db = n_ideas, timeout, model, db
        if backend == "auto":
            backend = "claude-code" if shutil.which("claude") else "api"
        self.backend = backend
        self._api = None
        self._i = 0

    # ------------------------------------------------------------------ prompting
    def context(self, role: str) -> str:
        notes = self.notes_path.read_text()[:12000] if self.notes_path.exists() else "(no notes yet)"
        return (f"{SYSTEM}\n\n{ROLES[role]}\n\nSymbols: {self.symbols}. Timeframes: {self.timeframes}.\n"
                "Only real market data is used; strategies must be profitable over 10+ years, survive Monte Carlo "
                "resampling, beat random entries, and be stable to +/-15% parameter changes.\n\n"
                f"Research notes so far:\n{notes}\n\nBlock catalogue (params: [type, lo, hi] or list of choices):\n"
                f"{catalogue()}\n\n{CONTRACT.format(n=self.n_ideas)}")

    def ask(self, role: str) -> str:
        ctx = self.context(role)
        if self.backend == "claude-code":
            cmd = ["claude", "-p", "Research and propose strategies following the brief on stdin.",
                   "--output-format", "text", "--allowedTools", "WebSearch", "WebFetch"]
            if self.model:
                cmd += ["--model", self.model]
            r = subprocess.run(cmd, input=ctx, capture_output=True, text=True, timeout=self.timeout)
            if r.returncode != 0:
                raise RuntimeError(f"claude CLI failed: {r.stderr[-500:]}")
            return r.stdout
        if self._api is None:
            from propfloor.factory.llm import ClaudeResearcher
            self._api = ClaudeResearcher(self.db)
        return self._api._ask(ctx, f"agent:{role}")

    # ------------------------------------------------------------------ running
    def run_role(self, role: str) -> int:
        text = self.ask(role)
        ideas = parse_genomes(text)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / f"{role}.md").write_text(text[-4000:])  # latest reply only: stays small
        if not ideas:
            return 0
        self.idea_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.idea_dir / f".{role}-{int(time.time())}.tmp"
        tmp.write_text(json.dumps([g.to_dict() for g in ideas]))
        tmp.rename(tmp.with_name(tmp.name[1:].replace(".tmp", ".json")))  # atomic: the factory never sees half a file
        log.info("agent %s queued %d ideas", role, len(ideas))
        return len(ideas)

    def run_next(self) -> int:
        role = self.roles[self._i % len(self.roles)]
        self._i += 1
        return self.run_role(role)

    def loop(self, stop: threading.Event, every_min: float = 30.0) -> None:
        while not stop.is_set():
            try:
                self.run_next()
            except Exception as e:  # noqa: BLE001 - agents are optional helpers; never kill the factory
                log.warning("research agent failed: %s", e)
            stop.wait(every_min * 60)

    def start(self, every_min: float = 30.0) -> threading.Event:
        stop = threading.Event()
        threading.Thread(target=self.loop, args=(stop, every_min), daemon=True, name="research-agents").start()
        return stop
