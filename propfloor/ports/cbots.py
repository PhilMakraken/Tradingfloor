"""Ports of the user's cTrader cBots (Combo 7 XAUUSD bots 4-7) for out-of-sample replays on M15 data.

Each ``run_botN`` takes UTC-indexed M15 bid bars (open/high/low/close[/volume]) and returns a trade DataFrame with
columns entry_time, exit_time, side, entry, exit, pnl_r, net (+ units, stop_dist, balance, n_closes, reason, info).
``df.attrs["equity"]`` holds the mark-to-market equity at every M15 close (for drawdown / daily P&L).

Execution model (cTrader has tick data, we only have M15 bars):
  * Signal bars (H1/H2/H4/H12/D1) are built from M15 on the broker clock = New York + 7h (00:00 = 17:00 NY roll),
    which is what the user's 2024-26 report shows (H4 entries at 01/05/09/13/17/21 UTC in summer, 02/06/.. in winter).
  * "BarOpened" logic runs on the first M15 bar of a new signal bar, using only completed signal bars; market
    orders fill at that M15 open (ask = bid + spread for buys) plus slippage.
  * Intrabar ticks are approximated by the path open -> low -> high -> close for an up bar (close >= open) and
    open -> high -> low -> close otherwise. Stops, targets, partials, breakevens and tick-driven entries (bot 6) are
    evaluated on that path in order; a jump between the previous close and the next open is treated as a gap
    (fills at the open).
  * Costs from the propfloor program (programs/generic_2stage.yaml): spread_pips, slippage_pips (each fill, adverse),
    commission_per_lot round turn; XAUUSD pip = 0.1 (propfloor convention), 1 lot = 100 oz, volume step 1 oz.
    Swaps are not modelled. The account currency is taken as USD (the user's reports are in GBP).
  * Portfolio stop rules: as in the C# when not live (LocalStorage only used live) - each bot tracks its own
    account's peak equity (heartbeat hourly + whenever an order is sized): >= 12% DD -> half risk until a new
    equity high, >= 20% DD -> no new trades for the rest of the run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

UNIT_STEP = 1.0  # cTrader XAUUSD: min 0.01 lot = 1 oz, step 1 oz
CONTRACT = 100.0  # oz per lot
TICK = 0.01

# ---------------------------------------------------------------------------------------------------------------
# Default parameters = the exact values from research/user_reports/parsed/C7N_N_params.json (risk 1%).
# ---------------------------------------------------------------------------------------------------------------
PORTFOLIO = {"Use Portfolio Stop Rules": True, "Half Risk At Account Drawdown %": 12.0,
             "Pause New Trades At Account Drawdown %": 20.0}

BOT4_PARAMS = {"Signal Timeframe": "h2", "Entry Mode": "SwingHLLH", "Donchian Period": 20,
               "Swing Strength (bars each side)": 4, "Allow Long": True, "Allow Short": False,
               "Use EMA Filter": True, "EMA Period": 75, "Use MACD Filter": True, "MACD Fast": 12, "MACD Slow": 26,
               "MACD Signal": 9, "ATR Period": 14, "Initial Stop (x ATR)": 1.5, "Partial Close %": 33,
               "Partial Target (R)": 2.0, "Breakeven After Partial": True, "Trailing Stop (x ATR, 0 = off)": 10.0,
               "Sizing Mode": "RiskPercent", "Risk %": 1.0, "Fixed Lots": 0.02, **PORTFOLIO}
BOT5_PARAMS = {"Signal Timeframe": "h4", "EMA Period": 40, "Use Daily EMA200 Trend Filter": True,
               "Allow Mondays": True, "Allow Short": False, "Take Profit (R multiple)": 2.0, "Risk % per trade": 1.0,
               "Time Exit (days, safety cap)": 30, **PORTFOLIO}
BOT6_PARAMS = {"Signal Timeframe": "h1", "Breakout Lookback (bars)": 24, "Fast EMA": 50, "Slow EMA": 200,
               "Use Daily EMA200 Trend Filter": True, "Stop Buffer (x ATR14)": 0.4, "Partial Close %": 80,
               "Partial Target (R)": 1.5, "Trailing Stop (x ATR14)": 2.5, "Risk % per trade": 1.0,
               "Time Exit (days, safety cap)": 30, **PORTFOLIO}
BOT7_PARAMS = {"Entry Timeframe": "h4", "MACD Fast": 12, "MACD Slow": 26, "MACD Signal": 9,
               "Filter Timeframe": "h12", "DEMA Period": 100, "ATR Period": 14, "Stop Loss (x ATR, 0 = off)": 2.0,
               "Take Profit (x ATR, 0 = off)": 3.0, "Risk % per trade (0 = fixed lots)": 1.0,
               "Fixed Lots (if risk = 0)": 0.1, "Enable Shorts (untested)": False, **PORTFOLIO}
DEFAULT_PARAMS = {4: BOT4_PARAMS, 5: BOT5_PARAMS, 6: BOT6_PARAMS, 7: BOT7_PARAMS}


def _p(params: dict | None, defaults: dict) -> dict:
    out = dict(defaults)
    for k, v in (params or {}).items():
        if isinstance(v, str) and v.lower() in ("true", "false"):
            v = v.lower() == "true"
        out[k] = v
    return out


# ---------------------------------------------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Costs:
    spread: float = 0.10  # price units (1.0 pip x 0.1)
    slippage: float = 0.03  # per fill, adverse (0.3 pip)
    commission_per_unit: float = 0.07  # round turn per oz ($7 per 100-oz lot)

    @classmethod
    def from_program(cls, program=None, symbol: str = "XAUUSD") -> "Costs":
        from propfloor.config import DEFAULT_PROGRAM, load_program
        from propfloor.data.store import pip_size
        prog = program if program is not None and not isinstance(program, str) else load_program(program or DEFAULT_PROGRAM)
        ps = pip_size(symbol)
        c = prog.costs
        return cls(c.spread_pips * ps, c.slippage_pips * ps, c.commission_per_lot / CONTRACT)


ZERO_COSTS = Costs(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------------------------------------------
# Bars / indicators
# ---------------------------------------------------------------------------------------------------------------
def server_clock(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC -> broker clock (New York + 7h, naive)."""
    return idx.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)


TF_HOURS = {"M15": 0.25, "M30": 0.5, "H1": 1, "H2": 2, "H4": 4, "H12": 12, "D1": 24}


class Frames:
    """M15 arrays + signal-timeframe bars built on the broker clock, with M15 -> bar index maps."""

    def __init__(self, m15: pd.DataFrame) -> None:
        m15 = m15.sort_index()
        self.t = m15.index
        self.o = m15["open"].to_numpy(float)
        self.h = m15["high"].to_numpy(float)
        self.l = m15["low"].to_numpy(float)
        self.c = m15["close"].to_numpy(float)
        self.v = m15["volume"].to_numpy(float) if "volume" in m15 else np.ones(len(m15))
        self.srv = server_clock(self.t)
        self.n = len(m15)
        self._cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

    def bars(self, tf: str) -> tuple[pd.DataFrame, np.ndarray]:
        """(bars indexed by broker-clock open time, bar id of every M15 bar)."""
        tf = tf.upper()
        if tf not in self._cache:
            hrs = TF_HOURS[tf]
            key = self.srv.normalize() if hrs == 24 else self.srv.floor(f"{int(hrs * 60)}min")
            codes, uniq = pd.factorize(key)  # key is monotone -> codes increase
            df = pd.DataFrame({"o": self.o, "h": self.h, "l": self.l, "c": self.c, "v": self.v, "k": codes})
            g = df.groupby("k", sort=True)
            b = pd.DataFrame({"open": g["o"].first(), "high": g["h"].max(), "low": g["l"].min(),
                              "close": g["c"].last(), "volume": g["v"].sum()})
            b.index = pd.DatetimeIndex(uniq)
            self._cache[tf] = (b, codes.astype(np.int64))
        return self._cache[tf]


def ema_seeded(x: np.ndarray, n: int) -> np.ndarray:
    """EMA seeded with the first value (cTrader EMA / the bots' hand-coded EMAs)."""
    out = np.empty(len(x))
    a = 2.0 / (n + 1)
    for i in range(len(x)):
        out[i] = x[0] if i == 0 else out[i - 1] + a * (x[i] - out[i - 1])
    return out


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h, pc) - np.minimum(l, pc)
    tr[0] = h[0] - l[0]
    return tr


def atr_smoothed(h, l, c, alpha: float) -> np.ndarray:
    tr = true_range(h, l, c)
    out = np.empty(len(tr))
    for i in range(len(tr)):
        out[i] = tr[0] if i == 0 else out[i - 1] + alpha * (tr[i] - out[i - 1])
    return out


def daily_trend(daily_close: np.ndarray, n: int = 200, min_idx: int = 210) -> np.ndarray:
    """+1/-1 = completed daily close above/below its EMA(n) (seeded), 0 while fewer than min_idx bars."""
    e = ema_seeded(daily_close, n)
    tr = np.where(daily_close > e, 1, -1)
    tr[: min_idx] = 0
    return tr


# ---------------------------------------------------------------------------------------------------------------
# Account / position engine
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Pos:
    dir: int
    units: float
    entry: float
    sl: float | None
    tp: float | None
    t_entry: pd.Timestamp
    i_sig: int
    stop_dist: float
    units0: float = 0.0
    partial_level: float | None = None
    partial_done: bool = True
    realized: float = 0.0
    exit_value: float = 0.0
    n_closes: int = 0
    best: float = 0.0
    entry_atr: float = 0.0
    info: dict = field(default_factory=dict)


class Book:
    """One cTrader account running one bot: balance, the bot's single position, portfolio stop rules."""

    def __init__(self, costs: Costs, balance: float, params: dict, partial_pct: float = 0.0,
                 breakeven: bool = False) -> None:
        self.c = costs
        self.balance = float(balance)
        self.use_rules = bool(params.get("Use Portfolio Stop Rules", True))
        self.half_dd = float(params.get("Half Risk At Account Drawdown %", 12.0))
        self.pause_dd = float(params.get("Pause New Trades At Account Drawdown %", 20.0))
        self.peak, self.half, self.pause = 0.0, False, False
        self.partial_pct, self.breakeven = float(partial_pct), bool(breakeven)
        self.partial_be_r = 0.0  # stop after partial = entry + be_r x initial risk (SRC "Stop after partial")
        self.pos: Pos | None = None
        self.trades: list[dict] = []
        self.half_trades = 0
        self.paused_at: pd.Timestamp | None = None

    # --- prices: x = favourable coordinate of the exit-side price (long: bid, short: -(ask))
    def xs(self, bid: float, d: int) -> float:
        return d * (bid + (self.c.spread if d < 0 else 0.0))

    def unreal(self, bid: float) -> float:
        p = self.pos
        if p is None:
            return 0.0
        px = bid if p.dir > 0 else bid + self.c.spread
        return p.dir * (px - p.entry) * p.units

    def equity(self, bid: float) -> float:
        return self.balance + self.unreal(bid)

    # --- portfolio stop rules (UpdatePortfolioState)
    def update_state(self, bid: float, t=None) -> None:
        if not self.use_rules:
            return
        eq = self.equity(bid)
        if eq >= self.peak:
            self.peak, self.half = eq, False
        dd = 1 - eq / self.peak if self.peak > 0 else 0.0
        if dd * 100 >= self.half_dd:
            self.half = True
        if dd * 100 >= self.pause_dd and not self.pause:
            self.pause, self.paused_at = True, t

    def paused(self, bid: float, t=None) -> bool:
        if not self.use_rules:
            return False
        self.update_state(bid, t)
        return self.pause

    def eff_risk(self, risk_pct: float, bid: float, t=None) -> float:
        if not self.use_rules:
            return risk_pct
        self.update_state(bid, t)
        return risk_pct * 0.5 if self.half else risk_pct

    def units_for(self, risk_pct: float, stop_dist: float) -> float:
        if not stop_dist > 0:
            return 0.0
        return math.floor(self.balance * risk_pct / 100.0 / stop_dist / UNIT_STEP + 1e-9) * UNIT_STEP

    # --- orders
    def open(self, d: int, t, bid: float, units: float, stop_dist: float | None, tp_dist: float | None,
             i_sig: int, partial_r: float | None = None, info: dict | None = None, half: bool = False) -> Pos:
        fill = self.market_fill(d, bid)
        sl = round(fill - d * stop_dist, 2) if stop_dist else None
        tp = round(fill + d * tp_dist, 2) if tp_dist else None
        self.half_trades += int(half)
        return self.open_fill(d, t, fill, units, sl, tp, i_sig, stop_dist or 0.0, partial_r, info)

    def market_fill(self, d: int, bid: float) -> float:
        return (bid + self.c.spread + self.c.slippage) if d > 0 else (bid - self.c.slippage)

    def open_fill(self, d: int, t, fill: float, units: float, sl: float | None, tp: float | None, i_sig: int,
                  stop_dist: float, partial_r: float | None = None, info: dict | None = None) -> Pos:
        """Open at a known fill price with absolute stop / target prices."""
        p = Pos(d, units, fill, sl, tp, t, i_sig, stop_dist, units0=units, best=fill, info=info or {})
        if partial_r is not None and stop_dist:
            p.partial_level, p.partial_done = fill + d * partial_r * stop_dist, False
        self.pos = p
        return p

    def close(self, units: float, x: float, t, reason: str) -> None:
        """Close `units` at favourable-coordinate exit-side price x (slippage applied here)."""
        p = self.pos
        units = min(units, p.units)
        fill = p.dir * x - p.dir * self.c.slippage
        pnl = p.dir * (fill - p.entry) * units - self.c.commission_per_unit * units
        self.balance += pnl
        p.realized += pnl
        p.exit_value += fill * units
        p.n_closes += 1
        p.units -= units
        if p.units <= 1e-9:
            risk_money = p.units0 * p.stop_dist
            self.trades.append({
                "entry_time": p.t_entry, "exit_time": t, "side": "long" if p.dir > 0 else "short",
                "entry": p.entry, "exit": p.exit_value / p.units0, "units": p.units0, "stop_dist": p.stop_dist,
                "pnl_r": p.realized / risk_money if risk_money > 0 else np.nan, "net": p.realized,
                "balance": self.balance, "n_closes": p.n_closes, "partial": p.partial_level is not None and p.partial_done
                and p.n_closes > 1, "reason": reason, "info": p.info})
            self.pos = None

    def close_all(self, bid: float, t, reason: str) -> None:
        if self.pos is not None:
            self.close(self.pos.units, self.xs(bid, self.pos.dir), t, reason)

    def do_partial(self, x: float, t) -> None:
        p = self.pos
        p.partial_done = True
        if self.partial_pct >= 100:
            self.close(p.units, x, t, "partial100")
            return
        cv = math.floor(p.units * self.partial_pct / 100.0 / UNIT_STEP + 1e-9) * UNIT_STEP
        if cv >= UNIT_STEP and p.units - cv >= UNIT_STEP:
            self.close(cv, x, t, "partial")
        if self.breakeven and self.pos is not None:
            be = round(p.entry + p.dir * self.partial_be_r * p.stop_dist, 2)
            if p.sl is None or p.dir * (be - p.sl) > 0:
                p.sl = be

    def segment(self, b0: float, b1: float, t, gap: bool = False) -> None:
        """Price (bid) moves b0 -> b1; fire stop / partial / target in path order."""
        p = self.pos
        if p is None or b0 == b1:
            return
        d = p.dir
        x0, x1 = self.xs(b0, d), self.xs(b1, d)
        if x1 < x0:
            if p.sl is not None and x1 <= d * p.sl:
                self.close(p.units, x1 if gap else min(x0, d * p.sl), t, "sl" if p.n_closes == 0 else "sl_after_partial")
            return
        lv = []
        if p.partial_level is not None and not p.partial_done:
            lv.append((d * p.partial_level, 0))
        if p.tp is not None:
            lv.append((d * p.tp, 1))
        for level, kind in sorted(lv):
            if x1 >= level and self.pos is not None:
                px = x1 if gap else max(x0, level)
                if kind == 1:
                    self.close(self.pos.units, px, t, "tp")
                else:
                    self.do_partial(px, t)


def _path(o: float, h: float, l: float, c: float) -> tuple[float, float, float, float]:
    return (o, l, h, c) if c >= o else (o, h, l, c)


def _finish(book: Book, fr: Frames, eq: np.ndarray, trade_start) -> pd.DataFrame:
    if book.pos is not None:
        book.close_all(fr.c[-1], fr.t[-1], "end_of_data")
        eq[-1] = book.balance
    df = pd.DataFrame(book.trades, columns=["entry_time", "exit_time", "side", "entry", "exit", "pnl_r", "net", "units",
                                            "stop_dist", "balance", "n_closes", "partial", "reason", "info"])
    for col in ("entry_time", "exit_time"):
        df[col] = pd.to_datetime(df[col], utc=True).astype("datetime64[ns, UTC]")
    m = np.ones(fr.n, bool) if trade_start is None else np.asarray(fr.t >= pd.Timestamp(trade_start, tz="UTC"))
    df.attrs["equity"] = pd.Series(eq[m], index=fr.t[m])
    df.attrs["half_risk_trades"] = book.half_trades
    df.attrs["paused_at"] = book.paused_at
    return df


def _start_mask(fr: Frames, trade_start) -> np.ndarray:
    if trade_start is None:
        return np.ones(fr.n, bool)
    return np.asarray(fr.t >= pd.Timestamp(trade_start, tz="UTC"))


# ---------------------------------------------------------------------------------------------------------------
# Bot 4: Swing Partial Bot G v3 (XAUUSD H2)
# ---------------------------------------------------------------------------------------------------------------
def swing_signals(high: np.ndarray, low: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """UpdateSwings for every closed bar i: (hl_confirmed, lh_confirmed, pivot low value, previous swing low)."""
    n = len(high)
    hl, lh = np.zeros(n, bool), np.zeros(n, bool)
    piv, prev = np.full(n, np.nan), np.full(n, np.nan)
    psl = psh = np.nan
    for i in range(n):
        j = i - k
        if j - k < 0:
            continue
        hw, lw = high[j - k: j + k + 1], low[j - k: j + k + 1]
        if not (hw > high[j]).any():
            lh[i] = not np.isnan(psh) and high[j] < psh
            psh = high[j]
        if not (lw < low[j]).any():
            hl[i] = not np.isnan(psl) and low[j] > psl
            piv[i], prev[i] = low[j], psl
            psl = low[j]
    return hl, lh, piv, prev


def run_bot4(m15: pd.DataFrame, params: dict | None = None, costs: Costs | None = None,
             balance: float = 10_000.0, trade_start=None) -> pd.DataFrame:
    P = _p(params, BOT4_PARAMS)
    costs = costs or Costs.from_program()
    fr = Frames(m15)
    b, bid_ = fr.bars(P["Signal Timeframe"].upper())
    H, L, C = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
    ema = ema_seeded(C, int(P["EMA Period"]))
    macd = ema_seeded(C, int(P["MACD Fast"])) - ema_seeded(C, int(P["MACD Slow"]))
    sig = ema_seeded(macd, int(P["MACD Signal"]))
    atr = atr_smoothed(H, L, C, 1.0 / int(P["ATR Period"]))  # Wilder smoothing
    hl, lh, piv, prev = swing_signals(H, L, int(P["Swing Strength (bars each side)"]))
    don = P["Entry Mode"] == "DonchianBreakout"
    dp = int(P["Donchian Period"])
    book = Book(costs, balance, P, partial_pct=float(P["Partial Close %"]), breakeven=P["Breakeven After Partial"])
    risk_pct, stop_atr, part_r, trail_atr = float(P["Risk %"]), float(P["Initial Stop (x ATR)"]), \
        float(P["Partial Target (R)"]), float(P["Trailing Stop (x ATR, 0 = off)"])
    ok = _start_mask(fr, trade_start)
    eq = np.empty(fr.n)

    def filt(i: int, d: int) -> bool:
        if P["Use EMA Filter"] and not (C[i] > ema[i] if d == 1 else C[i] < ema[i]):
            return False
        if P["Use MACD Filter"] and not (macd[i] > sig[i] if d == 1 else macd[i] < sig[i]):
            return False
        return True

    for j in range(fr.n):
        o, h, l, c, t = fr.o[j], fr.h[j], fr.l[j], fr.c[j], fr.t[j]
        if not ok[j]:
            eq[j] = book.balance
            continue
        if j > 0:
            book.segment(fr.c[j - 1], o, t, gap=True)
        if t.minute == 0:
            book.update_state(o, t)  # heartbeat timer (hourly)
        if j > 0 and bid_[j] != bid_[j - 1]:  # BarOpened
            i = bid_[j] - 1
            p = book.pos
            if p is not None:
                if trail_atr > 0 and p.i_sig <= i:
                    d = p.dir
                    best = p.entry
                    for k in range(p.i_sig, i + 1):
                        best = max(best, H[k]) if d == 1 else min(best, L[k])
                    new_sl = round(best - d * trail_atr * atr[i], 2)
                    market = o if d == 1 else o + costs.spread
                    if (p.sl is None or d * (new_sl - p.sl) > 0) and d * (market - new_sl) > 0:
                        p.sl = new_sl
            elif i >= 250:
                if don:
                    hh, ll = H[i - dp:i].max(), L[i - dp:i].min()
                    ls, ss = C[i] > hh, C[i] < ll
                else:
                    ls, ss = hl[i], lh[i]
                ls = ls and P["Allow Long"] and filt(i, 1)
                ss = ss and P["Allow Short"] and filt(i, -1)
                if ls != ss and not book.paused(o, t):
                    d = 1 if ls else -1
                    a = atr[i]
                    if a > 0:
                        risk = stop_atr * a
                        er = book.eff_risk(risk_pct, o, t)
                        units = (math.floor(P["Fixed Lots"] * CONTRACT) if P["Sizing Mode"] == "FixedLots"
                                 else book.units_for(er, risk))
                        if units >= UNIT_STEP:
                            info = {"signal_bar": str(b.index[i]), "close": C[i], "ema": ema[i], "macd": macd[i],
                                    "macd_signal": sig[i], "atr": a, "swing_low": piv[i], "prev_swing_low": prev[i]}
                            book.open(d, t, o, units, risk, None, i + 1, partial_r=part_r if book.partial_pct > 0
                                      else None, info=info, half=er < risk_pct)
        for a0, a1 in zip(_path(o, h, l, c)[:-1], _path(o, h, l, c)[1:]):
            book.segment(a0, a1, t)
        eq[j] = book.equity(c)
    return _finish(book, fr, eq, trade_start)


# ---------------------------------------------------------------------------------------------------------------
# Bot 5: Gold Wick v3 (XAUUSD H4)
# ---------------------------------------------------------------------------------------------------------------
def run_bot5(m15: pd.DataFrame, params: dict | None = None, costs: Costs | None = None,
             balance: float = 10_000.0, trade_start=None) -> pd.DataFrame:
    P = _p(params, BOT5_PARAMS)
    costs = costs or Costs.from_program()
    fr = Frames(m15)
    b, bid_ = fr.bars(P["Signal Timeframe"].upper())
    d1, did = fr.bars("D1")
    H, L, C = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
    ema = ema_seeded(C, int(P["EMA Period"]))
    dtr = daily_trend(d1["close"].to_numpy())
    dema200 = ema_seeded(d1["close"].to_numpy(), 200)
    book = Book(costs, balance, P)
    risk_pct, tr_r, cap = float(P["Risk % per trade"]), float(P["Take Profit (R multiple)"]), \
        pd.Timedelta(days=float(P["Time Exit (days, safety cap)"]))
    ok = _start_mask(fr, trade_start)
    eq = np.empty(fr.n)
    for j in range(fr.n):
        o, h, l, c, t = fr.o[j], fr.h[j], fr.l[j], fr.c[j], fr.t[j]
        if not ok[j]:
            eq[j] = book.balance
            continue
        if j > 0:
            book.segment(fr.c[j - 1], o, t, gap=True)
        if t.minute == 0:
            book.update_state(o, t)
        if book.pos is not None and t - book.pos.t_entry >= cap:
            book.close_all(o, t, "time_cap")
        if j > 0 and bid_[j] != bid_[j - 1] and book.pos is None:
            s = bid_[j] - 1
            if s >= 60:
                p_ = s - 1
                buy = L[s] < L[p_] and C[s] > H[p_] and C[s] > ema[s]
                sell = P["Allow Short"] and H[s] > H[p_] and C[s] < L[p_] and C[s] < ema[s]
                monday = (b.index[s] - pd.Timedelta(hours=2)).dayofweek == 0
                if (P["Allow Mondays"] or not monday) and not (buy and sell):
                    jd = did[j] - 1
                    if P["Use Daily EMA200 Trend Filter"]:
                        tr = dtr[jd] if jd >= 0 else 0
                        if tr == 0:
                            buy = sell = False
                        if buy and tr < 0:
                            buy = False
                        if sell and tr > 0:
                            sell = False
                    if buy or sell:
                        d = 1 if buy else -1
                        dist = (o + costs.spread - L[s]) if buy else (H[s] - o)
                        if not book.paused(o, t) and dist > 0:
                            er = book.eff_risk(risk_pct, o, t)
                            units = book.units_for(er, dist)
                            if units >= UNIT_STEP:
                                info = {"signal_bar": str(b.index[s]), "low_s": L[s], "low_p": L[p_], "close_s": C[s],
                                        "high_p": H[p_], "ema40": ema[s], "daily_bar": str(d1.index[jd]) if jd >= 0 else None,
                                        "daily_close": d1["close"].iloc[jd] if jd >= 0 else None,
                                        "daily_ema200": dema200[jd] if jd >= 0 else None}
                                book.open(d, t, o, units, dist, dist * tr_r, s + 1, info=info, half=er < risk_pct)
        for a0, a1 in zip(_path(o, h, l, c)[:-1], _path(o, h, l, c)[1:]):
            book.segment(a0, a1, t)
        eq[j] = book.equity(c)
    return _finish(book, fr, eq, trade_start)


# ---------------------------------------------------------------------------------------------------------------
# Bot 6: Breakout Trend Runner v3 (XAUUSD H1, tick-driven)
# ---------------------------------------------------------------------------------------------------------------
def run_bot6(m15: pd.DataFrame, params: dict | None = None, costs: Costs | None = None,
             balance: float = 10_000.0, trade_start=None) -> pd.DataFrame:
    P = _p(params, BOT6_PARAMS)
    costs = costs or Costs.from_program()
    fr = Frames(m15)
    b, bid_ = fr.bars(P["Signal Timeframe"].upper())
    d1, did = fr.bars("D1")
    H, L, C = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
    ef, es = ema_seeded(C, int(P["Fast EMA"])), ema_seeded(C, int(P["Slow EMA"]))
    atr = atr_smoothed(H, L, C, 2.0 / 15.0)  # EMA-smoothed TR, as hand-coded in the bot
    dtr = daily_trend(d1["close"].to_numpy())
    lb = int(P["Breakout Lookback (bars)"])
    hh_arr = pd.Series(H).rolling(lb).max().to_numpy()  # bars last-lb+1..last
    ll_arr = pd.Series(L).rolling(lb).min().to_numpy()
    book = Book(costs, balance, P, partial_pct=float(P["Partial Close %"]), breakeven=True)
    risk_pct, buf, part_r, trail_k = float(P["Risk % per trade"]), float(P["Stop Buffer (x ATR14)"]), \
        float(P["Partial Target (R)"]), float(P["Trailing Stop (x ATR14)"])
    cap = pd.Timedelta(days=float(P["Time Exit (days, safety cap)"]))
    use_daily = bool(P["Use Daily EMA200 Trend Filter"])
    ok = _start_mask(fr, trade_start)
    eq = np.empty(fr.n)
    last_level = np.nan
    blocked = 0

    def trail(bid: float) -> None:
        """Manage(): trail = best - k x ATR(entry) if it tightens the stop and is below the bid; then best = max."""
        p = book.pos
        if p is None:
            return
        p.best = max(p.best, bid)  # continuous path: the tick before the segment end is ~ the segment end
        tr = round(p.best - trail_k * p.entry_atr, 2)
        if p.sl is not None and tr > p.sl + TICK and tr < bid:
            p.sl = tr

    def unused(hh: float) -> bool:
        return np.isnan(last_level) or abs(last_level - hh) > TICK / 2

    def breakout(j: int, t, ebid: float, last: int, hh: float) -> None:
        nonlocal last_level, blocked
        last_level = hh  # level used once, even if a trade is open or the daily filter blocks it
        daily_ok = (not use_daily) or (did[j] - 1 >= 0 and dtr[did[j] - 1] == 1)
        if book.pos is not None:
            return
        if not daily_ok:
            blocked += 1
            return
        if book.paused(ebid, t):
            return
        stop = ll_arr[last] - buf * atr[last]
        dist = ebid + costs.spread - stop
        if not dist > 0:
            return
        er = book.eff_risk(risk_pct, ebid, t)
        units = book.units_for(er, dist)
        if units < UNIT_STEP:
            return
        info = {"h1_bar": str(b.index[last]), "hh24": hh, "ll24": ll_arr[last], "ema50": ef[last],
                "ema200": es[last], "atr": atr[last], "daily_trend": int(dtr[did[j] - 1]), "entry_bid": ebid}
        p = book.open(1, t, ebid, units, dist, None, last + 1, partial_r=part_r, info=info, half=er < risk_pct)
        p.entry_atr = atr[last]

    for j in range(fr.n):
        o, h, l, c, t = fr.o[j], fr.h[j], fr.l[j], fr.c[j], fr.t[j]
        if not ok[j]:
            eq[j] = book.balance
            continue
        last = bid_[j] - 1
        armed = last >= 250 and last >= lb - 1 and ef[last] > es[last] and atr[last] > 0
        hh = hh_arr[last] if last >= 0 else np.nan
        if j > 0 and book.pos is not None:  # gap from the previous close to this open (server-side stops)
            book.segment(fr.c[j - 1], o, t, gap=True)
            trail(o)
        if t.minute == 0:
            book.update_state(o, t)
        if book.pos is not None and t - book.pos.t_entry >= cap:
            book.close_all(o, t, "time_cap")
        if armed and o > hh and unused(hh):  # tick at the open
            breakout(j, t, o, last, hh)
        pts = _path(o, h, l, c)
        for b0, b1 in zip(pts[:-1], pts[1:]):
            if armed and b0 <= hh < b1 and unused(hh):
                e = min(round(hh + TICK, 2), b1)
                if book.pos is not None:
                    book.segment(b0, e, t)
                    trail(e)
                breakout(j, t, e, last, hh)
                b0 = e
            if book.pos is not None:
                book.segment(b0, b1, t)
                trail(b1)
        eq[j] = book.equity(c)
    out = _finish(book, fr, eq, trade_start)
    out.attrs["breakouts_blocked_by_daily"] = blocked
    return out


# ---------------------------------------------------------------------------------------------------------------
# Bot 7: XAU MACD-histogram + DEMA trend v4 (H4 entries, H12 filter)
# ---------------------------------------------------------------------------------------------------------------
def macd_hist_state(C: np.ndarray, H: np.ndarray, L: np.ndarray, fast: int, slow: int, sig_n: int, atr_n: int,
                    shorts: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """UpdateEntryIndicators over all bars: (hist, atr, signal, leg)."""
    n = len(C)
    ef, es = ema_seeded(C, fast), ema_seeded(C, slow)
    macd = ef - es
    sg = np.empty(n)
    sg[0] = 0.0
    a = 2.0 / (sig_n + 1)
    for i in range(1, n):
        sg[i] = sg[i - 1] + a * (macd[i] - sg[i - 1])
    hist = macd - sg
    hist[0] = 0.0
    atr = atr_smoothed(H, L, C, 1.0 / atr_n)
    sig = np.zeros(n, np.int64)
    leg = np.zeros(n, np.int64)
    state, lg = 0, 0
    for i in range(1, n):
        if i >= 2:
            if hist[i] > hist[i - 1] > hist[i - 2]:
                state = 1
            elif hist[i] < hist[i - 1] < hist[i - 2]:
                state = -1
        s = state if shorts else (1 if state == 1 else 0)
        sig[i] = s
        if s != sig[i - 1]:
            lg += 1
        leg[i] = lg
    return hist, atr, sig, leg


def dema(x: np.ndarray, n: int) -> np.ndarray:
    e1 = ema_seeded(x, n)
    e2 = ema_seeded(e1, n)
    out = 2 * e1 - e2
    out[0] = x[0]
    return out


def run_bot7(m15: pd.DataFrame, params: dict | None = None, costs: Costs | None = None,
             balance: float = 10_000.0, trade_start=None) -> pd.DataFrame:
    P = _p(params, BOT7_PARAMS)
    costs = costs or Costs.from_program()
    fr = Frames(m15)
    b, bid_ = fr.bars(P["Entry Timeframe"].upper())
    f, fid = fr.bars(P["Filter Timeframe"].upper())
    H, L, C = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
    shorts = bool(P["Enable Shorts (untested)"])
    hist, atr, sig, leg = macd_hist_state(C, H, L, int(P["MACD Fast"]), int(P["MACD Slow"]), int(P["MACD Signal"]),
                                          int(P["ATR Period"]), shorts)
    dn = int(P["DEMA Period"])
    fd = dema(f["close"].to_numpy(), dn)
    book = Book(costs, balance, P)
    risk_pct = float(P["Risk % per trade (0 = fixed lots)"])
    sl_k, tp_k = float(P["Stop Loss (x ATR, 0 = off)"]), float(P["Take Profit (x ATR, 0 = off)"])
    ok = _start_mask(fr, trade_start)
    eq = np.empty(fr.n)
    traded_leg = -1
    for j in range(fr.n):
        o, h, l, c, t = fr.o[j], fr.h[j], fr.l[j], fr.c[j], fr.t[j]
        if not ok[j]:
            eq[j] = book.balance
            continue
        if j > 0:
            book.segment(fr.c[j - 1], o, t, gap=True)
        if t.minute == 0:
            book.update_state(o, t)
        if j > 0 and bid_[j] != bid_[j - 1]:
            i = bid_[j] - 1
            if i >= 100:
                last = fid[j] - 1  # last H12 bar with open + 12h <= decision time
                filt = 0
                if last >= dn * 2:
                    dd_ = fd[last] - fd[last - 1]
                    filt = 1 if dd_ > 0 else (-1 if dd_ < 0 else 0)
                s = int(sig[i])
                if book.pos is not None and s != book.pos.dir:
                    book.close_all(o, t, "signal_flat")
                if book.pos is None and s != 0 and filt == s and traded_leg != leg[i]:
                    if not book.paused(o, t):
                        sl_d = round(sl_k * atr[i] / TICK, 1) * TICK if sl_k > 0 else 0.0
                        tp_d = round(tp_k * atr[i] / TICK, 1) * TICK if tp_k > 0 else 0.0
                        er = book.eff_risk(risk_pct, o, t)
                        if risk_pct > 0 and sl_d > 0:
                            units = book.units_for(er, sl_d)
                        else:
                            units = math.floor(P["Fixed Lots (if risk = 0)"] * CONTRACT)
                        units = max(units, UNIT_STEP)  # bot uses the broker minimum if below it
                        info = {"h4_bar": str(b.index[i]), "hist": hist[i], "hist_1": hist[i - 1], "hist_2": hist[i - 2],
                                "leg": int(leg[i]), "h12_bar": str(f.index[last]), "dema": fd[last],
                                "dema_prev": fd[last - 1], "atr": atr[i]}
                        book.open(s, t, o, units, sl_d or None, tp_d or None, i + 1, info=info, half=er < risk_pct)
                        if sl_d == 0:
                            book.pos.stop_dist = 2.0 * atr[i]  # only for R accounting
                        traded_leg = leg[i]
        for a0, a1 in zip(_path(o, h, l, c)[:-1], _path(o, h, l, c)[1:]):
            book.segment(a0, a1, t)
        eq[j] = book.equity(c)
    return _finish(book, fr, eq, trade_start)


# ---------------------------------------------------------------------------------------------------------------
# SRC round-5 cBots (SrcV1LondonMorning, SrcV2Main, SrcV8MarketEntry): M15 APPROXIMATION of M5 bots
# ---------------------------------------------------------------------------------------------------------------
# The three files share one engine; only defaults and FiltersAllow differ. The originals run on M5 bars with
# UK-clock windows. Here the chart bars are M15: every M5 bar count (range lookback, anchor lookback, limit
# validity, time stop, pivot strength, ATR and volume periods) is divided by 3, and the M5 ATR(14) is approximated
# by RMA(TR_M15, 5) / sqrt(3) (random-walk scaling; our gold data gives TR(H1)/TR(M15) = 4^0.52, i.e. M15/M5 ~1.78).
SRC_ENGINE = {"Max trades per UK calendar day (counted on close)": 1, "Use the variant's indicator filters": True,
              "Window 1 on": False, "Window 1 start": "13:00", "Window 1 end": "14:30", "Window 2 on": True,
              "Window 2 start": "14:30", "Window 2 end": "17:00", "Window 3 on": True, "Window 3 start": "23:00",
              "Window 3 end": "02:00", "Range mode: 0 fixed at window open, 1 rolling, 2 last confirmed swings": 1,
              "Range lookback (5M bars)": 144, "Swing strength for mode 2 (bars each side)": 2,
              "Min range height (x ATR, 0 = none)": 4.0, "Max range height (x ATR, 0 = none)": 0.0,
              "Break buffer beyond the level (x ATR)": 0.25, "Entry mode: 0 limit at fib, 1 market on break close": 0,
              "Fib entry level (pullback of breakout move)": 0.3, "Anchor swing lookback (5M bars)": 12,
              "Re-anchor if the breakout extends before the fill": True, "Limit order validity (5M bars)": 6,
              "Cancel the limit if price reaches the target first": False,
              "Stop mode: 0 beyond anchor + k x ATR, 1 entry +/- k x ATR, 2 fib k of the move beyond extreme": 0,
              "Stop parameter k": 1.0, "Min stop distance (x ATR, 0 = none)": 0.0,
              "Max stop distance (x ATR, 0 = none)": 0.0,
              "Target mode: 0 fixed R:R, 1 further of R:R and fib extension": 0, "Target R:R": 6.0,
              "Fib extension for target mode 1": 1.0, "Breakeven trigger (R, 0 = none)": 0.5,
              "Breakeven lock (R)": 0.2, "Trailing stop distance behind best close (R, 0 = none)": 0.0,
              "Partial take-profit at (R, 0 = none)": 0.0, "Partial fraction closed": 0.3,
              "Stop after partial: entry + (R)": 0.0, "Time stop (5M bars after fill, 0 = none)": 576,
              "Risk % of equity per trade (portfolio weight; use a quarter to start)": 1.0}
SRC1_PARAMS = {**SRC_ENGINE, "Window 1 on": True, "Window 1 start": "08:00", "Window 1 end": "12:00",
               "Window 2 on": False, "Window 2 end": "18:00", "Window 3 on": False,
               "Range mode: 0 fixed at window open, 1 rolling, 2 last confirmed swings": 0,
               "Range lookback (5M bars)": 24, "Swing strength for mode 2 (bars each side)": 5,
               "Max range height (x ATR, 0 = none)": 30.0, "Break buffer beyond the level (x ATR)": 0.0,
               "Limit order validity (5M bars)": 144,
               "Stop mode: 0 beyond anchor + k x ATR, 1 entry +/- k x ATR, 2 fib k of the move beyond extreme": 2,
               "Stop parameter k": 2.5, "Min stop distance (x ATR, 0 = none)": 1.0,
               "Target mode: 0 fixed R:R, 1 further of R:R and fib extension": 1, "Target R:R": 8.0,
               "Fib extension for target mode 1": 0.27, "Breakeven trigger (R, 0 = none)": 0.0,
               "Trailing stop distance behind best close (R, 0 = none)": 3.0,
               "Partial take-profit at (R, 0 = none)": 3.0, "Partial fraction closed": 0.5}
SRC2_PARAMS = dict(SRC_ENGINE)
SRC8_PARAMS = {**SRC_ENGINE, "Window 1 end": "16:00", "Range lookback (5M bars)": 192,
               "Swing strength for mode 2 (bars each side)": 3, "Min range height (x ATR, 0 = none)": 8.0,
               "Max range height (x ATR, 0 = none)": 30.0, "Entry mode: 0 limit at fib, 1 market on break close": 1,
               "Fib entry level (pullback of breakout move)": 0.618,
               "Re-anchor if the breakout extends before the fill": False, "Limit order validity (5M bars)": 12,
               "Min stop distance (x ATR, 0 = none)": 0.5, "Fib extension for target mode 1": 0.618,
               "Breakeven trigger (R, 0 = none)": 2.0, "Partial fraction closed": 0.5}
SRC_PARAMS = {"SRC1": SRC1_PARAMS, "SRC2": SRC2_PARAMS, "SRC8": SRC8_PARAMS}
M5_PER_BAR = 3  # M5 bars per M15 chart bar


def _m5(n: int) -> int:
    return max(1, int(round(n / M5_PER_BAR)))


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """AtrRma: running mean for the first n values, then Wilder smoothing."""
    out = np.empty(len(x))
    a = 0.0
    for i in range(len(x)):
        a += (x[i] - a) / (i + 1 if i < n else n)
        out[i] = a
    return out


def stoch_kd(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14, ks: int = 3, d: int = 3):
    hh = pd.Series(h).rolling(n).max().to_numpy()
    ll = pd.Series(l).rolling(n).min().to_numpy()
    rng = hh - ll
    raw = np.where(rng > 0, 100.0 * (c - ll) / np.where(rng > 0, rng, 1.0), 50.0)
    k = pd.Series(raw).rolling(ks).mean().to_numpy()
    return k, pd.Series(k).rolling(d).mean().to_numpy()


def structure(h: np.ndarray, l: np.ndarray, s: int) -> np.ndarray:
    """Structure(b, s, idx) for every idx: +1 HH+HL, -1 LH+LL, 0 otherwise (strict pivots, confirmed s bars later)."""
    n = len(h)
    out = np.zeros(n, np.int64)
    ph = [np.nan, np.nan]  # most recent, previous
    pl = [np.nan, np.nan]
    for idx in range(n):
        j = idx - s
        if j - s >= 0 and j >= s:
            wh, wl = h[j - s: j + s + 1], l[j - s: j + s + 1]
            if (np.delete(wh, s) < h[j]).all():
                ph = [h[j], ph[0]]
            if (np.delete(wl, s) > l[j]).all():
                pl = [l[j], pl[0]]
        if np.isnan(ph[1]) or np.isnan(pl[1]):
            continue
        if ph[0] > ph[1] and pl[0] > pl[1]:
            out[idx] = 1
        elif ph[0] < ph[1] and pl[0] < pl[1]:
            out[idx] = -1
    return out


def _hhmm(x: str) -> int:
    hh, mm = x.strip().split(":")[:2]
    return int(hh) * 60 + int(mm)


class _SrcFilters:
    """FiltersAllow(isLong, i, t) for one variant; every higher-TF value is read at the last bar completed by t."""

    def __init__(self, variant: str, fr: Frames, vol_n: int) -> None:
        self.v = variant
        self.fr = fr
        self.vol_n = vol_n
        self.tf = {}
        for tf in ("M15", "M30", "H1", "H4", "D1"):
            b, code = fr.bars(tf)
            self.tf[tf] = (b, code, b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy())
        b4, bd = self.tf["H4"], self.tf["D1"]
        self.st4 = structure(b4[2], b4[3], 2)
        self.stD = structure(bd[2], bd[3], 5)
        if variant == "SRC1":
            m = self.tf["M15"]
            self.k15, self.d15 = stoch_kd(m[2], m[3], m[4])
            self.e20 = ema_seeded(m[4], 20)
            self.e50_4 = ema_seeded(b4[4], 50)
        elif variant == "SRC2":
            h1 = self.tf["H1"]
            self.k1, _ = stoch_kd(h1[2], h1[3], h1[4])
            self.kD, _ = stoch_kd(bd[2], bd[3], bd[4])
        else:
            m30 = self.tf["M30"]
            self.eD100 = ema_seeded(bd[4], 100)
            self.e50_30 = ema_seeded(m30[4], 50)
        self.vol = fr.v

    def last(self, tf: str, i: int) -> int:
        return int(self.tf[tf][1][i]) - 1  # bar containing the open time of chart bar i, minus one

    def vol_ok(self, i: int, k: float) -> bool:
        n = self.vol_n
        if i - n < 0:
            return False
        return self.vol[i] >= k * self.vol[i - n:i].mean()

    def allow(self, is_long: bool, i: int) -> bool:
        sgn = 1 if is_long else -1
        not_against = (lambda st: st != -1) if is_long else (lambda st: st != 1)
        if self.v == "SRC1":
            i15 = self.last("M15", i)
            if i15 < 60 or not sgn * (self.k15[i15] - self.d15[i15]) > 0:
                return False
            if not sgn * (self.e20[i15] - self.e20[i15 - 3]) > 0:
                return False
            i4 = self.last("H4", i)
            if i4 < 60 or not sgn * (self.tf["H4"][4][i4] - self.e50_4[i4]) > 0 or not not_against(self.st4[i4]):
                return False
            iD = self.last("D1", i)
            return iD >= 30 and not_against(self.stD[iD])
        if self.v == "SRC2":
            i1 = self.last("H1", i)
            if i1 < 30 or not sgn * (self.k1[i1] - 50) > 0:
                return False
            iD = self.last("D1", i)
            if iD < 30 or not sgn * (self.kD[iD] - 50) > 0 or not not_against(self.stD[iD]):
                return False
            i4 = self.last("H4", i)
            if i4 < 30 or not not_against(self.st4[i4]):
                return False
            return self.vol_ok(i, 0.8)
        iD = self.last("D1", i)
        if iD < 300 or not not_against(self.stD[iD]):
            return False
        i4 = self.last("H4", i)
        if i4 < 30 or not not_against(self.st4[i4]):
            return False
        if not self.vol_ok(i, 1.0):
            return False
        if not sgn * (self.tf["D1"][4][iD] - self.eD100[iD]) > 0:
            return False
        i30 = self.last("M30", i)
        return i30 >= 200 and sgn * (self.e50_30[i30] - self.e50_30[i30 - 3]) > 0


def run_src(m15: pd.DataFrame, variant: str, params: dict | None = None, costs: Costs | None = None,
            balance: float = 10_000.0, trade_start=None) -> pd.DataFrame:
    """M15 approximation of one SRC round-5 cBot (variant 'SRC1', 'SRC2' or 'SRC8')."""
    P = _p(params, SRC_PARAMS[variant])
    costs = costs or Costs.from_program()
    fr = Frames(m15)
    uk = fr.t.tz_convert("Europe/London").tz_localize(None)
    H, L, C, O = fr.h, fr.l, fr.c, fr.o
    atr = rma(true_range(H, L, C), _m5(14)) / math.sqrt(M5_PER_BAR)
    filt = _SrcFilters(variant, fr, _m5(20))
    risk_pct = float(P["Risk % of equity per trade (portfolio weight; use a quarter to start)"])
    max_day = int(P["Max trades per UK calendar day (counted on close)"])
    use_f = bool(P["Use the variant's indicator filters"])
    wins = [(bool(P[f"Window {w} on"]), _hhmm(P[f"Window {w} start"]), _hhmm(P[f"Window {w} end"])) for w in (1, 2, 3)]
    rmode = int(P["Range mode: 0 fixed at window open, 1 rolling, 2 last confirmed swings"])
    rbars = _m5(int(P["Range lookback (5M bars)"]))
    pstr = _m5(int(P["Swing strength for mode 2 (bars each side)"]))
    min_rng, max_rng = float(P["Min range height (x ATR, 0 = none)"]), float(P["Max range height (x ATR, 0 = none)"])
    buf_k = float(P["Break buffer beyond the level (x ATR)"])
    emode = int(P["Entry mode: 0 limit at fib, 1 market on break close"])
    fib = float(P["Fib entry level (pullback of breakout move)"])
    swing_lb = _m5(int(P["Anchor swing lookback (5M bars)"]))
    re_anchor = bool(P["Re-anchor if the breakout extends before the fill"])
    valid = _m5(int(P["Limit order validity (5M bars)"]))
    cancel_tp = bool(P["Cancel the limit if price reaches the target first"])
    slmode = int(P["Stop mode: 0 beyond anchor + k x ATR, 1 entry +/- k x ATR, 2 fib k of the move beyond extreme"])
    slk = float(P["Stop parameter k"])
    min_risk, max_risk = float(P["Min stop distance (x ATR, 0 = none)"]), float(P["Max stop distance (x ATR, 0 = none)"])
    tpmode = int(P["Target mode: 0 fixed R:R, 1 further of R:R and fib extension"])
    rr, tpext = float(P["Target R:R"]), float(P["Fib extension for target mode 1"])
    be_trig, be_lock = float(P["Breakeven trigger (R, 0 = none)"]), float(P["Breakeven lock (R)"])
    trail_r = float(P["Trailing stop distance behind best close (R, 0 = none)"])
    part_r, part_frac = float(P["Partial take-profit at (R, 0 = none)"]), float(P["Partial fraction closed"])
    part_be = float(P["Stop after partial: entry + (R)"])
    tstop = _m5(int(P["Time stop (5M bars after fill, 0 = none)"])) if int(P["Time stop (5M bars after fill, 0 = none)"]) else 0
    book = Book(costs, balance, {"Use Portfolio Stop Rules": False}, partial_pct=part_frac * 100, breakeven=True)
    book.partial_be_r = part_be
    ok = _start_mask(fr, trade_start)
    eq = np.empty(fr.n)

    S = {"dir": 0, "anchor": 0.0, "extreme": 0.0, "entry": 0.0, "sl": 0.0, "tp": 0.0, "risk": 0.0, "atr": 0.0,
         "pend": False, "expiry": None, "fill_bar": -1, "best": 0.0, "be_done": False, "had": False,
         "today": None, "trades_today": 0, "in_window": False, "have_range": False, "win_end": None,
         "RH": 0.0, "RL": 0.0, "info": {}}
    win_day = [None, None, None]

    def compute_range(i: int) -> None:
        j0 = max(1, i - rbars)
        if rmode == 2:
            ph = pl = -1.0
            s = pstr
            for j in range(i - 1 - s, j0 + s, -1):
                if ph < 0 and j - s >= 0 and (np.delete(H[j - s:j + s + 1], s) < H[j]).all():
                    ph = H[j]
                if pl < 0 and j - s >= 0 and (np.delete(L[j - s:j + s + 1], s) > L[j]).all():
                    pl = L[j]
                if ph > 0 and pl > 0:
                    break
            if ph > 0 and pl > 0 and ph > pl:
                S["RH"], S["RL"] = ph, pl
                return
        S["RH"], S["RL"] = H[j0:i].max(), L[j0:i].min()

    def on_filled(j: int, _t, fill: float, units: float, bid_now: float) -> None:
        t = fr.t[j]  # fills happen during chart bar j (UTC timestamps in the trade list)
        d = S["dir"]
        S["fill_bar"], S["best"], S["be_done"], S["had"] = j, fill, False, True
        sl = S["sl"]
        if slmode == 1 and emode == 1:
            sl = fill - d * slk * S["atr"]
        risk = (fill - sl) * d
        if emode == 1:
            tp = fill + d * rr * risk
            if tpmode == 1:
                alt = S["extreme"] + d * tpext * abs(S["extreme"] - S["anchor"])
                if (alt - tp) * d > 0:
                    tp = alt
            S["tp"] = tp
        S["risk"] = risk
        book.open_fill(d, t, fill, units, round(sl, 2), round(S["tp"], 2), j, max(risk, 0.0),
                       partial_r=part_r if part_r > 0 else None, info=dict(S["info"], fill=fill, risk=risk))
        if risk <= 0:  # "Fill beyond the stop level; closing."
            book.pos.stop_dist = abs(risk) or 1e-9
            book.close_all(bid_now, t, "fill_beyond_stop")

    def place_or_cancel(j: int, t) -> None:
        bid = O[j]
        units = book.units_for(risk_pct * book.equity(bid) / max(book.balance, 1e-9), S["risk"])
        if units < UNIT_STEP:
            S["dir"] = 0
            return
        d = S["dir"]
        cur = bid + costs.spread if d > 0 else bid
        if (cur - S["entry"]) * d <= 0:  # limit already reached -> market
            on_filled(j, t, book.market_fill(d, bid), units, bid)
            return
        S["pend"], S["units"] = True, units

    def detect_break(i: int, j: int, t) -> None:
        if rmode == 1:
            compute_range(i)
        a = atr[i]
        if a <= 0:
            return
        rng = S["RH"] - S["RL"]
        if min_rng > 0 and rng < min_rng * a:
            return
        if max_rng > 0 and rng > max_rng * a:
            return
        buf = buf_k * a
        c, cp = C[i], C[i - 1]
        d = 0
        if c > S["RH"] + buf and cp <= S["RH"] and (not use_f or filt.allow(True, i)):
            d = 1
        elif c < S["RL"] - buf and cp >= S["RL"] and (not use_f or filt.allow(False, i)):
            d = -1
        if d == 0:
            return
        j0 = max(0, i - swing_lb + 1)
        anc = L[j0:i + 1].min() if d > 0 else H[j0:i + 1].max()
        ext = H[i] if d > 0 else L[i]
        move = abs(ext - anc)
        if move <= 0:
            return
        en = ext - d * fib * move
        s = anc - d * slk * a if slmode == 0 else (en - d * slk * a if slmode == 1 else ext - d * slk * move)
        rk = (en - s) * d
        if emode == 0:
            if rk <= 0 or (min_risk > 0 and rk < min_risk * a) or (max_risk > 0 and rk > max_risk * a):
                return
        target = en + d * rr * rk
        if tpmode == 1:
            alt = ext + d * tpext * move
            if (alt - target) * d > 0:
                target = alt
        S.update(dir=d, anchor=anc, extreme=ext, entry=en, sl=s, tp=target, risk=rk, atr=a, be_done=False)
        S["info"] = {"break_bar_uk": str(t), "RH": S["RH"], "RL": S["RL"], "close": c, "prev_close": cp, "atr": a,
                     "anchor": anc, "extreme": ext, "limit": en, "stop": s, "target": target}
        if emode == 1:
            bid = O[j]
            px = bid + costs.spread if d > 0 else bid
            sl = px - d * slk * a if slmode == 1 else s
            risk = (px - sl) * d
            if risk <= 0 or (min_risk > 0 and risk < min_risk * a) or (max_risk > 0 and risk > max_risk * a):
                S["dir"] = 0
                return
            S["sl"], S["risk"] = sl, risk
            units = book.units_for(risk_pct * book.equity(bid) / max(book.balance, 1e-9), risk)
            if units < UNIT_STEP:
                S["dir"] = 0
                return
            on_filled(j, t, book.market_fill(d, bid), units, bid)
            return
        S["expiry"] = t + pd.Timedelta(minutes=15 * (valid + 1))
        place_or_cancel(j, t)

    def manage_pending_at_close(i: int, j: int, t) -> None:
        if not re_anchor or S["dir"] == 0:
            return
        d = S["dir"]
        if not ((H[i] > S["extreme"]) if d > 0 else (L[i] < S["extreme"])):
            return
        S["extreme"] = H[i] if d > 0 else L[i]
        move = abs(S["extreme"] - S["anchor"])
        S["entry"] = S["extreme"] - d * fib * move
        if slmode == 1:
            S["sl"] = S["entry"] - d * slk * atr[i]
        elif slmode == 2:
            S["sl"] = S["extreme"] - d * slk * move
        S["risk"] = (S["entry"] - S["sl"]) * d
        S["pend"] = False
        if S["risk"] <= 0:
            S["dir"] = 0
            return
        S["tp"] = S["entry"] + d * rr * S["risk"]
        if tpmode == 1:
            alt = S["extreme"] + d * tpext * move
            if (alt - S["tp"]) * d > 0:
                S["tp"] = alt
        S["info"].update(limit=S["entry"], stop=S["sl"], target=S["tp"], re_anchored=True)
        place_or_cancel(j, uk[i])

    def manage_position_at_close(i: int, j: int, t) -> None:
        p = book.pos
        d = S["dir"] or p.dir
        if tstop > 0 and S["fill_bar"] >= 0 and i - S["fill_bar"] >= tstop:
            book.close_all(O[j], fr.t[j], "time_stop")
            return
        close = C[i]
        cur = p.sl if p.sl is not None else S["sl"]
        fp, risk = p.entry, S["risk"]
        if be_trig > 0 and not S["be_done"] and (close - fp) * d >= be_trig * risk:
            ns = round(fp + d * be_lock * risk, 2)
            if (ns - cur) * d > 0:
                p.sl, cur = ns, ns
            S["be_done"] = True
        if trail_r > 0:
            if (close - S["best"]) * d > 0:
                S["best"] = close
            ns = round(S["best"] - d * trail_r * risk, 2)
            if (ns - cur) * d > 0:
                p.sl = ns
                S["be_done"] = True

    def pending_segment(j: int, b0: float, b1: float, gap: bool) -> float | None:
        """Pending limit on the move b0 -> b1: returns the bid at which the order filled (None if not filled)."""
        d = S["dir"]
        if d > 0:
            fill_bid, cancel_bid = S["entry"] - costs.spread, S["sl"]
            if b1 < b0:
                hits = sorted([(fill_bid, 0), (cancel_bid, 1)], key=lambda z: -z[0])
                for lvl, kind in hits:
                    if b1 <= lvl:
                        if kind == 1:
                            S["pend"], S["dir"] = False, 0
                            return None
                        return b1 if gap else min(b0, lvl)
            elif cancel_tp and b1 >= S["tp"]:
                S["pend"], S["dir"] = False, 0
        else:
            fill_bid, cancel_bid = S["entry"], S["sl"] - costs.spread
            if b1 > b0:
                for lvl, kind in sorted([(fill_bid, 0), (cancel_bid, 1)]):
                    if b1 >= lvl:
                        if kind == 1:
                            S["pend"], S["dir"] = False, 0
                            return None
                        return b1 if gap else max(b0, lvl)
            elif cancel_tp and b1 + costs.spread <= S["tp"]:
                S["pend"], S["dir"] = False, 0
        return None

    for j in range(fr.n):
        o, h, l, c, t = O[j], H[j], L[j], C[j], fr.t[j]
        if not ok[j]:
            eq[j] = book.balance
            continue
        segs = ([(C[j - 1], o, True)] if j > 0 else []) + [(a0, a1, False) for a0, a1 in
                                                            zip(_path(o, h, l, c)[:-1], _path(o, h, l, c)[1:])]
        # gap tick: server-side stops / pending expiry / limit fill, then OnBar for the bar that just closed
        if S["pend"] and S["expiry"] is not None and uk[j] >= S["expiry"]:
            S["pend"] = False
        if j > 0:
            b0, b1, _g = segs[0]
            if book.pos is not None:
                book.segment(b0, b1, t, gap=True)
            elif S["pend"]:
                fb = pending_segment(j, b0, b1, True)
                if fb is not None:
                    S["pend"] = False
                    on_filled(j, t, fb + (costs.spread if S["dir"] > 0 else 0.0), S["units"], fb)
            segs = segs[1:]
            if book.pos is None and S["had"]:
                S["had"], S["dir"] = False, 0
                S["trades_today"] += 1
            i = j - 1
            if i >= rbars + 2 and i >= swing_lb + 2:
                ti = uk[i]
                if book.pos is not None:
                    manage_position_at_close(i, j, ti)
                elif S["pend"]:
                    manage_pending_at_close(i, j, ti)
                if S["today"] != ti.date():
                    S["today"], S["trades_today"] = ti.date(), 0
                mod = ti.hour * 60 + ti.minute
                for w, (on, st, en) in enumerate(wins):
                    if not on or not (st <= mod and win_day[w] != ti.date()):
                        continue
                    if en > st and not mod < en:
                        continue
                    win_day[w] = ti.date()
                    compute_range(i)
                    S["have_range"] = S["in_window"] = True
                    S["win_end"] = pd.Timestamp(ti.date()) + pd.Timedelta(minutes=en) + pd.Timedelta(days=0 if en > st else 1)
                    if S["pend"]:
                        S["pend"], S["dir"] = False, 0
                if S["in_window"] and ti >= S["win_end"]:
                    S["in_window"] = False
                if (book.pos is None and not S["pend"] and S["in_window"] and S["have_range"]
                        and S["trades_today"] < max_day):
                    detect_break(i, j, ti)
        # intrabar path
        for b0, b1, gap in segs:
            if book.pos is not None:
                book.segment(b0, b1, t, gap)
            elif S["pend"]:
                fb = pending_segment(j, b0, b1, gap)
                if fb is not None:
                    S["pend"] = False
                    on_filled(j, t, fb + (costs.spread if S["dir"] > 0 else 0.0), S["units"], fb)
                    if book.pos is not None:
                        book.segment(fb, b1, t, gap)
            if book.pos is None and S["had"]:
                S["had"], S["dir"] = False, 0
                S["trades_today"] += 1
        eq[j] = book.equity(c)
    out = _finish(book, fr, eq, trade_start)
    out.attrs["approximation"] = "M15 approximation of an M5 cBot"
    return out


def run_src1(m15, params=None, costs=None, balance=10_000.0, trade_start=None):
    return run_src(m15, "SRC1", params, costs, balance, trade_start)


def run_src2(m15, params=None, costs=None, balance=10_000.0, trade_start=None):
    return run_src(m15, "SRC2", params, costs, balance, trade_start)


def run_src8(m15, params=None, costs=None, balance=10_000.0, trade_start=None):
    return run_src(m15, "SRC8", params, costs, balance, trade_start)


SRC_BOTS = {"SRC1_LondonMorning (M15 approx)": run_src1, "SRC2_Main (M15 approx)": run_src2,
            "SRC8_MarketEntry (M15 approx)": run_src8}


BOTS = {
    "4_BotG_XAUUSD_H2_SwingPartial_v3": run_bot4,
    "5_GoldWickV3_XAUUSD_H4": run_bot5,
    "6_BreakoutTrendRunnerV3_XAUUSD_H1": run_bot6,
    "7_XAU_MACD_DEMA_H4_v4": run_bot7,
}


# ---------------------------------------------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------------------------------------------
def max_dd_pct(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    e = equity.to_numpy(float)
    peak = np.maximum.accumulate(e)
    return float(((peak - e) / peak).max() * 100)


def summarize(tr: pd.DataFrame, balance: float = 10_000.0) -> dict:
    eq = tr.attrs.get("equity", pd.Series(dtype=float))
    n = len(tr)
    if n == 0:
        return {"trades": 0}
    span = (eq.index[-1] - eq.index[0]).days / 365.25 if len(eq) else np.nan
    gp, gl = tr.net[tr.net > 0].sum(), -tr.net[tr.net < 0].sum()
    yr = tr.groupby(pd.to_datetime(tr.exit_time).dt.year)["net"].sum().round(0).astype(int).to_dict()
    side = tr.groupby("side")["net"].agg(["count", "sum"]).round(0)
    return {"trades": n, "trades_per_year": n / span if span else np.nan, "win_rate": float((tr.net > 0).mean()),
            "pf": float(gp / gl) if gl > 0 else np.inf, "avg_r": float(tr.pnl_r.mean()), "sum_r": float(tr.pnl_r.sum()),
            "net": float(tr.net.sum()), "roi_pct": float(tr.net.sum() / balance * 100), "max_dd_pct": max_dd_pct(eq),
            "closed_dd_pct": max_dd_pct(pd.concat([pd.Series([balance]), tr.balance])),
            "by_year": yr, "sides": {k: (int(v["count"]), float(v["sum"])) for k, v in side.iterrows()},
            "deals": int(tr.n_closes.sum()), "half_risk_trades": tr.attrs.get("half_risk_trades", 0),
            "paused_at": tr.attrs.get("paused_at")}


def combined_equity(results: dict[str, pd.DataFrame], balance: float = 10_000.0) -> pd.Series:
    """One account = start balance + sum of every bot's P&L (each bot sized from its own 10k sub-account)."""
    curves = [r.attrs["equity"] - balance for r in results.values()]
    df = pd.concat(curves, axis=1).sort_index().ffill().fillna(0.0)
    return balance + df.sum(axis=1)
