"""Bar-by-bar numba backtester producing per-trade records in R units.

Signals are evaluated on bar i close; entries fill at bar i+1 open plus costs.
Stops/targets are checked against bar high/low; if both are touched in one bar the stop wins
(pessimistic). Stops gapped through fill at the open. One position at a time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from propfloor.data.store import pip_size

TRADE_COLS = ["entry_idx", "exit_idx", "side", "entry", "exit", "stop_dist", "pnl_r", "mae_r", "mfe_r",
              "mae_idx", "mfe_idx", "post_mfe_r", "reason"]
REASONS = {0: "stop", 1: "target", 2: "time", 3: "opposite", 4: "weekend", 5: "end", 6: "trail"}


@njit(cache=True)
def _run(o, h, l, c, atr, go_long, go_short, force_flat, swing_lo, swing_hi, exit_long, exit_short,
         sl_type, sl_atr, tp_r, time_stop, trail_atr, be_r, cost, partial_r, trail_type):
    """exit_long/exit_short: signal exits evaluated at bar close (mean targets, channels, session end...).
    partial_r > 0: close half at +partial_r R and move the stop to breakeven (trailing-floor protection).
    trail_type 0: close - k*ATR; 1: chandelier (extreme since entry - k*ATR).
    MAE/MFE/post are measured on the remaining exposure plus banked partial profit, in R of the full size."""
    n = o.shape[0]
    out = np.zeros((n, 13))
    k = 0
    pos = 0
    entry = 0.0
    stop = 0.0
    tgt = 0.0
    dist = 0.0
    ei = 0
    mae = 0.0
    mfe = 0.0
    mae_i = 0
    mfe_i = 0
    post = 0.0
    size = 1.0
    banked = 0.0
    ext = 0.0
    i = 0
    while i < n - 1:
        if pos == 0:
            sig = 0
            if go_long[i] and not force_flat[i]:
                sig = 1
            elif go_short[i] and not force_flat[i]:
                sig = -1
            if sig != 0 and not np.isnan(atr[i]) and atr[i] > 0:
                a = atr[i]
                ep = o[i + 1] + sig * cost * 0.5  # half cost on entry, half on exit
                if sl_type == 1:  # swing
                    lvl = swing_lo[i] if sig == 1 else swing_hi[i]
                    d = (ep - lvl) * sig if not np.isnan(lvl) else 0.0
                    if d < 0.5 * a or d > 4.0 * a:
                        d = sl_atr * a
                elif sl_type == 2:  # signal bar
                    d = (ep - l[i]) if sig == 1 else (h[i] - ep)
                    if d < 0.5 * a or d > 4.0 * a:
                        d = sl_atr * a
                else:
                    d = sl_atr * a
                pos = sig
                entry = ep
                dist = d
                stop = ep - sig * d
                tgt = ep + sig * tp_r * d if tp_r > 0 else np.nan
                ei = i + 1
                mae = 0.0
                mfe = 0.0
                mae_i = ei
                mfe_i = ei
                post = 0.0
                size = 1.0
                banked = 0.0
                ext = ep
            i += 1
            continue
        # in position: bar i (i >= ei)
        reason = -1
        xp = 0.0
        worst = l[i] if pos == 1 else h[i]
        best = h[i] if pos == 1 else l[i]
        hit_stop = (l[i] <= stop) if pos == 1 else (h[i] >= stop)
        hit_tgt = False
        if not np.isnan(tgt):
            hit_tgt = (h[i] >= tgt) if pos == 1 else (l[i] <= tgt)
        # partial take-profit (only if the stop is not hit in the same bar: pessimistic)
        if partial_r > 0 and size == 1.0 and not hit_stop and (best - entry) * pos >= partial_r * dist:
            banked = 0.5 * partial_r
            size = 0.5
            if (stop - entry) * pos < 0:
                stop = entry
        if hit_stop:
            xp = min(o[i], stop) if pos == 1 else max(o[i], stop)
            reason = 0 if (stop - entry) * pos < 0 else 6
            worst = xp if (xp - worst) * pos < 0 else worst
        elif hit_tgt:
            xp = max(o[i], tgt) if pos == 1 else min(o[i], tgt)
            reason = 1
            best = xp
        elif force_flat[i]:
            xp = c[i]
            reason = 4
        elif time_stop > 0 and i - ei + 1 >= time_stop:
            xp = c[i]
            reason = 2
        elif (pos == 1 and exit_long[i]) or (pos == -1 and exit_short[i]):
            xp = c[i]
            reason = 3
        elif i == n - 2:
            xp = c[i]
            reason = 5
        fav = banked + size * (best - entry) * pos / dist
        adv = banked + size * (worst - entry) * pos / dist
        if fav > mfe:
            mfe = fav
            mfe_i = i
            post = 0.0
        if adv < mae:
            mae = adv
            mae_i = i
        if i > mfe_i and adv < post:
            post = adv
        if reason >= 0:
            xp_net = xp - pos * cost * 0.5
            pnl = banked + size * (xp_net - entry) * pos / dist
            out[k, 0] = ei
            out[k, 1] = i
            out[k, 2] = pos
            out[k, 3] = entry
            out[k, 4] = xp_net
            out[k, 5] = dist
            out[k, 6] = pnl
            out[k, 7] = min(mae, pnl)
            out[k, 8] = max(mfe, pnl)
            out[k, 9] = mae_i
            out[k, 10] = mfe_i
            out[k, 11] = min(post, pnl)
            out[k, 12] = reason
            k += 1
            pos = 0
            # flat now: the loop re-evaluates bar i's close for a new entry at i+1
            continue
        # end-of-bar management (effective next bar)
        if pos == 1:
            ext = max(ext, h[i])
        else:
            ext = min(ext, l[i])
        if be_r > 0 and mfe >= be_r and (stop - entry) * pos < 0:
            stop = entry
        if trail_atr > 0 and not np.isnan(atr[i]):
            ref = c[i] if trail_type == 0 else ext
            ns = ref - pos * trail_atr * atr[i]
            if (ns - stop) * pos > 0:
                stop = ns
        i += 1
    return out[:k]


def run_backtest(ctx, long_sig: np.ndarray, short_sig: np.ndarray, exit_p: dict,
                 spread_pips: float = 1.0, slippage_pips: float = 0.3, commission_per_lot: float = 7.0,
                 pip_value_per_lot: float = 10.0) -> pd.DataFrame:
    ps = pip_size(ctx.symbol)
    atr = ctx.atr(14)
    sl_type = {"atr": 0, "swing": 1, "bar": 2}[exit_p.get("sl_type", "atr")]
    sh, sl = ctx.swings(3)
    ff = ctx.get("force_flat", lambda: _force_flat(ctx))
    cost = (spread_pips + 2 * slippage_pips) * ps
    from propfloor.strategies.blocks import exit_signals

    xl, xs = exit_signals(ctx, exit_p)
    if exit_p.get("exit_opposite", False):
        xl = xl | short_sig
        xs = xs | long_sig
    raw = _run(ctx.o, ctx.h, ctx.l, ctx.c, atr, long_sig.astype(np.bool_), short_sig.astype(np.bool_), ff, sl, sh,
               xl.astype(np.bool_), xs.astype(np.bool_),
               sl_type, float(exit_p.get("sl_atr", 2.0)), float(exit_p.get("tp_r", 2.0)), int(exit_p.get("time_stop", 0)),
               float(exit_p.get("trail_atr", 0.0)), float(exit_p.get("breakeven_r", 0.0)), cost,
               float(exit_p.get("partial_r", 0.0)), 1 if exit_p.get("trail_type", "close") == "chandelier" else 0)
    df = pd.DataFrame(raw, columns=TRADE_COLS)
    for col in ("entry_idx", "exit_idx", "side", "mae_idx", "mfe_idx", "reason"):
        df[col] = df[col].astype(np.int64)
    df["stop_pips"] = df["stop_dist"] / ps
    # commission expressed in R: money per lot / money risked per lot
    df["pnl_r"] -= commission_per_lot / (pip_value_per_lot * df["stop_pips"].clip(lower=1e-9))
    t = ctx.time
    df["entry_time"] = t[df["entry_idx"].to_numpy()]
    df["exit_time"] = t[df["exit_idx"].to_numpy()] + _bar_len(ctx)  # exit happens within/at end of bar
    df["mae_time"] = t[df["mae_idx"].to_numpy()]
    df["mfe_time"] = t[df["mfe_idx"].to_numpy()]
    df["reason"] = df["reason"].map(REASONS)
    df["symbol"] = ctx.symbol
    return df


def _bar_len(ctx) -> pd.Timedelta:
    from propfloor.data.clean import TF_RULES

    r = TF_RULES.get(ctx.tf.upper(), "1h")
    return pd.Timedelta(r) - pd.Timedelta(seconds=1) if r != "D1" else pd.Timedelta(hours=23)


def _force_flat(ctx) -> np.ndarray:
    """Be flat before the weekend: Friday from 15:00 New York, or any bar followed by a >24h gap."""
    ny = ctx.time.tz_convert("America/New_York")
    late_fri = (ny.weekday == 4) & (ny.hour >= 15)
    nxt_gap = np.r_[np.diff(ctx.t_ns) > 24 * 3_600_000_000_000, False]
    return np.asarray(late_fri | (ny.weekday >= 5) | nxt_gap, dtype=np.bool_)
