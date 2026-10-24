"""Out-of-sample replay of the user's cTrader gold cBots on our XAUUSD M15 history (paper research only).

usage: python scripts/cbot_replay.py [--start 2012-11-01] [--end 2022-03-05] [--validate] [--out DIR]
                                     [--params-dir research/user_reports/parsed]

Prints per bot: trades, trades/year, win rate, profit factor, avg R, ROI, max DD %, net by year, long/short split,
next to the user's 2024-26 cTrader report numbers (summary.json, when present), plus the combined account
(sum of every bot's P&L on one 10k account) and its max drawdown. --validate prints 5 sample trades per bot with
the indicator values that triggered them and re-checks each trade against the C# entry rules independently.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from propfloor.data.store import DataStore
from propfloor.ports import cbots as cb
from propfloor.strategies import indicators as ind

REPORT_KEY = {"4_BotG_XAUUSD_H2_SwingPartial_v3": "C7N_4", "5_GoldWickV3_XAUUSD_H4": "C7N_5",
              "6_BreakoutTrendRunnerV3_XAUUSD_H1": "C7N_6", "7_XAU_MACD_DEMA_H4_v4": "C7N_7"}
BOT_NO = {k: int(k[0]) for k in REPORT_KEY}


def check_params(pdir: Path) -> None:
    """The ported defaults must equal the user's params.json (trading-relevant keys)."""
    for name, n in BOT_NO.items():
        f = pdir / f"C7N_{n}_params.json"
        if not f.exists():
            print(f"[params] {f} missing - using built-in defaults")
            continue
        js = json.loads(f.read_text())
        ours = cb.DEFAULT_PARAMS[n]
        bad = []
        for k, v in ours.items():
            if k not in js:
                continue
            jv = js[k]
            if isinstance(v, bool):
                jv = str(jv).lower() == "true"
            elif isinstance(v, (int, float)):
                jv = float(jv)
                v = float(v)
            elif isinstance(v, str):
                jv, v = str(jv).lower(), v.lower()
            if jv != v:
                bad.append((k, v, js[k]))
        print(f"[params] bot {n}: {'OK, all ' + str(len([k for k in ours if k in js])) + ' keys match' if not bad else bad}")
        if bad:
            sys.exit(1)


def fmt_stats(name: str, s: dict, rep: dict | None) -> str:
    lines = [f"\n=== {name} ==="]
    if s.get("trades", 0) == 0:
        return lines[0] + "\n  no trades"
    lines.append(f"  positions {s['trades']} (cTrader-style deals incl. partial closes: {s['deals']}), "
                 f"{s['trades_per_year']:.1f}/yr | win {s['win_rate']:.1%} | PF {s['pf']:.2f} | avg R {s['avg_r']:+.3f} "
                 f"(sum {s['sum_r']:+.1f}R) | ROI {s['roi_pct']:+.1f}% | max DD {s['max_dd_pct']:.1f}% equity "
                 f"({s['closed_dd_pct']:.1f}% closed)")
    lines.append(f"  net by year: {s['by_year']}")
    lines.append(f"  long/short (n, net): {s['sides']} | trades at half risk: {s['half_risk_trades']} | "
                 f"paused (20% DD) at: {s['paused_at']}")
    if rep:
        lines.append(f"  user 2024-26 cTrader: trades {rep['trades']} | win {rep['win']:.1%} | PF {rep['pf']:.2f} | "
                     f"ROI {rep['roi']:+.1f}% | max DD {rep['maxdd'] * 100:.1f}% | by year {rep['by_year']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# validation: independent re-check of each trade's entry conditions
# ---------------------------------------------------------------------------------------------------------------
def validate(name: str, tr: pd.DataFrame, m15: pd.DataFrame) -> None:
    fr = cb.Frames(m15)
    n = BOT_NO[name]
    print(f"\n--- validation {name}: {len(tr)} positions ---")
    if len(tr) == 0:
        return
    fails = 0
    if n == 4:
        b, _ = fr.bars("H2")
        H, L, C = (b[c].to_numpy() for c in ("high", "low", "close"))
        e75 = ind.ema(C, 75)
        mline, msig, _h = ind.macd(C, 12, 26, 9)
        a14 = ind.atr(H, L, C, 14)
        for _, r in tr.iterrows():
            i = b.index.get_loc(pd.Timestamp(r["info"]["signal_bar"]))
            j = i - 4
            piv_ok = L[j] == L[j - 4:j + 5].min()
            ok = piv_ok and r["info"]["swing_low"] > r["info"]["prev_swing_low"] and C[i] > e75[i] and mline[i] > msig[i]
            ok &= abs(r.stop_dist - 1.5 * r["info"]["atr"]) < 1e-6
            fails += not ok
    elif n == 5:
        b, _ = fr.bars("H4")
        d1, _ = fr.bars("D1")
        e200 = ind.ema(d1["close"].to_numpy(), 200)
        for _, r in tr.iterrows():
            inf = r["info"]
            jd = d1.index.get_loc(pd.Timestamp(inf["daily_bar"]))
            ok = inf["low_s"] < inf["low_p"] and inf["close_s"] > inf["high_p"] and inf["close_s"] > inf["ema40"]
            ok &= d1["close"].iloc[jd] > e200[jd] * (1 - 0.002)  # SMA- vs first-close-seeded EMA: tiny tolerance
            ok &= abs((r.entry - r.stop_dist) - (inf["low_s"] + 0.03)) < 0.02  # stop = signal low (+ slippage shift)
            fails += not ok
    elif n == 6:
        for _, r in tr.iterrows():
            inf = r["info"]
            ok = inf["entry_bid"] > inf["hh24"] and inf["ema50"] > inf["ema200"] and inf["daily_trend"] == 1
            ok &= abs((r.entry - r.stop_dist) - (inf["ll24"] - 0.4 * inf["atr"] + 0.03)) < 0.02  # + slippage shift
            fails += not ok
    elif n == 7:
        b, _ = fr.bars("H4")
        _m, _s, hist = ind.macd(b["close"].to_numpy(), 12, 26, 9)
        up = (hist > np.roll(hist, 1)) & (np.roll(hist, 1) > np.roll(hist, 2))
        dn = (hist < np.roll(hist, 1)) & (np.roll(hist, 1) < np.roll(hist, 2))
        for _, r in tr.iterrows():
            inf = r["info"]
            i = b.index.get_loc(pd.Timestamp(inf["h4_bar"]))
            ev = [k for k in range(i, max(i - 300, 2), -1) if up[k] or dn[k]]
            ok = bool(ev) and up[ev[0]]  # latest 2-bar histogram run before the entry is a rising one
            ok &= inf["dema"] > inf["dema_prev"]
            ok &= pd.Timestamp(inf["h12_bar"]) + pd.Timedelta(hours=12) <= cb.server_clock(pd.DatetimeIndex([r.entry_time]))[0]
            ok &= abs(r.stop_dist - round(2 * inf["atr"] / 0.01, 1) * 0.01) < 1e-6
            fails += not ok
    print(f"independent entry-rule re-check: {len(tr) - fails}/{len(tr)} pass")
    reasons = tr.reason.value_counts().to_dict()
    print(f"exit reasons: {reasons}")
    smp = tr.iloc[np.linspace(0, len(tr) - 1, 5).astype(int)]
    for _, r in smp.iterrows():
        info = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["info"].items()}
        print(f"  {r.entry_time} -> {r.exit_time} {r.side} entry {r.entry:.2f} exit {r.exit:.2f} units {r.units:.0f} "
              f"stop {r.stop_dist:.2f} pnl {r.pnl_r:+.2f}R ({r.net:+.2f}) closes {r.n_closes} reason {r.reason}\n"
              f"      {info}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/parquet")
    ap.add_argument("--start", default="2012-11-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--params-dir", default="research/user_reports/parsed")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=None, help="write per-bot trade csvs + combined equity here")
    ap.add_argument("--balance", type=float, default=10_000.0)
    a = ap.parse_args()

    pdir = Path(a.params_dir)
    check_params(pdir)
    rep = json.loads((pdir / "summary.json").read_text()) if (pdir / "summary.json").exists() else {}
    costs = cb.Costs.from_program()
    print(f"costs (programs/generic_2stage.yaml, XAUUSD pip 0.1): spread {costs.spread:.2f}, slippage {costs.slippage:.2f}"
          f" per fill, commission {costs.commission_per_unit:.3f}/oz round turn; swaps not modelled")
    m15 = DataStore(a.data_dir).load("XAUUSD", "M15", end=a.end)
    print(f"XAUUSD M15 {m15.index[0]} .. {m15.index[-1]} ({len(m15)} bars); trading from {a.start}")

    res = {}
    for name, fn in cb.BOTS.items():
        tr = fn(m15, costs=costs, balance=a.balance, trade_start=a.start)
        res[name] = tr
        print(fmt_stats(name, cb.summarize(tr, a.balance), rep.get(REPORT_KEY[name])))
        if a.validate:
            validate(name, tr, m15)

    print("\n=== same bots with 'Use Portfolio Stop Rules' = false (how the user's expected numbers were quoted) ===")
    res_off = {}
    for name, fn in cb.BOTS.items():
        tr = fn(m15, params={"Use Portfolio Stop Rules": False}, costs=costs, balance=a.balance, trade_start=a.start)
        res_off[name] = tr
        s = cb.summarize(tr, a.balance)
        print(f"  {name:<36} n {s['trades']:>4} win {s['win_rate']:.1%} PF {s['pf']:.2f} avgR {s['avg_r']:+.3f} "
              f"sumR {s['sum_r']:+.1f} ROI {s['roi_pct']:+.1f}% maxDD {s['max_dd_pct']:.1f}% by year {s['by_year']}")
    co = cb.combined_equity(res_off, a.balance)
    print(f"  combined (rules off): ROI {(co.iloc[-1] / a.balance - 1) * 100:+.1f}% max DD {cb.max_dd_pct(co):.1f}%")

    comb = cb.combined_equity(res, a.balance)
    daily = comb.resample("1D").last().dropna()
    net = comb.iloc[-1] - a.balance
    yr = pd.concat([pd.DataFrame({"exit_time": r.exit_time.to_numpy(), "net": r.net.to_numpy()}) for r in res.values()])
    yr = yr.groupby(pd.to_datetime(yr.exit_time).dt.year)["net"].sum().round(0).astype(int).to_dict()
    dpl = daily.diff().dropna()
    print(f"\n=== COMBINED (4 bots on one {a.balance:,.0f} account, sum of P&L) ===")
    print(f"  net {net:+,.0f} | ROI {net / a.balance * 100:+.1f}% | max DD {cb.max_dd_pct(comb):.1f}% (M15 equity), "
          f"{cb.max_dd_pct(daily):.1f}% (daily close) | worst day {dpl.min():+,.0f} | by year {yr}")
    dd = 1 - comb / comb.cummax()
    for lvl in (0.12, 0.20):
        hit = dd.index[dd.to_numpy() >= lvl]
        print(f"  combined DD first >= {lvl:.0%}: {hit[0] if len(hit) else 'never'}"
              f"{'  (live, the shared LocalStorage rules would pause ALL bots here)' if lvl == 0.20 and len(hit) else ''}")
    for x, y in [(a, b) for i, a in enumerate(res) for b in list(res)[i + 1:]]:
        dx, dy = (res[x].attrs["equity"].resample("1D").last().diff(), res[y].attrs["equity"].resample("1D").last().diff())
        print(f"  daily P&L corr {x[:1]} vs {y[:1]}: {dx.corr(dy):+.2f}")
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, tr in res.items():
            tr.drop(columns=["info"]).to_csv(out / f"{name}_trades.csv", index=False)
        comb.resample("1D").last().dropna().to_csv(out / "combined_daily_equity.csv")


if __name__ == "__main__":
    main()
