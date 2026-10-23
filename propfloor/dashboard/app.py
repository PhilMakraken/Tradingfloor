"""Streamlit dashboard: streamlit run propfloor/dashboard/app.py [-- --db propfloor.sqlite]"""
from __future__ import annotations

import json
import sys

import pandas as pd
import streamlit as st

from propfloor.db import ResultsDB

db_path = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else "propfloor.sqlite"
st.set_page_config(page_title="propfloor", layout="wide")
db = ResultsDB(db_path)

st.title("propfloor - research lab & trading floor")
c = st.columns(4)
c[0].metric("Strategies tried (trials)", db.n_trials)
c[1].metric("Generation", db.get_meta("generation", 0))
c[2].metric("Validated", len(db.validated()))
acc = pd.DataFrame(db.query("SELECT * FROM floor_accounts"))
c[3].metric("Floor accounts", len(acc))

tab_floor, tab_lead, tab_val, tab_lab = st.tabs(["Trading floor", "Leaderboard", "Validated", "Lab log"])

with tab_floor:
    if acc.empty:
        st.info("No floor accounts yet - run `python -m propfloor floor run`.")
    else:
        stages = acc["stage"].value_counts()
        cc = st.columns(len(stages))
        for col, (k, v) in zip(cc, stages.items()):
            col.metric(k, int(v))
        acc["trailing_buffer"] = acc["equity"] - acc["floor"]
        acc["daily_buffer"] = acc["equity"] - acc["daily_floor"]
        acc["pnl_%"] = (acc["balance"] / acc["start_balance"] - 1) * 100
        st.dataframe(acc[["id", "stage", "status", "balance", "equity", "pnl_%", "trailing_buffer", "daily_buffer",
                          "payouts", "strategies", "violation"]], width="stretch")
        st.bar_chart(acc.set_index("id")[["trailing_buffer", "daily_buffer"]])
        ev = pd.DataFrame(db.query("SELECT * FROM floor_events ORDER BY id DESC LIMIT 300"))
        st.subheader("Events")
        st.dataframe(ev, width="stretch")

with tab_lead:
    top = pd.DataFrame(db.top(200))
    if not top.empty:
        st.dataframe(top[["id", "status", "fitness", "p_pass_is", "p_pass_oos", "dsr", "exp_is", "exp_oos", "n_is", "n_oos",
                          "origin", "description"]], width="stretch")
        hist = pd.DataFrame(db.query("SELECT generation, MAX(fitness) AS best, AVG(fitness) AS mean FROM strategies GROUP BY generation"))
        st.line_chart(hist.set_index("generation"))

with tab_val:
    val = db.validated()
    if not val:
        st.info("No strategy has cleared all anti-overfitting gates yet.")
    for r in val:
        with st.expander(f"{r['id']}  OOS P(pass both)={r['p_pass_oos']:.2f}  {r['description']}"):
            m = json.loads(r["validation"])
            co = m.get("challenge_oos", {})
            st.write({"OOS P(pass stage1)": co.get("p_pass_stage1"), "OOS P(pass both)": co.get("p_pass_all"),
                      "funded survival": co.get("p_funded_survive"), "expected payout": co.get("expected_payout"),
                      "median days": co.get("median_days"), "DSR": m.get("dsr")})
            st.write("Breach causes", co.get("breach_causes"))
            st.write("Walk-forward fold expectancy (R)", m.get("fold_expectancy"))

with tab_lab:
    rej = pd.DataFrame(db.query("SELECT reasons FROM strategies WHERE status='rejected'"))
    if not rej.empty:
        reasons = pd.Series([x for r in rej["reasons"] for x in json.loads(r)]).str.replace(r"[\d.]+", "#", regex=True)
        st.subheader("Why candidates are rejected")
        st.bar_chart(reasons.value_counts())
    crit = db.get_meta("llm_critique")
    if crit:
        st.subheader("Latest Claude critique")
        st.write(crit)
    st.dataframe(pd.DataFrame(db.query("SELECT * FROM runs ORDER BY id DESC")), width="stretch")
