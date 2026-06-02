"""
dashboard/pages/live_dashboard.py — Page 1 "FTMO HQ".
Daily P&L, target progress, DD gauge, PASS streak, trade meter, live accuracy
SMA chart, open positions. Auto-refresh every 2s. Import-safe.
"""
from __future__ import annotations
import json, os


def _read_accuracy(repo_root):
    try:
        with open(os.path.join(repo_root, "logs", "accuracy_sma.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def render():
    import streamlit as st
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=2000, key="live_refresh")
    except Exception:
        pass
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    st.title("FTMO HQ — Live Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Daily P&L", "+0.00%")
    c2.progress(0.0, text="Target Progress 0 / 2.5%")
    c3.metric("Daily DD", "0.00%")
    c4.metric("PASS Streak", "0")
    c5.metric("Today", "IN PROGRESS")

    acc = _read_accuracy(repo_root)
    st.subheader("Live Accuracy SMA (last 20 closed trades)")
    val = acc.get("accuracy_sma")
    if val is None:
        st.info("No closed trades yet — accuracy SMA will populate as trades close.")
    else:
        st.line_chart({"accuracy_%": [val]})
        if val < 40:
            st.error("Model accuracy declining — check Jordan")

    st.subheader("Open Positions")
    st.dataframe([], use_container_width=True)


render()
