"""
dashboard/pages/performance.py — Page 4 "Performance".
Equity curve, drawdown curve, PASS/FAIL bars, backtest-vs-live divergence,
phase history timeline, long accuracy SMA. Import-safe.
"""
from __future__ import annotations


def render():
    import streamlit as st
    st.title("Performance")
    c1, c2 = st.columns(2)
    c1.subheader("Equity Curve"); c1.line_chart({"equity": [100000]})
    c2.subheader("Drawdown Curve"); c2.area_chart({"dd_%": [0.0]})
    st.subheader("Daily PASS / FAIL (30d)")
    st.bar_chart({"status": [0]})
    st.subheader("Backtest vs Live Divergence")
    st.dataframe([], use_container_width=True)
    st.subheader("Phase History")
    st.dataframe([], use_container_width=True)
    st.subheader("Live Accuracy SMA (last 200 trades)")
    st.line_chart({"accuracy_%": [0]})


render()
