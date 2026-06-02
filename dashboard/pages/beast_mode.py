"""
dashboard/pages/beast_mode.py — Page 5 "Beast Mode".
Peak equity, DD-from-peak, today P&L, trailing DD bar, Beast inputs, trade table.
Only meaningful when mode=beast. Import-safe.
"""
from __future__ import annotations


def render():
    import streamlit as st
    st.title("Beast Mode")
    if st.session_state.get("mode") != "beast":
        st.info("Beast Mode is active only when Mode = Beast (sidebar).")
    a, b, c = st.columns(3)
    a.metric("Peak Equity", "$100,000")
    b.metric("DD from Peak %", "0.00%")
    c.metric("Today P&L", "+0.00%")
    st.progress(0.0, text="Trailing DD")
    st.number_input("Beast Max DD %", value=5.0, step=0.5, key="beast_dd")
    st.number_input("Beast Profit Target %", value=10.0, step=1.0, key="beast_target")
    st.subheader("Today's Trades")
    st.dataframe([], use_container_width=True)


render()
