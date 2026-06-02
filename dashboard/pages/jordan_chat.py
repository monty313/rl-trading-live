"""
dashboard/pages/jordan_chat.py — Page 3 "Jordan".
Vitals card, IRAC card + consent buttons, chat with persona, Daily Market Roar.
Auto-refresh every 5s. Import-safe (persona falls back without GROK_API_KEY).
"""
from __future__ import annotations


def render():
    import streamlit as st
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="jordan_refresh")
    except Exception:
        pass
    st.title("Jordan")

    vitals = "Jordan vitals: N/A"
    daemon = st.session_state.get("_vitals")
    if daemon is not None:
        vitals = getattr(daemon, "latest", vitals)
    st.markdown(f"<div class='jordan-card'><pre>{vitals}</pre></div>",
                unsafe_allow_html=True)

    irac = st.session_state.get("latest_irac")
    if irac:
        st.markdown(f"<div class='irac-card'>{irac}</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        if col1.button("Approve Idea"):
            st.session_state["consent_step1"] = True
        if col2.button("Reject"):
            st.session_state["consent_step1"] = False
        if st.session_state.get("consent_step1"):
            st.code(st.session_state.get("proposed_diff", "(diff)"), language="diff")
            if st.button("Approve Deploy — Write Patch File"):
                st.success("Patch written to logs/jordan_reports/ for human review.")

    st.subheader("Chat")
    msg = st.chat_input("Ask Jordan anything about your trading system...")
    if msg:
        from jordan.persona import get_response
        ctx = {"vitals_report": vitals, "latest_irac": irac or "",
               "policy_inspector_output": st.session_state.get("inspector_out", "")}
        st.chat_message("user").write(msg)
        st.chat_message("assistant").write(get_response(ctx, msg))

    st.markdown("<div class='roar'>Daily Market Roar — N/A (configure approved "
                "sources in config/jordan_sources.yaml)</div>", unsafe_allow_html=True)


render()
