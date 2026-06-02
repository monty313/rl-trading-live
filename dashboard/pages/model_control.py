"""
dashboard/pages/model_control.py — Page 2 "Training Control".
Heartbeat status, 3 checkpoint cards, Promote/Revert buttons, alert log,
manifest table. Import-safe.
"""
from __future__ import annotations
import json, os, time


def _heartbeat_status(repo_root):
    p = os.path.join(repo_root, "logs", "heartbeat_training.txt")
    if not os.path.exists(p):
        return ("red", "TRAINING CRASHED — run restart cell in Colab")
    age = time.time() - os.path.getmtime(p)
    if age < 120:
        return ("green", "TRAINING RUNNING")
    if age < 300:
        return ("yellow", f"TRAINING SLOW — last heartbeat {int(age)}s ago")
    return ("red", "TRAINING CRASHED — run restart cell in Colab")


def render():
    import streamlit as st
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    st.title("Training Control")
    color, msg = _heartbeat_status(repo_root)
    {"green": st.success, "yellow": st.warning, "red": st.error}[color](msg)

    a, b, c = st.columns(3)
    a.markdown("**latest.pt**\n\nΦ: — | ep: —")
    b.markdown("**best_eval.pt**\n\nΦ: — | pass_rate: —")
    c.markdown("**live_trading.pt**\n\nΦ: — | deployed: —")

    p, r = st.columns(2)
    if p.button("⬆ Promote Best Eval → Live", use_container_width=True):
        st.session_state["promote_requested"] = True
        st.success("Promotion requested — confirm to copy best_eval.pt → live_trading.pt")
    if r.button("↩ Revert to Previous Live", use_container_width=True):
        st.session_state["revert_requested"] = True
        st.info("Revert requested — live_trading.pt will roll back to prior checkpoint")

    st.subheader("Checkpoint Manifest")
    st.dataframe([], use_container_width=True)
    st.subheader("Recent Alerts")
    st.dataframe(st.session_state.get("alerts", [])[-20:], use_container_width=True)


render()
