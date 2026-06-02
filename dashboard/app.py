"""
dashboard/app.py
────────────────────────────────────────────────────────────────────────────
Streamlit entry point for the Jordan RL Trading dashboard. Dark theme + Orbitron
font. Starts the vitals daemon once per session. Multi-page nav via st.Page.

A CONVENIENT GUI: every common action (mode toggle, target/DD/lot, emergency
halt, promote-to-live) is one click in the sidebar or a page button — no code.

Run:  streamlit run dashboard/app.py --server.port 8501 --server.headless true
Import-safe: `python -c "import dashboard.app"` must succeed with no Streamlit
server running (all st.* calls are inside main()).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Orbitron', sans-serif; }
.stApp { background: #0d0d0d; color: #e0e0e0; }
section[data-testid="stSidebar"] { background: #1a1a2e; border-right: 1px solid #333; }
.jordan-card { background: #1a1a2e; border: 1px solid #333; border-radius: 10px;
               padding: 16px; margin-bottom: 12px; }
.metric-green { color: #00ff88; } .metric-red { color: #ff3333; }
.metric-yellow { color: #ffcc00; }
.irac-card { background: #1a1a2e; border: 1px solid #ff3333; border-radius: 10px; padding: 16px; }
.roar { border: 1px solid #d4af37; border-radius: 8px; padding: 12px; font-style: italic; color: #d4af37; }
.stButton>button { border: 1px solid #333; }
</style>
"""


def _start_vitals():
    """Start the read-only vitals daemon once per session (HARD RULE 6)."""
    import streamlit as st
    if st.session_state.get("_vitals_started"):
        return
    try:
        from jordan.vitals_daemon import VitalsDaemon
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        daemon = VitalsDaemon(repo_root=repo_root, interval_sec=900)
        daemon.start()
        st.session_state["_vitals"] = daemon
        st.session_state["_vitals_started"] = True
    except Exception as exc:
        st.session_state["_vitals_error"] = str(exc)


def main():
    import streamlit as st
    st.set_page_config(page_title="Jordan RL Trading", layout="wide", page_icon="🤖")
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    _start_vitals()

    with st.sidebar:
        st.markdown("### Jordan RL Trading")
        st.markdown("**MT5:** :grey[MOCK MODE]")   # CONNECTED/DISCONNECTED set live
        mode = st.radio("Mode", ["FTMO", "Beast"], horizontal=True)
        st.number_input("Daily Target %", value=2.5, step=0.1, key="target_pct")
        st.number_input("Max DD %", value=1.0, step=0.1, key="max_dd_pct")
        st.number_input("Max Lot", value=2.0, step=0.1, key="max_lot")
        if st.button("🛑 EMERGENCY HALT", use_container_width=True):
            st.session_state["emergency_halt"] = True
            st.error("EMERGENCY HALT engaged — all trading blocked.")
        st.session_state["mode"] = mode.lower()

    # Multi-page navigation (Streamlit 1.32+ st.Page API), guarded for older versions.
    pages_dir = os.path.join(os.path.dirname(__file__), "pages")
    try:
        nav = st.navigation([
            st.Page(os.path.join(pages_dir, "live_dashboard.py"), title="FTMO HQ", default=True),
            st.Page(os.path.join(pages_dir, "model_control.py"), title="Training Control"),
            st.Page(os.path.join(pages_dir, "jordan_chat.py"), title="Jordan"),
            st.Page(os.path.join(pages_dir, "performance.py"), title="Performance"),
            st.Page(os.path.join(pages_dir, "beast_mode.py"), title="Beast Mode"),
        ])
        nav.run()
    except Exception:
        st.title("Jordan RL Trading")
        st.info("Multi-page nav requires Streamlit 1.32+. Pages live in dashboard/pages/.")


if __name__ == "__main__":
    main()
