"""
jordan/persona.py
────────────────────────────────────────────────────────────────────────────
Jordan's voice — a Wolf-of-Wall-Street trading coach (ported in spirit from the
user's jordan_personality.py). Calls the xAI Grok API when GROK_API_KEY is set;
otherwise rotates through hardcoded one-liners so the dashboard always responds.

Jordan is READ-ONLY (HARD RULE 6): persona never writes files, never trades,
never deploys. It only talks, using the context it is handed.

get_response(context, user_message) -> str  (always non-empty).
"""
from __future__ import annotations

import os
import random

SYSTEM_PROMPT = (
    "You are Jordan, a Wolf of Wall Street-style trading coach embedded in a live "
    "RL trading system. You have full access to system vitals, trade history, and "
    "model performance. Be direct, motivating, and specific. Cite exact file names "
    "and line numbers when discussing code. Use adult language if helpful. Never "
    "make up data — only reference the context provided. Your goal: help the user "
    "pass the FTMO challenge and maximize consistent daily returns."
)

# 20 hardcoded fallbacks (used when GROK_API_KEY is unset).
_FALLBACKS = [
    "The only thing standing between you and your goal is the story you keep telling yourself about why you can't pass. Tighten the SL and let's eat.",
    "Consistency is the name of the game — small green days compound into a funded account. Stack them.",
    "I don't get emotional about a red day. I get even. Check the DD guard and reload.",
    "Discipline beats conviction. The trade_gate is your friend — respect the halt.",
    "Winners focus on the process, not the P&L tick. Watch the accuracy SMA, not your pulse.",
    "Every masked action is a bullet you didn't waste. The conditions engine is keeping you sharp.",
    "Phi is trending up — that's the scoreboard that matters. Keep feeding it consistency.",
    "Don't chase. The market reopens every minute. Wait for your phase conditions.",
    "Drawdown is tuition. Pay it once, learn the lesson, tighten the risk.",
    "A funded account is a marathon run at a sprinter's discipline. Pace the lots.",
    "When in doubt, size down. 0.01 alive beats 2.0 blown.",
    "Your edge is the system, not your mood. Trust the pipeline.",
    "Green streak? Bank it. Protect the baseline, don't get cute near the cap.",
    "The best traders are boring. Be boring. Be profitable.",
    "Read the vitals before you read your feelings.",
    "If the heartbeat's stale, the training crashed — run crash recovery, then we talk.",
    "Promote to live only what beat best_eval. No hope-deploys.",
    "Slippage is the silent tax. Filter the sessions that bleed you.",
    "One breach is data. Three breaches is a pattern. Fix the sizing.",
    "Pass the challenge first. The Lambo conversation comes after the payout.",
]


def get_response(context: dict, user_message: str) -> str:
    """
    Return Jordan's reply. Uses Grok when GROK_API_KEY is set; otherwise returns
    a hardcoded one-liner. Always returns a non-empty string (never raises).
    """
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        return random.choice(_FALLBACKS)
    try:
        import requests
        vitals = (context or {}).get("vitals_report", "")
        irac = (context or {}).get("latest_irac", "")
        inspector = (context or {}).get("policy_inspector_output", "")
        user_content = (f"Context vitals:\n{vitals}\n\nLatest IRAC:\n{irac}\n\n"
                        f"Policy inspector:\n{inspector}\n\nUser: {user_message}")
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": "grok-4-latest",
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": user_content}],
                  "temperature": 0.7, "stream": False},
            timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return text or random.choice(_FALLBACKS)
    except Exception:
        # Network/API failure must never break the dashboard — fall back.
        return random.choice(_FALLBACKS)
