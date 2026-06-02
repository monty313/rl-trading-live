"""
jordan/consent_flow.py
────────────────────────────────────────────────────────────────────────────
Two-step Y/N approval (HARD RULE 6). Jordan proposes; the human approves twice.

Step 1: show IRAC. [Approve Idea] / [Reject]
Step 2 (only if Step 1 approved): show the proposed diff. [Approve Deploy] / [Cancel]

ONLY on Step-2 approval is a file written — and the ONLY file Jordan may ever
write: logs/jordan_reports/pending_patch_{timestamp}.md. No code execution, no
auto-apply. While a consent is unresolved, trade_gate consent is set False.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone


class ConsentFlow:
    def __init__(self, trade_gate=None, reports_dir="logs/jordan_reports"):
        self.gate = trade_gate
        self.reports_dir = reports_dir
        self.step1_approved = False
        self.step2_approved = False

    def begin(self):
        """Open a consent flow — blocks trades until resolved."""
        self.step1_approved = False
        self.step2_approved = False
        if self.gate is not None:
            self.gate.set_consent(False)

    def approve_idea(self):
        self.step1_approved = True

    def reject(self):
        self.step1_approved = False
        self.step2_approved = False
        if self.gate is not None:
            self.gate.set_consent(True)   # nothing changes; resume normal ops

    def approve_deploy(self, irac_text: str, proposed_diff_text: str) -> str:
        """
        Step-2 approval. Writes the ONLY file Jordan may write and returns its path.
        Requires Step-1 to have been approved first (two Y clicks).
        """
        if not self.step1_approved:
            raise PermissionError("Step 1 (Approve Idea) must be approved first.")
        self.step2_approved = True
        os.makedirs(self.reports_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = os.path.join(self.reports_dir, f"pending_patch_{ts}.md")
        with open(path, "w") as f:
            f.write(f"# Pending Patch (human review only — NOT applied)\n\n"
                    f"- timestamp: {ts}\n- user_approved: true\n\n"
                    f"## IRAC\n\n{irac_text}\n\n## Proposed Diff\n\n```diff\n"
                    f"{proposed_diff_text}\n```\n")
        if self.gate is not None:
            self.gate.set_consent(True)
        return path
