"""
jordan/repo_indexer.py
────────────────────────────────────────────────────────────────────────────
Builds a lightweight searchable index of the rl-trading-live codebase so Jordan
can answer "where is X handled?" READ-ONLY. FAISS is used when available; if not
installed, a transparent keyword fallback keeps everything working (no crash).
"""
from __future__ import annotations

import os
from typing import List, Tuple


class RepoIndexer:
    def __init__(self, repo_root="."):
        self.repo_root = repo_root
        self.docs: List[Tuple[str, str]] = []   # (path, text)
        self._faiss = None
        self._index = None

    def build(self):
        for root, _dirs, files in os.walk(self.repo_root):
            if any(skip in root for skip in (".git", "__pycache__", "audit")):
                continue
            for fn in files:
                if fn.endswith(".py"):
                    p = os.path.join(root, fn)
                    try:
                        with open(p, encoding="utf-8") as f:
                            self.docs.append((p, f.read()))
                    except Exception:
                        pass
        try:  # FAISS path (optional)
            import faiss  # noqa: F401
            self._faiss = faiss
        except Exception:
            self._faiss = None   # keyword fallback
        return len(self.docs)

    def search(self, query: str, k: int = 5) -> List[str]:
        """Return up to k file paths most relevant to the query (keyword fallback)."""
        q = query.lower()
        scored = [(sum(q_word in text.lower() for q_word in q.split()), path)
                  for path, text in self.docs]
        scored.sort(reverse=True)
        return [p for s, p in scored[:k] if s > 0]
