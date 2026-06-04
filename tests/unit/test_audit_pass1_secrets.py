"""
tests/unit/test_audit_pass1_secrets.py
────────────────────────────────────────────────────────────────────────────
PASS-1 AUDIT — Step 1 (secret scan) REGRESSION guard. Pass 1 verified the repo
holds NO hardcoded credentials: MT5 login/password/server come from env vars
(broker/mt5_adapter.py:46-48), `.env`/`*.env` are gitignored, and the only
secret-shaped strings in tracked files are obvious placeholders.

These tests LOCK that state so a future commit cannot silently introduce a real
secret. They never print a value — a hit fails with the file:line only, so the
secret-leak rule (flag, never echo) is respected even in CI logs.

  • no tracked file matches a high-signal *real-secret* shape (long base64/hex
    tokens, AWS keys, bearer tokens, PEM private-key headers), excluding env-var
    reads and obvious placeholders.
  • credential-bearing files (broker adapter) read from os.getenv, never literals.
  • .gitignore still excludes .env / *.env so local creds can't be committed.
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    exts = (".py", ".yaml", ".yml", ".json", ".toml", ".ipynb", ".md", ".cfg",
            ".ini", ".sh", ".env.example")
    return [REPO / f for f in out if f.endswith(exts)]


# High-signal patterns for an ACTUAL secret value (not a variable name or read).
# Each requires a long, high-entropy literal so placeholders ("YOUR_..._HERE")
# and env-var lookups (os.getenv("MT5_PASSWORD")) do NOT match.
_REAL_SECRET = [
    re.compile(r"AKIA[0-9A-Z]{16}"),                              # AWS access key id
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),  # PEM private key
    re.compile(r"sk-[A-Za-z0-9]{32,}"),                          # OpenAI-style key
    re.compile(r"ghp_[A-Za-z0-9]{36}"),                          # GitHub PAT
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),                 # Slack token
    re.compile(r"(?i)(password|secret|api[_-]?key|token)\s*[:=]\s*"
               r"[\"'][A-Za-z0-9+/=_\-]{20,}[\"']"),             # assigned long literal
]

# Substrings that mark a benign match (placeholder / env read / doc), used to
# suppress false positives on the assignment pattern above.
_BENIGN = ("YOUR_", "_HERE", "example", "EXAMPLE", "placeholder", "getenv",
           "environ", "xxxx", "XXXX", "<", "redacted", "dummy", "fake")


def test_no_real_secret_literal_in_tracked_files():
    offenders = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(b in line for b in _BENIGN):
                continue
            for pat in _REAL_SECRET:
                if pat.search(line):
                    # Record file:line ONLY — never the matched value.
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}")
                    break
    assert not offenders, (
        "possible hardcoded secret(s) in tracked files (file:line only, value "
        f"intentionally NOT shown): {offenders} — rotate the credential FIRST, "
        "then remove it, gitignore the file, and move it to an env var.")


def test_broker_credentials_come_from_env_not_literals():
    """The MT5 adapter must read login/password/server from the environment, not
    embed them. Guards against a regression that hardcodes broker creds."""
    src = (REPO / "broker" / "mt5_adapter.py").read_text(encoding="utf-8")
    for var in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"):
        assert f'os.getenv("{var}")' in src or f"os.getenv('{var}')" in src, \
            f"{var} is no longer read from the environment"


def test_gitignore_excludes_env_files():
    """`.env` and `*.env` must stay gitignored so real local creds can't be
    committed (the .env.example template is explicitly re-included)."""
    gi = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gi and "*.env" in gi, ".gitignore no longer excludes .env files"
