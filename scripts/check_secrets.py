#!/usr/bin/env python3
"""Pre-publish secret scan for Intent.

Reports file paths, line numbers, and detector names only -- never the matched
value, so the report itself is safe to paste anywhere.

    python3 scripts/check_secrets.py          # scan tracked files
    python3 scripts/check_secrets.py --staged # scan what is about to be committed

Exits 1 if anything is found. Install as a pre-commit hook with:

    ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".mp3", ".wav", ".pt", ".bin")

# High-signal detectors only: a noisy scanner gets ignored, and an ignored
# scanner catches nothing.
DETECTORS: list[tuple[str, re.Pattern[str]]] = [
    ("google-api-key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("openai-api-key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("private-key-block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("bearer-token", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}")),
    # Deepgram-style: a bare 40-char hex string sitting in a key/token field.
    (
        "hex-api-key",
        re.compile(r"(?i)(?:api[_-]?key|token|secret|password)\W{0,4}\b[0-9a-f]{32,64}\b"),
    ),
]


def _git(*args: str) -> list[str]:
    out = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def scan_text(text: str, path: str) -> list[tuple[str, int, str]]:
    findings = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pattern in DETECTORS:
            if pattern.search(line):
                findings.append((path, lineno, name))
    return findings


def tracked_but_ignored() -> list[str]:
    """Files .gitignore claims to exclude but git is tracking anyway.

    .gitignore has no effect on already-tracked files, so this is exactly how a
    config file full of keys stays in the repository while looking protected.
    """
    return _git("ls-files", "-i", "-c", "--exclude-standard")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged", action="store_true", help="scan staged content instead of the tree"
    )
    args = parser.parse_args()

    paths = _git("diff", "--cached", "--name-only", "--diff-filter=ACM") if args.staged else _git("ls-files")

    findings: list[tuple[str, int, str]] = []
    for path in paths:
        if path.endswith(SKIP_SUFFIXES) or path == "scripts/check_secrets.py":
            continue
        if args.staged:
            blob = subprocess.run(
                ["git", "show", f":{path}"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            text = blob.stdout
        else:
            file_path = ROOT / path
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
        findings.extend(scan_text(text, path))

    leaked = tracked_but_ignored()

    for path, lineno, name in findings:
        print(f"{path}:{lineno}: {name}")
    for path in leaked:
        print(f"{path}: tracked-but-gitignored (gitignore does not untrack; use git rm --cached)")

    total = len(findings) + len(leaked)
    if total:
        print(f"\n{total} finding(s). Rotate anything real, then untrack it.", file=sys.stderr)
        return 1
    print(f"clean: {len(paths)} file(s) scanned, no findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
