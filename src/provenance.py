"""The git commit a result was produced from, recorded in every report so each number can be traced to code."""

from __future__ import annotations

import subprocess

from src.config import PROJECT_ROOT


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed or not on PATH; reports must record the commit they came from") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"`git {' '.join(args)}` failed in {PROJECT_ROOT}: {exc.stderr.strip()}") from exc
    return out.stdout.strip()


def git_state() -> dict[str, str | bool]:
    """{"commit": full SHA of HEAD, "dirty": True if tracked files differ from it}. Untracked files are ignored."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }
