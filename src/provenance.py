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


def git_tree(commit: str, path: str) -> str:
    """Hash of the directory `path` at `commit`. Two commits with equal hashes have identical contents there."""
    return _git("rev-parse", f"{commit}:{path}")


def git_state() -> dict[str, str | bool]:
    """{"commit": full SHA of HEAD, "dirty": True if tracked files differ from it}. Untracked files are ignored."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def last_commit_of(path: str) -> str | None:
    """Full SHA of the last commit that changed `path` (repository-relative), or None if it was never committed."""
    return _git("log", "-1", "--format=%H", "--", path) or None


def uncommitted(paths: list[str]) -> list[str]:
    """`git status --porcelain` lines for these repository-relative paths: modified, staged or untracked files."""
    out = _git("status", "--porcelain", "--untracked-files=all", "--", *paths)
    return out.splitlines() if out else []
