"""Load config/default.yaml and resolve repository-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read a YAML config file. Missing file or a non-mapping document is an error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping, got {type(cfg).__name__}")
    return cfg


def project_path(relative: str | Path) -> Path:
    """Resolve a path from the config against the repository root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
