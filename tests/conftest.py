"""Shared fixtures. Real-data fixtures need `python -m src.data.download` to have been run."""

from __future__ import annotations

import pytest

from src.config import load_config
from src.data.prepare import LabelledSplits, load_labelled_splits


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def data(cfg: dict) -> LabelledSplits:
    """Labels and splits built exactly as training will build them."""
    return load_labelled_splits(cfg)
