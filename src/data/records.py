"""Read PTB-XL metadata and WFDB records from data/raw/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb


def read_database(raw_dir: Path, database_csv: str) -> pd.DataFrame:
    """ptbxl_database.csv indexed by ecg_id."""
    path = raw_dir / database_csv
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data.download` first.")
    return pd.read_csv(path, index_col="ecg_id")


def load_record(raw_dir: Path, relative_path: str, dataset_cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one WFDB record as a (samples, leads) array in physical units, checking rate and shape.

    relative_path is a value of the filename_lr column, e.g. "records100/00000/00001_lr".
    """
    base = raw_dir / relative_path
    for ext in (".hea", ".dat"):
        f = base.with_name(base.name + ext)
        if not f.is_file():
            raise FileNotFoundError(f"Record file missing: {f}")
    signal, fields = wfdb.rdsamp(str(base))

    fs = dataset_cfg["sampling_rate_hz"]
    if fields["fs"] != fs:
        raise ValueError(f"{relative_path}: sampling rate {fields['fs']} Hz, expected {fs} Hz")
    expected_shape = (fs * dataset_cfg["record_duration_s"], dataset_cfg["n_leads"])
    if signal.shape != expected_shape:
        raise ValueError(f"{relative_path}: signal shape {signal.shape}, expected {expected_shape}")
    if not np.isfinite(signal).all():
        raise ValueError(f"{relative_path}: signal contains NaN or infinite samples")
    return signal, fields
