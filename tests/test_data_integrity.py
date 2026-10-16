"""Checks on the real PTB-XL download in data/raw/. Run `python -m src.data.download` first.

These fail (not skip) when the data is missing, so a passing run always means the data was checked.
"""

from __future__ import annotations

from src.config import load_config, project_path
from src.data.download import verify
from src.data.inspect_record import pick_record
from src.data.records import load_record, read_database

CFG = load_config()
RAW_DIR = project_path(CFG["paths"]["raw_dir"])
DS = CFG["dataset"]


def test_record_and_patient_counts_match_physionet():
    counts = verify(CFG, RAW_DIR)
    assert counts == {"records": 21799, "patients": 18869}


def test_one_record_loads_as_1000_by_12():
    db = read_database(RAW_DIR, DS["database_csv"])
    ecg_id = pick_record(db, CFG["splits"]["train_folds"], None)
    signal, fields = load_record(RAW_DIR, db.loc[ecg_id, DS["record_path_column"]], DS)
    assert signal.shape == (1000, 12)
    assert fields["fs"] == 100
    assert fields["sig_name"] == ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
