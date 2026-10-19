"""The frozen release model: config comparison, tamper detection, and the committed manifest against the files.

Fake files in tmp_path appear here only as test inputs. The last test checks the real reports/release.json.
"""

from __future__ import annotations

import copy
import json

import pytest

from src.config import PROJECT_ROOT, project_path
from src.evaluation.calibration import file_sha256
from src.evaluation.release import changed_settings, load_release, sha256_bytes


def test_changed_settings_ignores_added_sections_only():
    run_cfg = {"training": {"lr": 1e-3, "seed": 42}, "model": {"stem": 64}}
    later = copy.deepcopy(run_cfg) | {"evaluation": {"model_run": "x"}}  # a later session added a section
    assert changed_settings(run_cfg, later) == {}
    later["training"]["lr"] = 2e-3
    assert changed_settings(run_cfg, later) == {"training.lr": {"run": 1e-3, "config": 2e-3}}
    del later["model"]
    assert "model" in changed_settings(run_cfg, later)  # a removed setting is a change too
    assert changed_settings({"a": 1}, {"a": 1.0}) == {"a": {"run": 1, "config": 1.0}}  # a type change counts


def test_sha256_bytes_hashes_the_exact_bytes(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"line\r\n")
    b.write_bytes(b"line\n")
    assert sha256_bytes(a) != sha256_bytes(b)  # unlike file_sha256, no line-ending normalisation for weights
    with pytest.raises(FileNotFoundError):
        sha256_bytes(tmp_path / "missing.bin")


@pytest.fixture
def fake_release(tmp_path):
    """A manifest naming three fake files, and a config pointing at them."""
    ckpt, cal, thr = tmp_path / "release.pt", tmp_path / "calibration.json", tmp_path / "thresholds.json"
    ckpt.write_bytes(b"weights")
    cal.write_text("{}\n", encoding="utf-8")
    thr.write_text("{\"t\": 1}\n", encoding="utf-8")
    manifest = {"release_checkpoint": {"path": ckpt.as_posix(), "sha256": sha256_bytes(ckpt)},
                "run": {"run_id": "run-1"},
                "calibration": {"file": cal.as_posix(), "sha256": file_sha256(cal)},
                "thresholds": {"file": thr.as_posix(), "sha256": file_sha256(thr)}}
    (tmp_path / "release.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg = {"paths": {"reports_dir": tmp_path.as_posix()},
           "release": {"manifest_file": "release.json", "checkpoint": ckpt.as_posix()},
           "evaluation": {"model_run": "run-1"}}
    return cfg, ckpt, cal, thr


def test_load_release_accepts_the_frozen_files(fake_release):
    cfg, *_ = fake_release
    assert load_release(cfg)["run"]["run_id"] == "run-1"


@pytest.mark.parametrize("victim", [1, 2, 3])
def test_load_release_refuses_any_changed_file(fake_release, victim):
    cfg, *files = fake_release
    path = files[victim - 1]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed since"):
        load_release(cfg)


def test_load_release_refuses_a_missing_checkpoint_or_another_run(fake_release):
    cfg, ckpt, *_ = fake_release
    with pytest.raises(ValueError, match="evaluates"):
        load_release(cfg | {"evaluation": {"model_run": "run-2"}})
    ckpt.unlink()
    with pytest.raises(ValueError, match="missing"):
        load_release(cfg)


def test_committed_manifest_matches_the_files_on_disk(cfg):
    """reports/release.json pins release.pt, calibration.json and thresholds.json (needs the frozen checkpoint)."""
    if not project_path(cfg["release"]["checkpoint"]).is_file():
        pytest.skip("checkpoints/release.pt is not present (checkpoints are not committed)")
    manifest = load_release(cfg)
    assert manifest["model_version"] == cfg["release"]["model_version"]
    assert manifest["run"]["run_id"] == cfg["evaluation"]["model_run"]
    assert manifest["source_checkpoint"]["sha256"] == manifest["release_checkpoint"]["sha256"]
    assert sha256_bytes(PROJECT_ROOT / manifest["source_checkpoint"]["path"]) == manifest["release_checkpoint"]["sha256"]
