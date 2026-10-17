"""Multi-seed summary of paired comparisons. The run records and reports here are minimal test fixtures, not results."""

from __future__ import annotations

import copy
import json

import pytest

import src.training.summarize_comparisons as sc

FLAG = "augmentation.amplitude_scale.enabled"


def _run(run_id: str, seed: int, enabled: bool, commit: str = "c1", dirty: bool = False, lr: float = 1e-3) -> dict:
    return {"run_id": run_id, "mode": "train", "status": "completed", "seed": seed,
            "config": {"training": {"seed": seed, "lr": lr}, "augmentation": {"amplitude_scale": {"enabled": enabled}}},
            "git": {"commit": commit, "dirty": dirty}, "data": {"val_ecg_ids_sha256": "same-records"}}


def _comparison(a: dict, b: dict, auprc_a: float, auprc_b: float) -> dict:
    return {"runs": {"a": {"run_id": a["run_id"], "seed": a["seed"]}, "b": {"run_id": b["run_id"], "seed": b["seed"]}},
            "config_differences": {FLAG: {"a": True, "b": False}},
            "metrics": {"HYP_auprc": {"a": auprc_a, "b": auprc_b, "b_minus_a": auprc_b - auprc_a,
                                      "ci_low": -0.01, "ci_high": 0.05}}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "git_tree", lambda commit, path: "tree-of-src")
    cfg = {"paths": {"runs_dir": str(tmp_path)}}
    values = {44: (0.66, 0.65), 42: (0.65, 0.67), 43: (0.64, 0.68)}  # deliberately not in seed order
    runs, paths = [], []
    for seed, (on, off) in values.items():
        a, b = _run(f"on-s{seed}", seed, True), _run(f"off-s{seed}", seed, False)
        runs += [a, b]
        path = tmp_path / f"compare_s{seed}.json"
        path.write_text(json.dumps(_comparison(a, b, on, off)), encoding="utf-8")
        paths.append(path)
    for r in runs:
        (tmp_path / f"{r['run_id']}.json").write_text(json.dumps(r), encoding="utf-8")
    return cfg, paths, tmp_path


def test_summary_gives_per_seed_differences_mean_and_sample_std(setup):
    cfg, paths, _ = setup
    m = sc.run(cfg, paths)["metrics"]["HYP_auprc"]
    assert [s["seed"] for s in m["per_seed"]] == [42, 43, 44]
    assert [round(s["b_minus_a"], 6) for s in m["per_seed"]] == [0.02, 0.04, -0.01]
    assert m["mean_b_minus_a"] == pytest.approx(0.05 / 3)
    assert m["std_b_minus_a"] == pytest.approx(0.025166, abs=1e-6)  # sample std (ddof 1) of 0.02, 0.04, -0.01
    assert (m["seeds_b_better"], m["seeds"]) == (2, 3)
    assert m["a_mean"] == pytest.approx(0.65) and m["b_mean"] == pytest.approx(2.0 / 3)


def _rewrite_run(tmp_path, run_id: str, **changes) -> None:
    path = tmp_path / f"{run_id}.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if key == "lr":
            run["config"]["training"]["lr"] = value
        else:
            run[key] = value
    path.write_text(json.dumps(run), encoding="utf-8")


def test_different_code_is_refused(setup, monkeypatch):
    cfg, paths, tmp_path = setup
    _rewrite_run(tmp_path, "off-s43", git={"commit": "c2", "dirty": False})
    monkeypatch.setattr(sc, "git_tree", lambda commit, path: f"tree-{commit}")
    with pytest.raises(ValueError, match="different src/ code"):
        sc.run(cfg, paths)


def test_same_code_from_another_commit_is_accepted(setup):
    cfg, paths, tmp_path = setup
    _rewrite_run(tmp_path, "off-s43", git={"commit": "c2", "dirty": False})  # git_tree says the code is the same
    sc.run(cfg, paths)


def test_an_arm_differing_in_more_than_the_seed_is_refused(setup):
    cfg, paths, tmp_path = setup
    _rewrite_run(tmp_path, "on-s43", lr=2e-3)
    with pytest.raises(ValueError, match="training.lr"):
        sc.run(cfg, paths)


def test_dirty_runs_are_refused(setup):
    cfg, paths, tmp_path = setup
    _rewrite_run(tmp_path, "on-s44", git={"commit": "c1", "dirty": True})
    with pytest.raises(ValueError, match="dirty"):
        sc.run(cfg, paths)


def test_repeated_seeds_and_single_comparisons_are_refused(setup):
    cfg, paths, _ = setup
    with pytest.raises(ValueError, match="not all different"):
        sc.run(cfg, [paths[0], paths[0], paths[1]])
    with pytest.raises(ValueError, match="at least 2 seeds"):
        sc.run(cfg, paths[:1])


def test_a_pair_with_two_seeds_is_refused(setup):
    cfg, paths, tmp_path = setup
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    bad = copy.deepcopy(report)
    bad["runs"]["b"] = {"run_id": "off-s42", "seed": 42}
    bad["runs"]["a"] = {"run_id": "on-s43", "seed": 43}
    paths[0].write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="differ within the pair"):
        sc.run(cfg, paths)
