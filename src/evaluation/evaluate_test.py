"""The one-time evaluation of the frozen release model on the test split (fold 10), exactly as reports/TEST_PLAN.md says.

    python -m src.evaluation.evaluate_test                             # the one-time run: writes reports/metrics.json
    python -m src.evaluation.evaluate_test --rehearse <out_dir>        # the same pipeline on fold 9 only, into out_dir
    python -m src.evaluation.evaluate_test --rerun-reason "<bug fix>"  # only after fixing a bug in evaluation code

This is the only module outside src/data/ allowed to open fold 10 (tests/test_dataset.py, MAY_UNLOCK_TEST). Nothing
is chosen, fitted or tuned on fold 10: the model, the calibrators and the thresholds are the frozen files named in
reports/release.json, and every one of them was chosen on fold 9.

Everything that can be checked without fold 10 is checked first, and the run stops at the first failure:
- reports/TEST_PLAN.md is committed; tracked files are unmodified and nothing under final_evaluation.guarded_paths
  (the code, the config, the tests, the plan) is untracked, so the recorded commit holds the code that ran;
- fold 10 has never been opened: no reports/metrics.json and no line in the fold-10 access log. A rerun needs
  --rerun-reason, the same plan as the first opening, and (when metrics.json exists) a new commit with the fix;
- release.pt, calibration.json and thresholds.json still have their frozen SHA-256 values (release.load_release);
- on fold 9: the release model reproduces the run's logged metrics, re-choosing the thresholds gives the frozen ones,
  the seed runs reproduce reports/seed_stability_val.json, the refitted baselines reproduce
  reports/baselines_val.json, and the scoring code below reproduces reports/metrics_val.json and the ECEs in
  reports/calibration.json. So the code that scores fold 10 is the code that produced the validation numbers, and the
  validation column next to the test numbers is recomputed here, not copied.
Only then is a line appended to the access log, and fold 10 opened. A crash after that point is on record.

On fold 10: the release model's logits, calibrated with the frozen calibrators; AUROC and AUPRC per class and macro
(all five, and the four abnormal classes); sensitivity, specificity, precision, NPV and F1 at both frozen operating
points; ECE before and after calibration with reliability bins; both baselines (fitted on folds 1-8 as before); and
for the estimated metrics a 95% percentile interval from a bootstrap that resamples test PATIENTS with replacement
(a patient's records stay together). The figures are regenerated for fold 10.

--rehearse runs the same functions with the split set to fold 9 and writes into another folder. It never opens fold
10; it exists so the prediction, baseline scoring, bootstrap, figures and report code have all run before the one
real run. Only the split name differs between the two.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from torch import nn

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.cache import open_cache
from src.data.dataset import make_loader
from src.data.norm_stats import ecg_id_digest
from src.evaluation import plots
from src.evaluation.calibration import Calibrator, file_sha256, load_calibration
from src.evaluation.evaluate_val import POINTS, calibration_path, point_labels, point_rules, write_json
from src.evaluation.metrics import (THRESHOLD_METRICS, expected_calibration_error, macro_mean, operating_point_metrics,
                                    ranking_metrics, reliability_bins)
from src.evaluation.release import (load_release, read_json, release_validation_predictions, sha256_bytes,
                                    thresholds_path, val_metrics_path, verify_thresholds)
from src.evaluation.seed_stability import report_path as seed_report_path
from src.evaluation.seed_stability import seed_stability
from src.evaluation.thresholds import abnormal_classes
from src.models.baselines import FeatureBaseline, PriorBaseline, extract_features
from src.provenance import git_state, last_commit_of, uncommitted
from src.training.compare_runs import load_run
from src.training.train import (environment, load_checkpoint, predict, probabilities, repo_path, resolve_device,
                                seed_everything)

COMMAND = "python -m src.evaluation.evaluate_test"
RANKING = ("auroc", "auprc")
ECE_VARIANTS = ("uncalibrated", "calibrated")
MACROS = ("macro", "macro_abnormal")
SEP = "/"
ECE_INTERVAL_NOTE = ("ECE is a binned, non-negative statistic and is biased upward under resampling, so its percentile "
                     "interval (and that of the calibrated-minus-uncalibrated difference) describes spread, is not a "
                     "confidence interval for the true ECE, and can lie above the point value.")


class Inputs:
    """The frozen inputs, read and checked once: manifest, run, calibration, thresholds."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.manifest = load_release(cfg)
        self.run = load_run(project_path(cfg["paths"]["runs_dir"]), self.manifest["run"]["run_id"])
        self.classes: list[str] = list(self.run["classes"])
        if self.classes != list(cfg["labels"]["superclasses"]):
            raise ValueError(f"The release model's classes {self.classes} differ from the config's")
        self.abnormal = abnormal_classes(self.classes, cfg["evaluation"]["normal_class"])
        self.calibration = load_calibration(calibration_path(cfg), self.run, self.classes,
                                            self.manifest["validation"]["ecg_ids_sha256"])
        if self.calibration["sha256"] != self.manifest["calibration"]["sha256"]:
            raise ValueError("calibration.json is not the frozen one")
        self.thresholds_report = read_json(thresholds_path(cfg))
        if self.thresholds_report["calibration"]["sha256"] != self.calibration["sha256"]:
            raise ValueError("thresholds.json was chosen with another calibration file")
        self.thresholds = {p: dict(self.thresholds_report["operating_points"][p]["thresholds"]) for p in POINTS}
        if sorted(self.thresholds["max_f1"]) != sorted(self.classes) or \
                sorted(self.thresholds["high_sensitivity"]) != sorted(self.abnormal):
            raise ValueError(f"Frozen thresholds cover {({p: sorted(t) for p, t in self.thresholds.items()})}")

    @property
    def calibrators(self) -> dict[str, Calibrator]:
        return self.calibration["calibrators"]


# ------------------------------------------------------------------------------------------------ scoring

def calibrate(logits: np.ndarray, calibrators: Mapping[str, Calibrator], classes: Sequence[str]) -> np.ndarray:
    return np.stack([calibrators[c].probabilities(logits[:, j]) for j, c in enumerate(classes)], axis=1)


def _defined(key: str, value: float | None) -> float:
    if value is None or not np.isfinite(value):
        raise ValueError(f"{key} is undefined ({value}); TEST_PLAN.md: an undefined metric stops the evaluation")
    return float(value)


def score(y: np.ndarray, p_uncal: np.ndarray, p_cal: np.ndarray, p_gbdt: np.ndarray, classes: Sequence[str],
          abnormal: Sequence[str], thresholds: Mapping[str, Mapping[str, float]], n_bins: int) -> dict[str, float]:
    """Every bootstrapped number for one set of records, as {"model/group/class/metric": value}.

    The same function gives the point estimates (all records) and each bootstrap resample, so the two always agree
    in definition. Thresholds are applied as they are, never re-chosen. Raises if any value is undefined.
    """
    out: dict[str, float] = {}

    def ranking(model: str, p: np.ndarray) -> dict[str, dict[str, float]]:
        r = ranking_metrics(y, p, classes)
        rows = {c: r["per_class"][c] for c in classes} | {"macro": r["macro"]}
        rows["macro_abnormal"] = {m: float(np.mean([r["per_class"][c][m] for c in abnormal])) for m in RANKING}
        for c, row in rows.items():
            for m in RANKING:
                out[SEP.join((model, "ranking", c, m))] = _defined(f"{model} {c} {m}", row[m])
        return rows

    cnn, gbdt = ranking("cnn", p_cal), ranking("feature_gbdt", p_gbdt)
    for c in cnn:
        for m in RANKING:
            out[SEP.join(("cnn_minus_feature_gbdt", "ranking", c, m))] = cnn[c][m] - gbdt[c][m]

    for point in POINTS:
        op = operating_point_metrics(y, p_cal, classes, thresholds[point])
        rows = dict(op["per_class"])
        if point == "max_f1":
            rows["macro"] = op["macro"]
        rows["macro_abnormal"] = macro_mean(op["per_class"], abnormal)
        for c, row in rows.items():
            for m in THRESHOLD_METRICS:
                out[SEP.join(("cnn", point, c, m))] = _defined(f"{point} {c} {m}", row[m])

    for variant, p in zip(ECE_VARIANTS, (p_uncal, p_cal)):
        eces = [expected_calibration_error(y[:, j], p[:, j], n_bins) for j in range(len(classes))]
        for c, v in zip(classes, eces):
            out[SEP.join(("cnn", "ece", variant, c))] = v
        out[SEP.join(("cnn", "ece", variant, "macro"))] = float(np.mean(eces))
    for c in [*classes, "macro"]:
        out[SEP.join(("cnn", "ece", "calibrated_minus_uncalibrated", c))] = (
            out[SEP.join(("cnn", "ece", "calibrated", c))] - out[SEP.join(("cnn", "ece", "uncalibrated", c))])
    return out


def patient_resamples(patient_ids: np.ndarray, n_resamples: int, seed: int):
    """Yield n_resamples record-index arrays: patients drawn with replacement, all records of a drawn patient kept.

    A patient drawn k times contributes each of its records k times. Same draws as compare_runs.paired_bootstrap.
    """
    patient_ids = np.asarray(patient_ids)
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    patients, record_patient = np.unique(patient_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    records = np.arange(len(patient_ids))
    for _ in range(n_resamples):
        drawn = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
        yield np.repeat(records, drawn[record_patient])


def bootstrap(score_fn: Callable[[np.ndarray], dict[str, float]], patient_ids: np.ndarray, n_resamples: int,
              ci_level: float, seed: int) -> dict[str, dict[str, float]]:
    """{key: {"value": on all records, "ci_low", "ci_high"}}: percentile interval over patient resamples."""
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    point = score_fn(np.arange(len(patient_ids)))
    samples = {k: np.empty(n_resamples) for k in point}
    for i, idx in enumerate(patient_resamples(patient_ids, n_resamples, seed)):
        values = score_fn(idx)
        if values.keys() != point.keys():
            raise RuntimeError("A resample produced a different set of metrics than the full split")
        for k, v in values.items():
            samples[k][i] = v
    alpha = 1 - ci_level
    return {k: {"value": point[k], "ci_low": float(np.quantile(s, alpha / 2)),
                "ci_high": float(np.quantile(s, 1 - alpha / 2))} for k, s in samples.items()}


def nest(flat: Mapping[str, Any]) -> dict[str, Any]:
    """{"a/b/c": v} -> {"a": {"b": {"c": v}}}."""
    out: dict[str, Any] = {}
    for key, v in flat.items():
        node = out
        *parents, leaf = key.split(SEP)
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = v
    return out


# ------------------------------------------------------------------------------------------------ checks

def compare(name: str, got: Mapping[str, float], expected: Mapping[str, float], tolerance: float) -> float:
    """Raise unless every expected value is reproduced; returns the largest absolute difference."""
    missing = sorted(set(expected) - set(got))
    if missing:
        raise RuntimeError(f"{name}: not recomputed: {missing[:5]}")
    largest = 0.0
    for k, v in expected.items():
        gap = abs(float(got[k]) - float(v))
        largest = max(largest, gap)
        if gap > tolerance:
            raise RuntimeError(f"{name}: {k} recomputed as {got[k]!r}, the report has {v!r} (difference {gap:.2e})")
    return largest


def expected_from_val_reports(val_metrics: dict[str, Any], calibration: dict[str, Any], classes: Sequence[str],
                              abnormal: Sequence[str]) -> dict[str, float]:
    """The ranking, operating-point and ECE numbers of metrics_val.json and calibration.json, under score()'s keys."""
    out: dict[str, float] = {}
    for c in [*classes, "macro"]:
        row = val_metrics["ranking"]["per_class"][c] if c in classes else val_metrics["ranking"]["macro"]
        for m in RANKING:
            out[SEP.join(("cnn", "ranking", c, m))] = row[m]
    for point in POINTS:
        op = val_metrics["operating_points"][point]
        rows = dict(op["per_class"]) | {"macro_abnormal": op["macro_abnormal"]}
        if point == "max_f1":
            rows["macro"] = op["macro"]
        for c, row in rows.items():
            for m in THRESHOLD_METRICS:
                out[SEP.join(("cnn", point, c, m))] = row[m]
    per_class = calibration["report"]["per_class"]
    for c in classes:
        out[SEP.join(("cnn", "ece", "uncalibrated", c))] = per_class[c]["candidates"]["uncalibrated"]["ece"]
        out[SEP.join(("cnn", "ece", "calibrated", c))] = per_class[c]["candidates"][per_class[c]["chosen"]]["ece"]
    macro = calibration["report"]["macro_ece"]
    out[SEP.join(("cnn", "ece", "uncalibrated", "macro"))] = macro["uncalibrated"]
    out[SEP.join(("cnn", "ece", "calibrated", "macro"))] = macro["chosen"]
    return out


def expected_from_baselines(report: dict[str, Any], classes: Sequence[str]) -> dict[str, dict[str, float]]:
    """{"prior"|"feature_gbdt": {"class/metric": value}} from baselines_val.json."""
    out = {}
    for name in ("prior", "feature_gbdt"):
        m = report["baselines"][name]["metrics"]
        out[name] = {f"{c}{SEP}{k}": (m["per_class"][c] if c in classes else m["macro"])[k]
                     for c in [*classes, "macro"] for k in RANKING}
    return out


def flat_ranking(y: np.ndarray, p: np.ndarray, classes: Sequence[str]) -> dict[str, float]:
    r = ranking_metrics(y, p, classes)
    return {f"{c}{SEP}{k}": (r["per_class"][c] if c in classes else r["macro"])[k]
            for c in [*classes, "macro"] for k in RANKING}


def seed_values(report: dict[str, Any]) -> dict[str, float]:
    """Every mean and std of a seed-stability report, and every run's macro values, under distinct keys."""
    out = {f"{k}{SEP}{s}": v[s] for k, v in report["summary"].items() for s in ("mean", "std")}
    return out | {f"{r['run_id']}{SEP}macro{SEP}{m}": r["macro"][m] for r in report["runs"] for m in RANKING}


def access_log_path(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["paths"]["reports_dir"]) / cfg["final_evaluation"]["access_log"]


def read_access_log(path: Path) -> list[dict[str, Any]]:
    """One entry per earlier opening of fold 10 (empty when it was never opened)."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_one_time(cfg: dict[str, Any], git: dict[str, Any], rerun_reason: str | None) -> dict[str, Any]:
    """Refuse to open fold 10 unless the plan and code are committed and this is the first opening (or a fix).

    Returns {"file", "sha256", "commit"} of the plan.
    """
    reports_dir = project_path(cfg["paths"]["reports_dir"])
    f = cfg["final_evaluation"]
    plan = reports_dir / f["plan_file"]
    plan_commit = last_commit_of(repo_path(plan))
    if not plan.is_file() or plan_commit is None:
        raise RuntimeError(f"{repo_path(plan)} must exist and be committed before fold 10 is opened")
    if git["dirty"]:
        raise RuntimeError("Tracked files differ from the last commit. Commit first: every number in metrics.json "
                           "must trace to a commit, and TEST_PLAN.md must be the committed one.")
    loose = uncommitted(f["guarded_paths"])
    if loose:
        raise RuntimeError(f"Uncommitted or untracked files under {f['guarded_paths']}: {loose[:5]}. The commit "
                           "recorded in metrics.json must hold the code, config, tests and plan that ran.")
    plan_sha = file_sha256(plan)
    log = read_access_log(access_log_path(cfg))
    metrics = reports_dir / f["metrics_file"]
    opened = bool(log) or metrics.exists()
    if opened and not rerun_reason:
        raise RuntimeError("Fold 10 has been opened already (see the access log and metrics.json); it is evaluated "
                           "once. Only a bug in this evaluation code justifies a rerun: fix only the bug, commit, rerun "
                           "with --rerun-reason, and record what changed and why in reports/TEST_RUN.md.")
    if rerun_reason and not opened:
        raise RuntimeError("--rerun-reason given, but fold 10 has never been opened: there is nothing to rerun")
    if log and log[0]["test_plan_sha256"] != plan_sha:
        raise RuntimeError("TEST_PLAN.md changed after fold 10 was first opened; a rerun must use the same plan")
    if rerun_reason and metrics.exists() and read_json(metrics)["git"]["commit"] == git["commit"]:
        raise RuntimeError("metrics.json already comes from this commit; a rerun follows a committed bug fix")
    return {"file": repo_path(plan), "sha256": plan_sha, "commit": plan_commit}


def record_access(cfg: dict[str, Any], git: dict[str, Any], plan: dict[str, Any], rerun_reason: str | None) -> None:
    """Append one line to the access log just before fold 10 is opened. The first line creates the file exclusively,
    so two runs started together cannot both pass as the first."""
    path = access_log_path(cfg)
    entry = {"opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git["commit"],
             "command": " ".join([COMMAND, *sys.argv[1:]]), "rerun_reason": rerun_reason,
             "test_plan_sha256": plan["sha256"]}
    with path.open("a" if rerun_reason else "x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------------------------------------ predictions

def split_predictions(cfg: dict[str, Any], inputs: Inputs,
                      split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(logits, targets, ecg_ids, patient_ids) of release.pt on one split. With split "test", fold 10 is opened here."""
    run = inputs.run
    seed_everything(run["seed"], run["config"]["training"]["deterministic"])
    device = resolve_device(cfg["training"]["device"])
    model, ckpt = load_checkpoint(project_path(inputs.manifest["release_checkpoint"]["path"]), device)
    if ckpt["run_id"] != run["run_id"] or ckpt["classes"] != inputs.classes:
        raise ValueError(f"release.pt holds run {ckpt['run_id']} with classes {ckpt['classes']}")
    run_cfg = ckpt["config"]
    if open_cache(run_cfg, build_if_missing=False).meta["signals_sha256"] != ckpt["signals_sha256"]:
        raise ValueError("The signal cache on disk is not the one the release model was trained and validated on")
    loader = make_loader(run_cfg, split, allow_test=split == "test")
    ds = loader.dataset
    if list(ds.classes) != inputs.classes:
        raise ValueError(f"{split} classes {list(ds.classes)} differ from the model's {inputs.classes}")
    pos_weight = torch.tensor([ckpt["pos_weight"][c] for c in ckpt["classes"]], device=device)
    logits, targets, _ = predict(model, loader, nn.BCEWithLogitsLoss(pos_weight=pos_weight), device)
    if not np.array_equal(targets, ds.targets):
        raise RuntimeError("Predicted targets are not the dataset's, in order")
    return logits, targets, ds.ecg_ids, ds.patient_ids


class Baselines:
    """Both baselines refitted on folds 1-8 and checked against reports/baselines_val.json on fold 9."""

    def __init__(self, cfg: dict[str, Any], val_reference: dict[str, Any]) -> None:
        tol = cfg["release"]["recompute_tolerance"]
        self.cfg = cfg
        self.classes = list(cfg["labels"]["superclasses"])
        self.cache = open_cache(cfg, build_if_missing=False)
        rows = {s: self.cache.rows(s) for s in ("train", "val")}
        features = {s: extract_features(self.cache, r, cfg, f"features {s}") for s, r in rows.items()}
        y_fit = self.cache.labels[rows["train"]]
        self.prior = PriorBaseline().fit(y_fit)
        self.gbdt = FeatureBaseline(cfg["baselines"]["gradient_boosting"], cfg["training"]["seed"]).fit(
            features["train"].X, y_fit)
        self.val_ecg_ids = self.cache.index["ecg_id"].to_numpy()[rows["val"]]
        self.y_val = self.cache.labels[rows["val"]].astype(np.int64)
        self.p_prior_val = self.prior.predict_proba(len(self.y_val))
        self.p_gbdt_val = self.gbdt.predict_proba(features["val"].X)
        expected = expected_from_baselines(val_reference, self.classes)
        gaps = [compare("prior baseline on fold 9", flat_ranking(self.y_val, self.p_prior_val, self.classes),
                        expected["prior"], tol),
                compare("feature baseline on fold 9", flat_ranking(self.y_val, self.p_gbdt_val, self.classes),
                        expected["feature_gbdt"], tol)]
        self.largest_gap = max(gaps)
        self.seconds = {s: round(f.seconds, 1) for s, f in features.items()}

    def predict(self, split: str, ecg_ids: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(prior, feature-GBDT) probabilities of the split's records. With split "test", fold 10 is read here."""
        rows = self.cache.rows(split, allow_test=split == "test")
        ids = self.cache.index["ecg_id"].to_numpy()
        if not (np.array_equal(ids[rows], ecg_ids) and np.array_equal(self.cache.labels[rows], y)):
            raise RuntimeError(f"The baselines' {split} records or labels are not the CNN's, in the same order")
        features = extract_features(self.cache, rows, self.cfg, f"features {split}")
        self.seconds[split] = round(features.seconds, 1)
        return self.prior.predict_proba(len(rows)), self.gbdt.predict_proba(features.X)


# ------------------------------------------------------------------------------------------------ report

def _point_ranking(y: np.ndarray, p: np.ndarray, classes: Sequence[str], abnormal: Sequence[str]) -> dict[str, Any]:
    r = ranking_metrics(y, p, classes)
    return {"per_class": {c: {k: r["per_class"][c][k] for k in RANKING} for c in classes},
            "macro": {k: r["macro"][k] for k in RANKING},
            "macro_abnormal": {k: float(np.mean([r["per_class"][c][k] for c in abnormal])) for k in RANKING}}


def build_report(cfg: dict[str, Any], split: str, inputs: Inputs, git: dict[str, Any], plan: dict[str, Any] | None,
                 data: dict[str, Any], results: dict[str, dict[str, float]], val_point: dict[str, float],
                 baselines: Baselines, p_prior: np.ndarray, seeds: dict[str, Any], checks: dict[str, str],
                 figures: dict[str, Path], reruns: list[dict[str, Any]], access_log: list[dict[str, Any]],
                 device: torch.device) -> dict[str, Any]:
    y, classes, abnormal = data["targets"], inputs.classes, inputs.abnormal
    e, b, ds = cfg["evaluation"], cfg["final_evaluation"]["bootstrap"], cfg["dataset"]
    tree = nest({k: dict(v) for k, v in results.items()})
    cnn = tree["cnn"]
    counts = {p: operating_point_metrics(y, data["p_cal"], classes, inputs.thresholds[p])["per_class"] for p in POINTS}
    rules = point_rules(e["high_sensitivity_target"], e["normal_class"])
    support = ranking_metrics(y, data["p_cal"], classes)["per_class"]

    ranking = {"per_class": {c: cnn["ranking"][c] | {k: support[c][k] for k in ("n_positive", "n_negative",
                                                                                  "prevalence")} for c in classes},
               "macro": cnn["ranking"]["macro"] | {"prevalence": float(np.mean([support[c]["prevalence"]
                                                                                  for c in classes]))},
               "macro_abnormal": cnn["ranking"]["macro_abnormal"] | {
                   "prevalence": float(np.mean([support[c]["prevalence"] for c in abnormal]))}}
    operating_points = {}
    for p in POINTS:
        per_class = {c: {"threshold": inputs.thresholds[p][c], **{k: counts[p][c][k] for k in ("tp", "fp", "fn", "tn")},
                         **cnn[p][c]} for c in inputs.thresholds[p]}
        operating_points[p] = {"rule": rules[p], "classes": [c for c in classes if c in inputs.thresholds[p]],
                               "per_class": per_class, **{m: cnn[p][m] for m in MACROS if m in cnn[p]}}
    operating_points["high_sensitivity"]["target_sensitivity"] = e["high_sensitivity_target"]

    n_bins = e["reliability_bins"]
    calibration_error = {
        "bins": n_bins,
        "definition": "per class, sum over equal-width bins of (records in bin / records) x |observed frequency - mean "
                      "predicted probability|; macro = mean over the five classes",
        "interval_note": ECE_INTERVAL_NOTE,
        "per_class": {c: {"calibrator": inputs.calibrators[c].to_json(),
                          **{v: cnn["ece"][v][c] for v in (*ECE_VARIANTS, "calibrated_minus_uncalibrated")},
                          "reliability": {"uncalibrated": reliability_bins(y[:, j], data["p_uncal"][:, j], n_bins),
                                          "calibrated": reliability_bins(y[:, j], data["p_cal"][:, j], n_bins)}}
                      for j, c in enumerate(classes)},
        "macro": {v: cnn["ece"][v]["macro"] for v in (*ECE_VARIANTS, "calibrated_minus_uncalibrated")},
    }
    gb = cfg["baselines"]["gradient_boosting"]
    baselines_block = {
        "fitted_on": "the training split (folds 1-8), refitted in this run exactly as for reports/baselines_val.json",
        "prior": {"description": "every record scored with each class's training-split frequency; AUROC is 0.5 and "
                                 "AUPRC the prevalence by construction, so no interval is computed",
                  "scores": {c: float(p) for c, p in zip(classes, baselines.prior.prior_)},
                  "ranking": _point_ranking(y, p_prior, classes, abnormal)},
        "feature_gbdt": {"description": "79 features per record (RR intervals from lead II beats, per-lead amplitude "
                                        f"statistics), one sklearn {sklearn.__version__} HistGradientBoostingClassifier "
                                        "per class with scikit-learn's default hyperparameters except "
                                        "early_stopping=False (a fixed max_iter; sklearn's early stopping would hold "
                                        "out records, not patients)",
                         "params": {**gb, "random_state": cfg["training"]["seed"]},
                         "ranking": {"per_class": {c: tree["feature_gbdt"]["ranking"][c] for c in classes},
                                     **{m: tree["feature_gbdt"]["ranking"][m] for m in MACROS}}},
        "cnn_minus_feature_gbdt": {"description": "paired: the same resampled patients score both models",
                                   "ranking": {"per_class": {c: tree["cnn_minus_feature_gbdt"]["ranking"][c]
                                                             for c in classes},
                                               **{m: tree["cnn_minus_feature_gbdt"]["ranking"][m] for m in MACROS}}},
    }
    val_tree = nest(val_point)
    validation = {
        "description": "The same scoring code on the validation split (fold 9) in this run: point values only. The "
                       "calibration, the thresholds and the checkpoint epoch were chosen on fold 9, so these are "
                       "optimistic; they reproduce reports/metrics_val.json and reports/calibration.json (checked).",
        "records": int(len(baselines.y_val)), "patients": data["val_patients"],
        "cnn": val_tree["cnn"], "feature_gbdt": val_tree["feature_gbdt"],
        "cnn_minus_feature_gbdt": val_tree["cnn_minus_feature_gbdt"],
        "prior": {"ranking": _point_ranking(baselines.y_val, baselines.p_prior_val, classes, abnormal)},
    }
    run, manifest = inputs.run, inputs.manifest
    is_test = split == "test"
    val_figures = read_json(val_metrics_path(cfg))["figures"]
    return {
        "description": ("The one-time evaluation of the frozen release model on the test split (fold 10), following "
                        "reports/TEST_PLAN.md. Nothing was chosen on these records: model, calibrators and thresholds "
                        "were frozen on fold 9 before fold 10 was opened. Every number below was computed in this run."
                        if is_test else
                        "REHEARSAL on the validation split (fold 9): the test pipeline run on fold 9 to check it. Not "
                        "a test result; fold 10 was not opened."),
        "generated_by": " ".join([COMMAND, *sys.argv[1:]]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date_evaluated": datetime.now(timezone.utc).date().isoformat(),
        "git": git | {"note": "the commit the code ran from; the commit adding this file and the tag are its children"},
        "test_plan": plan,
        "fold10_access_log": access_log,
        "environment": environment(device) | {"sklearn": sklearn.__version__},
        "dataset": {"name": ds["name"], "version": ds["version"], "source": ds["base_url"],
                    "sampling_rate_hz": ds["sampling_rate_hz"],
                    "preprocessing": run["data"]["preprocessing"]},
        "split": split,
        "folds": data["folds"],
        "records": int(len(y)),
        "patients": int(len(np.unique(data["patient_ids"]))),
        "ecg_ids_sha256": ecg_id_digest(data["ecg_ids"]),
        "classes": classes,
        "class_descriptions": {c: cfg["labels"]["descriptions"][c] for c in classes},
        "abnormal_classes": abnormal,
        "model": {"version": manifest["model_version"], "tag": manifest["tag"], "run_id": run["run_id"],
                  "checkpoint": manifest["release_checkpoint"]["path"],
                  "sha256": manifest["release_checkpoint"]["sha256"], "best_epoch": run["best"]["epoch"],
                  "seed": run["seed"], "training_commit": run["git"]["commit"], "overrides": run["overrides"],
                  "parameters": run["parameters"], "manifest": repo_path(project_path(cfg["paths"]["reports_dir"])
                                                                         / cfg["release"]["manifest_file"])},
        "probabilities": "calibrated: per class sigmoid(a * logit + b) with the frozen calibrator of calibration.json",
        "calibration": {"file": manifest["calibration"]["file"], "sha256": manifest["calibration"]["sha256"]},
        "thresholds": {"file": manifest["thresholds"]["file"], "sha256": manifest["thresholds"]["sha256"],
                       **inputs.thresholds},
        "decision_rule": "flag the class when its calibrated probability is >= the class's frozen threshold",
        "bootstrap": {"unit": "patient", "resamples": b["resamples"], "ci_level": b["ci_level"], "seed": b["seed"],
                      "method": "percentile: the 2.5th and 97.5th percentiles (numpy quantile, linear) of the values "
                                "over resamples; each resample draws as many patients as the split has, with "
                                "replacement, and keeps every record of each drawn patient",
                      "format": "each estimated metric is {value: on all records, ci_low, ci_high}; support, "
                                "prevalence, confusion counts, thresholds, reliability bins and the prior baseline are "
                                "point values"},
        "ranking": ranking,
        "operating_points": operating_points,
        "calibration_error": calibration_error,
        "baselines": baselines_block,
        "seed_stability": seeds,
        "validation": validation,
        "checks": checks,
        "figures": {k: repo_path(v) for k, v in figures.items()},
        "figures_validation": {"description": "not test-set figures: training curves of the release run and class "
                                              "support of the training and validation splits",
                               **{k: val_figures[k] for k in ("training_curves", "class_distribution")}},
        "reruns": reruns,
    }


def seed_block(report: dict[str, Any]) -> dict[str, Any]:
    return {"description": report["description"], "split": report["split"], "std_definition": report["std"],
            "release_run": report["release_run"],
            "runs": [{k: r[k] for k in ("run_id", "seed", "best_epoch", "macro")} for r in report["runs"]],
            "macro_auroc": report["summary"]["macro_auroc"], "macro_auprc": report["summary"]["macro_auprc"],
            "per_class": {k: v for k, v in report["summary"].items() if not k.startswith("macro_")}}


def draw_figures(cfg: dict[str, Any], inputs: Inputs, data: dict[str, Any], results: dict[str, dict[str, float]],
                 split: str, figures_dir: Path) -> dict[str, Path]:
    classes, abnormal, y = inputs.classes, inputs.abnormal, data["targets"]
    e, names = cfg["evaluation"], cfg["final_evaluation"]["figures"]
    style = {"dpi": e["figures"]["dpi"], "panel_in": e["figures"]["panel_size_in"]}
    desc = cfg["labels"]["descriptions"]
    labels = point_labels(e["high_sensitivity_target"])
    scored = {p: operating_point_metrics(y, data["p_cal"], classes, inputs.thresholds[p])["per_class"] for p in POINTS}
    drawn = {p: {"label": labels[p], "per_class": scored[p]} for p in POINTS}
    fold = ", ".join(map(str, data["folds"]))
    name = "Test" if split == "test" else "REHEARSAL on validation"
    source = (f"{name} split (fold {fold}): {len(y):,} records, {len(np.unique(data['patient_ids'])):,} patients. "
              f"Release model v{inputs.manifest['model_version']} ({inputs.run['run_id']}). Calibration and thresholds "
              "frozen on fold 9.")
    value = {m: {c: results[SEP.join(("cnn", "ranking", c, m))]["value"] for c in classes} for m in RANKING}
    paths = {
        "roc": plots.plot_roc_curves(y, data["p_cal"], classes, desc, value["auroc"], drawn,
                                     figures_dir / names["roc"], title="ROC curves per class", subtitle=source, **style),
        "pr": plots.plot_pr_curves(y, data["p_cal"], classes, desc, value["auprc"], drawn, figures_dir / names["pr"],
                                   title="Precision-recall curves per class", subtitle=source, **style),
        "confusion_max_f1": plots.plot_confusion_matrices(
            scored["max_f1"], classes, desc, figures_dir / names["confusion_max_f1"],
            title="Confusion matrices at the max-F1 threshold", subtitle=source, **style),
        "confusion_high_sensitivity": plots.plot_confusion_matrices(
            scored["high_sensitivity"], abnormal, desc, figures_dir / names["confusion_high_sensitivity"],
            title=f"Confusion matrices at the high-sensitivity threshold (chosen for sensitivity ≥ "
                  f"{e['high_sensitivity_target']:.2f} on fold 9)", subtitle=source, **style),
    }
    n_bins = e["reliability_bins"]
    annotations = {}
    for c in classes:
        cal = inputs.calibrators[c]
        params = f"T {cal.t:.2f}" if cal.method == "temperature" else f"a {cal.a:.2f}, b {cal.b:+.2f}"
        before = results[SEP.join(("cnn", "ece", "uncalibrated", c))]
        after = results[SEP.join(("cnn", "ece", "calibrated", c))]
        annotations[c] = (f"ECE {before['value']:.3f} → {after['value']:.3f}\n"
                          f"{cal.method.capitalize()}: {params}")
    paths["reliability"] = plots.plot_reliability_diagrams(
        {c: reliability_bins(y[:, j], data["p_uncal"][:, j], n_bins) for j, c in enumerate(classes)},
        {c: reliability_bins(y[:, j], data["p_cal"][:, j], n_bins) for j, c in enumerate(classes)},
        ("Uncalibrated (sigmoid of the logit)", "Calibrated (frozen calibrator, fitted on fold 9)"), classes, desc,
        annotations, figures_dir / names["reliability"],
        title=f"Reliability diagrams per class, before and after calibration ({n_bins} bins)", subtitle=source,
        dpi=style["dpi"], panel_in=style["panel_in"])
    return paths


def archive_previous(cfg: dict[str, Any], reason: str) -> dict[str, Any]:
    """Before a bug-fix rerun: copy the previous metrics.json and test figures (whatever exists) aside, with hashes."""
    reports_dir, f = project_path(cfg["paths"]["reports_dir"]), cfg["final_evaluation"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = reports_dir / f"superseded_{stamp}"
    folder.mkdir()
    sources = [reports_dir / f["metrics_file"],
               *(project_path(cfg["paths"]["figures_dir"]) / n for n in f["figures"].values())]
    files = {}
    for src in sources:
        if src.is_file():
            dest = folder / src.name
            shutil.copyfile(src, dest)
            files[repo_path(dest)] = file_sha256(dest) if dest.suffix == ".json" else sha256_bytes(dest)
    previous = read_json(reports_dir / f["metrics_file"]) if (reports_dir / f["metrics_file"]).is_file() else None
    return {"reason": reason, "archived_at": stamp, "superseded_files": files,
            "superseded_generated_at": previous["generated_at"] if previous else None,
            "superseded_commit": previous["git"]["commit"] if previous else None,
            "earlier_reruns": previous.get("reruns", []) if previous else []}


def evaluate(cfg: dict[str, Any], split: str, out_dir: Path | None, rerun_reason: str | None) -> dict[str, Any]:
    """Run the plan on `split` ("test": the one-time run; "val": a rehearsal into out_dir). Returns the report."""
    if split not in ("test", "val"):
        raise ValueError(f"split must be 'test' or 'val', got {split!r}")
    if (split == "val") != (out_dir is not None):
        raise ValueError("A rehearsal on 'val' needs an output folder; the test run writes to reports/ only")
    if split == "val" and rerun_reason:
        raise ValueError("--rerun-reason applies to the test run only")
    tol = cfg["release"]["recompute_tolerance"]
    b, e = cfg["final_evaluation"]["bootstrap"], cfg["evaluation"]
    reports_dir = project_path(cfg["paths"]["reports_dir"])
    git = git_state()  # before anything is written
    plan = check_one_time(cfg, git, rerun_reason) if split == "test" else None

    # ---- Fold 9 only: every check that does not need fold 10.
    inputs = Inputs(cfg)
    classes, abnormal = inputs.classes, inputs.abnormal
    val = release_validation_predictions(cfg, inputs.manifest)
    threshold_gap = verify_thresholds(cfg, val, inputs.calibration, inputs.thresholds_report)
    seeds_report = seed_stability(cfg)
    seed_gap = compare("seed stability", seed_values(seeds_report), seed_values(read_json(seed_report_path(cfg))), tol)
    baselines = Baselines(cfg, read_json(reports_dir / cfg["baselines"]["report_file"]))
    if not np.array_equal(baselines.val_ecg_ids, val.ecg_ids):
        raise RuntimeError("The baselines' validation records are not the CNN's, in the same order")
    y_val = np.asarray(val.targets).astype(np.int64)
    val_point = score(y_val, probabilities(val.logits), calibrate(val.logits, inputs.calibrators, classes),
                      baselines.p_gbdt_val, classes, abnormal, inputs.thresholds, e["reliability_bins"])
    val_gap = compare("validation scoring", val_point,
                      expected_from_val_reports(read_json(val_metrics_path(cfg)), inputs.calibration, classes,
                                                abnormal), tol)

    # ---- The evaluated split. For "test" this is the first access to fold 10, recorded in the access log first.
    reruns: list[dict[str, Any]] = []
    if split == "test":
        if rerun_reason:
            reruns = [archive_previous(cfg, rerun_reason)]
        record_access(cfg, git, plan, rerun_reason)
    logits, targets, ecg_ids, patient_ids = split_predictions(cfg, inputs, split)
    y = np.asarray(targets).astype(np.int64)
    p_prior, p_gbdt = baselines.predict(split, ecg_ids, y)
    if split == "val" and not (np.array_equal(logits, val.logits) and np.array_equal(p_gbdt, baselines.p_gbdt_val)):
        raise RuntimeError("Rehearsal: the evaluated-split code does not reproduce the fold-9 predictions exactly")
    folds = inputs.run["config"]["splits"]["test_folds" if split == "test" else "val_folds"]
    data = {"targets": y, "ecg_ids": ecg_ids, "patient_ids": patient_ids, "folds": list(folds),
            "p_uncal": probabilities(logits), "p_cal": calibrate(logits, inputs.calibrators, classes),
            "val_patients": int(len(np.unique(val.patient_ids)))}

    def score_idx(idx: np.ndarray) -> dict[str, float]:
        return score(y[idx], data["p_uncal"][idx], data["p_cal"][idx], p_gbdt[idx], classes, abnormal,
                     inputs.thresholds, e["reliability_bins"])

    results = bootstrap(score_idx, patient_ids, b["resamples"], b["ci_level"], b["seed"])

    out = reports_dir if split == "test" else Path(out_dir)
    figures_dir = project_path(cfg["paths"]["figures_dir"]) if split == "test" else out / "figures"
    figures = draw_figures(cfg, inputs, data, results, split, figures_dir)
    checks = {
        "test_plan_and_code_committed": "reports/TEST_PLAN.md committed; no modified tracked file and no untracked file "
                                        "under " + ", ".join(cfg["final_evaluation"]["guarded_paths"])
                                        if split == "test" else "not applicable to a rehearsal",
        "frozen_files": "release.pt, calibration.json and thresholds.json have the SHA-256 values in reports/release.json",
        "release_reproduces_validation": "release.pt reproduces the run's logged validation macro AUROC and AUPRC",
        "thresholds_rechosen_on_validation": f"re-choosing both operating points on fold 9 gives the frozen thresholds "
                                             f"(largest difference {threshold_gap:.1e})",
        "seed_stability_reproduced": f"the three seed checkpoints reproduce every mean, std and per-run macro value of "
                                     f"seed_stability_val.json (largest difference {seed_gap:.1e})",
        "baselines_reproduce_validation": f"refitted on folds 1-8, both baselines reproduce reports/baselines_val.json "
                                          f"on fold 9 (largest difference {baselines.largest_gap:.1e})",
        "scoring_reproduces_validation_reports": f"this run's scoring code on fold 9 reproduces the ranking, "
                                                 f"operating-point and ECE values of metrics_val.json and "
                                                 f"calibration.json (largest difference {val_gap:.1e})",
        "order": "all checks above passed before fold 10 was opened" if split == "test" else
                 "rehearsal: the evaluated-split code reproduced the fold-9 logits and baseline scores exactly",
    }
    device = resolve_device(cfg["training"]["device"])
    access_log = read_access_log(access_log_path(cfg)) if split == "test" else []
    report = build_report(cfg, split, inputs, git, plan, data, results, val_point, baselines, p_prior,
                          seed_block(seeds_report), checks, figures, reruns, access_log, device)
    metrics_path = out / cfg["final_evaluation"]["metrics_file"] if split == "test" else out / "metrics_rehearsal_val.json"
    write_json(metrics_path, report)
    report["_path"] = metrics_path
    return report


# ------------------------------------------------------------------------------------------------ printing

def _iv(m: Mapping[str, float]) -> str:
    return f"{m['value']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"


def format_tables(report: dict[str, Any]) -> str:
    """Markdown tables: the evaluated split with 95% intervals next to the validation values, both operating points."""
    classes = report["classes"]
    val = report["validation"]
    name = "Test" if report["split"] == "test" else "Rehearsal (fold 9)"
    lines = [f"Ranking metrics ({name}, 95% patient-bootstrap interval; validation for comparison):", "",
             f"| Class | Prevalence | AUROC {name} | AUROC val | AUPRC {name} | AUPRC val |", "|---|---|---|---|---|---|"]
    rk = report["ranking"]
    for c in [*classes, *MACROS]:
        row = rk["per_class"][c] if c in classes else rk[c]
        lines.append(f"| {c} | {row['prevalence']:.3f} | {_iv(row['auroc'])} | {val['cnn']['ranking'][c]['auroc']:.3f} | "
                     f"{_iv(row['auprc'])} | {val['cnn']['ranking'][c]['auprc']:.3f} |")
    for p in POINTS:
        op = report["operating_points"][p]
        title = "Max-F1 operating point" if p == "max_f1" else \
            f"High-sensitivity operating point (chosen for sensitivity >= {op['target_sensitivity']:.2f} on fold 9)"
        lines += ["", f"{title}:", "",
                  f"| Class | Threshold | Sensitivity {name} | Sens. val | Specificity {name} | Spec. val | "
                  f"Precision {name} | Prec. val | F1 {name} | F1 val |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c in [*op["classes"], *(m for m in MACROS if m in op)]:
            row = op["per_class"][c] if c in op["per_class"] else op[c]
            t = f"{row['threshold']:.3f}" if "threshold" in row else ""
            cells = [f"{_iv(row[m])} | {val['cnn'][p][c][m]:.3f}" for m in ("sensitivity", "specificity", "precision",
                                                                             "f1")]
            lines.append(f"| {c} | {t} | " + " | ".join(cells) + " |")
    ce = report["calibration_error"]
    lines += ["", "Expected calibration error (10 bins; intervals describe spread, see interval_note):", "",
              f"| Class | Uncalibrated {name} | Calibrated {name} | Uncal. val | Cal. val (fitted there) |",
              "|---|---|---|---|---|"]
    for c in [*classes, "macro"]:
        row = ce["per_class"][c] if c in classes else ce["macro"]
        lines.append(f"| {c} | {_iv(row['uncalibrated'])} | {_iv(row['calibrated'])} | "
                     f"{val['cnn']['ece']['uncalibrated'][c]:.3f} | {val['cnn']['ece']['calibrated'][c]:.3f} |")
    bl = report["baselines"]
    vb = {"prior": val["prior"]["ranking"]["macro"], "gbdt": val["feature_gbdt"]["ranking"]["macro"],
          "diff": val["cnn_minus_feature_gbdt"]["ranking"]["macro"]}
    lines += ["", "Baselines (ranking only; no thresholds were ever chosen for them):", "",
              f"| | Macro AUROC {name} | Macro AUPRC {name} | Macro AUROC val | Macro AUPRC val |", "|---|---|---|---|---|",
              f"| Prior | {bl['prior']['ranking']['macro']['auroc']:.3f} | {bl['prior']['ranking']['macro']['auprc']:.3f} "
              f"| {vb['prior']['auroc']:.3f} | {vb['prior']['auprc']:.3f} |",
              f"| Feature GBDT | {_iv(bl['feature_gbdt']['ranking']['macro']['auroc'])} | "
              f"{_iv(bl['feature_gbdt']['ranking']['macro']['auprc'])} | {vb['gbdt']['auroc']:.3f} | "
              f"{vb['gbdt']['auprc']:.3f} |",
              f"| CNN - GBDT (paired) | {_iv(bl['cnn_minus_feature_gbdt']['ranking']['macro']['auroc'])} | "
              f"{_iv(bl['cnn_minus_feature_gbdt']['ranking']['macro']['auprc'])} | {vb['diff']['auroc']:.3f} | "
              f"{vb['diff']['auprc']:.3f} |"]
    s = report["seed_stability"]
    lines += ["", f"Seed stability (validation, {s['std_definition']}): macro AUROC {s['macro_auroc']['mean']:.4f} +/- "
                  f"{s['macro_auroc']['std']:.4f}, macro AUPRC {s['macro_auprc']['mean']:.4f} +/- "
                  f"{s['macro_auprc']['std']:.4f}."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The one-time evaluation of the frozen release model on fold 10.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rehearse", type=Path, metavar="OUT_DIR",
                       help="run the whole pipeline on fold 9 instead, writing into OUT_DIR; fold 10 stays closed")
    group.add_argument("--rerun-reason", help="only after fixing a bug in this evaluation code: why fold 10 is scored "
                                              "again (recorded in metrics.json and the access log)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    split = "val" if args.rehearse else "test"
    report = evaluate(cfg, split, args.rehearse, args.rerun_reason)
    path = report.pop("_path")
    print(f"{report['split']} split (fold {', '.join(map(str, report['folds']))}): {report['records']:,} records, "
          f"{report['patients']:,} patients. Release model v{report['model']['version']} ({report['model']['run_id']}).")
    print()
    for name, text in report["checks"].items():
        print(f"  check {name}: {text}")
    print()
    print(format_tables(report))
    print()
    for p in [repo_path(path), *report["figures"].values()]:
        print(f"Saved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
