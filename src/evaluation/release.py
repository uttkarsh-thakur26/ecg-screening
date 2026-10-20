"""Freeze the release model, and check later that what is loaded is still exactly what was frozen.

    python -m src.evaluation.release

The release model is the best checkpoint of evaluation.model_run (the committed config, seed 42). Freezing it
writes checkpoints/release.pt, a byte copy of that best.pt, the run's effective config next to it
(release.config_file), and reports/release.json, the committed manifest, which pins the weights by SHA-256 together
with the calibration and thresholds files they go with. release.pt and its config are the only files of checkpoints/
in git (committed after the test evaluation, so that a fresh clone can run the app); the manifest's description
still says checkpoints/ is not committed, as it did when it was generated.

Before copying anything it proves, on the validation split only, that the calibration and the thresholds were
fitted on exactly this checkpoint:
1. The run is evaluation.model_run and trained with the committed config: every setting in its run record still
   has the same value in config/default.yaml (later sessions only added settings for later stages).
2. The checkpoint's validation predictions reproduce the macro AUROC and AUPRC the run logged for its best epoch
   (evaluate_val.predict_validation).
3. reports/calibration.json names this run, checkpoint path, best epoch and these validation records
   (calibration.load_calibration). Refitting both calibrators to these logits gives the stored parameters, ECEs and
   choice, to within release.recompute_tolerance. A calibration fitted on any other checkpoint's logits would not.
4. reports/thresholds.json and reports/metrics_val.json name the same run and the calibration file's SHA-256, and
   choosing the thresholds again on the calibrated probabilities gives the stored thresholds and confusion counts.
An existing release.pt that differs from best.pt is never overwritten.

load_release() is what every later user of the release model calls (the error analysis, the one-time test
evaluation, the API): it raises unless release.pt, calibration.json and thresholds.json still have the SHA-256
values in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.norm_stats import ecg_id_digest
from src.evaluation.calibration import calibrated_probabilities, file_sha256, load_calibration
from src.evaluation.evaluate_val import (COUNTS, POINTS, ValidationPredictions, calibration_path, predict_validation,
                                         write_json)
from src.evaluation.thresholds import select_thresholds
from src.provenance import git_state
from src.training.calibrate import calibrate_class
from src.training.compare_runs import load_run, validation_logits
from src.training.train import repo_path, resolve_device, seed_everything

COMMAND = "python -m src.evaluation.release"
CHUNK_BYTES = 1 << 20


def sha256_bytes(path: Path) -> str:
    """SHA-256 of a binary file's exact bytes (a checkpoint). Text reports use calibration.file_sha256 instead."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No file {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["paths"]["reports_dir"]) / cfg["release"]["manifest_file"]


def thresholds_path(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["paths"]["reports_dir"]) / cfg["evaluation"]["thresholds_file"]


def val_metrics_path(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["paths"]["reports_dir"]) / cfg["evaluation"]["val_metrics_file"]


def read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"No report {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ checks

def changed_settings(run_cfg: dict[str, Any], cfg: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    """{"dotted.key": {"run": value, "config": value}} for every setting of the run that cfg changed or removed.

    Settings that only cfg has (sections added for later stages) are not differences.
    """
    out: dict[str, dict[str, Any]] = {}
    for k, v in run_cfg.items():
        key = f"{prefix}{k}"
        if k not in cfg:
            out[key] = {"run": v, "config": "<removed>"}
        elif isinstance(v, dict) and isinstance(cfg[k], dict):
            out |= changed_settings(v, cfg[k], f"{key}.")
        elif v != cfg[k] or type(v) is not type(cfg[k]):
            out[key] = {"run": v, "config": cfg[k]}
    return out


def check_committed_config(run: dict[str, Any], cfg: dict[str, Any]) -> None:
    changed = changed_settings(run["config"], cfg)
    if changed:
        raise ValueError(f"Run {run['run_id']} did not train with the committed config; these settings differ "
                         f"(run -> config/default.yaml): {changed}")


def check_model_block(name: str, report: dict[str, Any], run: dict[str, Any]) -> None:
    """A report's "model" block must name this run, its checkpoint path and its best epoch."""
    model = report.get("model", {})
    expected = {"run_id": run["run_id"], "checkpoint": run["checkpoint"], "best_epoch": run["best"]["epoch"]}
    wrong = {k: (model.get(k), v) for k, v in expected.items() if model.get(k) != v}
    if wrong:
        raise ValueError(f"{name} was not produced for this checkpoint: (file, expected) {wrong}")


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def verify_calibration(cfg: dict[str, Any], preds: ValidationPredictions, calibration: dict[str, Any]) -> float:
    """Refit both calibrators per class to these logits; they must give the stored parameters, ECEs and choice.

    Returns the largest absolute difference found, for the record.
    """
    tol = cfg["release"]["recompute_tolerance"]
    run, report = preds.run, calibration["report"]
    targets = np.asarray(preds.targets)
    largest = 0.0
    for j, c in enumerate(run["classes"]):
        refit = calibrate_class(preds.logits[:, j], targets[:, j], float(run["pos_weight"][c]), cfg, c)
        stored = report["per_class"][c]
        if refit["chosen"] != stored["chosen"]:
            raise ValueError(f"{c}: refitting chooses {refit['chosen']}, calibration.json has {stored['chosen']}")
        pairs = [(f"calibrator {k}", refit["calibrator"][k], stored["calibrator"][k]) for k in ("a", "b")]
        for variant, cand in stored["candidates"].items():
            pairs += [(f"{variant} {m}", refit["candidates"][variant][m], cand[m]) for m in ("ece", "bce")]
        for name, new, old in pairs:
            largest = max(largest, abs(new - old))
            if not _close(new, old, tol):
                raise ValueError(f"{c} {name}: refitting on this checkpoint's validation logits gives {new!r}, "
                                 f"calibration.json has {old!r}; it was fitted on other logits")
    return largest


def verify_thresholds(cfg: dict[str, Any], preds: ValidationPredictions, calibration: dict[str, Any],
                      thresholds_report: dict[str, Any]) -> float:
    """Re-choose both operating points on the calibrated probabilities; they must equal thresholds.json.

    Returns the largest absolute threshold difference found.
    """
    tol = cfg["release"]["recompute_tolerance"]
    e = cfg["evaluation"]
    classes = preds.run["classes"]
    if thresholds_report["calibration"]["sha256"] != calibration["sha256"]:
        raise ValueError("thresholds.json was chosen with a different calibration file than the current one")
    if thresholds_report["ecg_ids_sha256"] != ecg_id_digest(preds.ecg_ids) or thresholds_report["split"] != "val":
        raise ValueError("thresholds.json was not chosen on these validation records")
    probs = calibrated_probabilities(preds.logits, calibration["calibrators"], classes)
    selection = select_thresholds(preds.targets, probs, classes, e["normal_class"], e["high_sensitivity_target"])
    largest = 0.0
    for point in POINTS:
        stored = thresholds_report["operating_points"][point]
        if sorted(stored["thresholds"]) != sorted(selection[point]):
            raise ValueError(f"{point}: thresholds.json covers {sorted(stored['thresholds'])}, re-choosing covers "
                             f"{sorted(selection[point])}")
        for c, chosen in selection[point].items():
            largest = max(largest, abs(chosen["threshold"] - stored["thresholds"][c]))
            if not _close(chosen["threshold"], stored["thresholds"][c], tol):
                raise ValueError(f"{point} {c}: re-choosing gives threshold {chosen['threshold']!r}, thresholds.json "
                                 f"has {stored['thresholds'][c]!r}")
            if any(chosen[k] != stored["selection"][c][k] for k in COUNTS):
                raise ValueError(f"{point} {c}: re-choosing counts {[chosen[k] for k in COUNTS]}, thresholds.json "
                                 f"{[stored['selection'][c][k] for k in COUNTS]} (tp, fp, fn, tn)")
    return largest


# ------------------------------------------------------------------------------------------------ freeze

def freeze(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run every check, copy best.pt to the release path, write its config and the manifest. Returns the manifest."""
    r = cfg["release"]
    git = git_state()  # before anything is written
    run_id = cfg["evaluation"]["model_run"]
    run = load_run(project_path(cfg["paths"]["runs_dir"]), run_id)
    check_committed_config(run, cfg)

    preds = predict_validation(cfg, run_id)  # reproduces the logged validation macro AUROC and AUPRC
    val_digest = ecg_id_digest(preds.ecg_ids)
    calibration = load_calibration(calibration_path(cfg), run, run["classes"], val_digest)
    check_model_block("calibration.json", calibration["report"], run)
    calibration_diff = verify_calibration(cfg, preds, calibration)

    thresholds_report = read_json(thresholds_path(cfg))
    check_model_block("thresholds.json", thresholds_report, run)
    threshold_diff = verify_thresholds(cfg, preds, calibration, thresholds_report)
    val_metrics = read_json(val_metrics_path(cfg))
    check_model_block("metrics_val.json", val_metrics, run)
    if val_metrics["calibration"]["sha256"] != calibration["sha256"]:
        raise ValueError("metrics_val.json was computed with a different calibration file than the current one")

    source = project_path(run["checkpoint"])
    ckpt = torch.load(source, map_location="cpu", weights_only=True)
    if ckpt["run_id"] != run_id or ckpt["epoch"] != run["best"]["epoch"]:
        raise ValueError(f"{run['checkpoint']} holds run {ckpt['run_id']} epoch {ckpt['epoch']}, the run record says "
                         f"{run_id} epoch {run['best']['epoch']}")
    source_sha = sha256_bytes(source)
    release = project_path(r["checkpoint"])
    if release.exists():
        if sha256_bytes(release) != source_sha:
            raise FileExistsError(f"{release} exists and differs from {source}. A frozen release is never overwritten; "
                                  "move it away by hand if a new release is really intended.")
    else:
        release.parent.mkdir(parents=True, exist_ok=True)
        tmp = release.with_suffix(".tmp")
        shutil.copyfile(source, tmp)
        tmp.replace(release)
    if sha256_bytes(release) != source_sha:
        raise RuntimeError(f"{release} does not have the SHA-256 of {source} after copying")
    config_file = project_path(r["config_file"])
    config_file.write_text(f"# Effective config of run {run_id}, the release model {repo_path(release)}.\n"
                           + yaml.safe_dump(run["config"], sort_keys=False), encoding="utf-8")

    tol = r["recompute_tolerance"]
    manifest = {
        "description": "The frozen release model. checkpoints/ is not committed: this manifest pins the weights by "
                       "SHA-256, with the calibration and thresholds files they go with. Any user of the release "
                       "model (the error analysis, the one-time test evaluation, the API) checks these hashes first.",
        "generated_by": " ".join([COMMAND, *sys.argv[1:]]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "model_version": r["model_version"],
        "tag": r["tag"],
        "release_checkpoint": {"path": repo_path(release), "sha256": source_sha, "bytes": release.stat().st_size},
        "source_checkpoint": {"path": run["checkpoint"], "sha256": source_sha},
        "config_file": repo_path(config_file),
        "run": {"run_id": run_id, "seed": run["seed"], "best_epoch": run["best"]["epoch"],
                "epochs_run": run["epochs_run"], "training_commit": run["git"]["commit"],
                "overrides": run["overrides"], "classes": run["classes"],
                "val_macro": {m: run["best"]["macro"][m] for m in ("auroc", "auprc")}},
        "calibration": {"file": repo_path(calibration_path(cfg)), "sha256": calibration["sha256"],
                        "per_class": {c: calibration["calibrators"][c].to_json() for c in run["classes"]}},
        "thresholds": {"file": repo_path(thresholds_path(cfg)), "sha256": file_sha256(thresholds_path(cfg)),
                       **{p: thresholds_report["operating_points"][p]["thresholds"] for p in POINTS}},
        "validation": {"records": int(len(preds.ecg_ids)), "patients": int(len(np.unique(preds.patient_ids))),
                       "ecg_ids_sha256": val_digest},
        "checks": {
            "committed_config": "every setting in the run record has the same value in config/default.yaml",
            "reproduces_logged_metrics": f"validation macro AUROC and AUPRC within "
                                         f"{cfg['comparison']['reproduce_tolerance']:.0e} of the logged best epoch",
            "calibration_refit": f"refitting both calibrators per class on this checkpoint's validation logits gives "
                                 f"the stored parameters, ECEs and choice; largest difference {calibration_diff:.1e} "
                                 f"(tolerance {tol:.0e})",
            "thresholds_rechosen": f"re-choosing both operating points gives the stored thresholds (largest difference "
                                   f"{threshold_diff:.1e}) and identical confusion counts",
            "reports_name_this_checkpoint": "calibration.json, thresholds.json and metrics_val.json name this run, "
                                            "checkpoint and best epoch, and the same calibration file SHA-256",
        },
    }
    write_json(manifest_path(cfg), manifest)
    return manifest


def load_release(cfg: dict[str, Any]) -> dict[str, Any]:
    """The manifest, after checking that release.pt, calibration.json and thresholds.json are still the frozen files."""
    manifest = read_json(manifest_path(cfg))
    problems = []
    release = project_path(manifest["release_checkpoint"]["path"])
    if repo_path(release) != cfg["release"]["checkpoint"]:
        problems.append(f"the manifest names {repo_path(release)}, the config {cfg['release']['checkpoint']}")
    if manifest["run"]["run_id"] != cfg["evaluation"]["model_run"]:
        problems.append(f"the manifest is for run {manifest['run']['run_id']}, the config evaluates "
                        f"{cfg['evaluation']['model_run']}")
    if not release.is_file():
        problems.append(f"{release} is missing; freeze it with `{COMMAND}`")
    elif sha256_bytes(release) != manifest["release_checkpoint"]["sha256"]:
        problems.append(f"{release} has changed since it was frozen (SHA-256 differs)")
    for key in ("calibration", "thresholds"):
        path = project_path(manifest[key]["file"])
        if not path.is_file():
            problems.append(f"{path} is missing")
        elif file_sha256(path) != manifest[key]["sha256"]:
            problems.append(f"{path} has changed since the model was frozen (SHA-256 differs)")
    if problems:
        raise ValueError(f"The release in {manifest_path(cfg)} no longer matches the files:\n  " + "\n  ".join(problems))
    return manifest


def release_validation_predictions(cfg: dict[str, Any], manifest: dict[str, Any]) -> ValidationPredictions:
    """release.pt on the validation split, checked against the logged metrics of the run it was frozen from.

    The returned run record is the source run's, so its checkpoint path is the one the calibration and thresholds
    files name; release.pt is a byte copy of it (load_release checks the SHA-256).
    """
    run = load_run(project_path(cfg["paths"]["runs_dir"]), manifest["run"]["run_id"])
    seed_everything(run["seed"], run["config"]["training"]["deterministic"])
    device = resolve_device(cfg["training"]["device"])
    release_run = {**run, "checkpoint": manifest["release_checkpoint"]["path"]}
    logits, y, ecg_ids, patient_ids = validation_logits(release_run, device, cfg["comparison"]["reproduce_tolerance"])
    if not ecg_id_digest(ecg_ids) == run["data"]["val_ecg_ids_sha256"] == manifest["validation"]["ecg_ids_sha256"]:
        raise ValueError("The validation records differ from the ones the release model was frozen on")
    return ValidationPredictions(run, logits, y, ecg_ids, patient_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the release model after checking its calibration and "
                                                 "thresholds were fitted on it (validation split only).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    manifest = freeze(cfg)
    load_release(cfg)  # read back: the files on disk are the ones the manifest names
    if manifest["git"]["dirty"]:
        print(f"WARNING: tracked files differ from commit {manifest['git']['commit'][:7]}; the manifest records "
              "dirty=true. Commit, then rerun.")
    run = manifest["run"]
    print(f"Release model v{manifest['model_version']}: run {run['run_id']}, seed {run['seed']}, best epoch "
          f"{run['best_epoch']} of {run['epochs_run']}, trained at commit {run['training_commit'][:7]}.")
    print(f"  {manifest['release_checkpoint']['path']}  sha256 {manifest['release_checkpoint']['sha256']}")
    print(f"  {manifest['calibration']['file']}  sha256 {manifest['calibration']['sha256']}")
    print(f"  {manifest['thresholds']['file']}  sha256 {manifest['thresholds']['sha256']}")
    print()
    print("Checks passed:")
    for name, text in manifest["checks"].items():
        print(f"  {name}: {text}")
    print()
    print(f"Saved {manifest['release_checkpoint']['path']}, {manifest['config_file']} and "
          f"{repo_path(manifest_path(cfg))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
