"""Calibration: ECE and reliability bins, the two calibrators and their fits, and the saved calibration report.

Random logits and labels appear here only as test inputs. The calibration in reports/calibration.json is
fitted on the real validation logits by src/training/calibrate.py; the last tests check that file and recompute
it from the checkpoint.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import project_path
from src.data.norm_stats import ecg_id_digest
from src.evaluation.calibration import Calibrator, calibrated_probabilities, choose, file_sha256, load_calibration
from src.evaluation.metrics import expected_calibration_error, reliability_bins
from src.training.calibrate import UNCALIBRATED, VARIANTS, check_order_preserved, fit_platt, fit_temperature, mean_bce

RANK_TOLERANCE = 1e-12  # "unchanged to within rounding"


@pytest.fixture(scope="module")
def settings(cfg) -> dict:
    return cfg["calibration"]


def _logits_and_labels(seed: int, n: int, slope: float, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from p = sigmoid(slope * z + shift), so (slope, shift) is the true Platt calibration."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.5, n)
    t = (rng.random(n) < expit(slope * z + shift)).astype(int)
    return z, t


# ------------------------------------------------------------------------------------- ECE and bins

def test_ece_hand_worked():
    # Bin [0.1, 0.2): p 0.1, 0.1 (mean 0.1), labels 0, 1 (freq 0.5): gap 0.4, weight 2/5.
    # Bin [0.8, 0.9): p 0.8, 0.8, 0.8 (mean 0.8), labels 1, 1, 0 (freq 2/3): gap 0.8 - 2/3, weight 3/5.
    t = np.array([0, 1, 1, 1, 0])
    p = np.array([0.1, 0.1, 0.8, 0.8, 0.8])
    expected = 2 / 5 * 0.4 + 3 / 5 * (0.8 - 2 / 3)
    assert expected_calibration_error(t, p, 10) == pytest.approx(expected, abs=1e-15)


def test_ece_is_zero_when_every_bin_matches_its_frequency():
    t = np.array([0, 0, 0, 1] * 5 + [1, 1, 1, 0] * 5)
    p = np.array([0.25] * 20 + [0.75] * 20)
    assert expected_calibration_error(t, p, 10) == pytest.approx(0.0, abs=1e-15)


def test_reliability_bins_edges_and_assignment():
    t = np.array([0, 1, 0, 1, 1])
    p = np.array([0.0, 0.0999, 0.5, 0.95, 1.0])
    bins = reliability_bins(t, p, 10)
    assert bins["edges"][0] == 0.0 and bins["edges"][-1] == 1.0 and len(bins["edges"]) == 11
    assert bins["count"] == [2, 0, 0, 0, 0, 1, 0, 0, 0, 2]  # 1.0 falls in the last bin, not an 11th
    assert bins["mean_predicted"][1] is None and bins["observed_frequency"][1] is None  # empty bin
    assert bins["mean_predicted"][9] == pytest.approx(0.975)
    assert bins["observed_frequency"][0] == 0.5


@pytest.mark.parametrize("t, p, n_bins, message", [
    (np.array([0, 1]), np.array([0.2, 1.3]), 10, "probabilities"),
    (np.array([0, 2]), np.array([0.2, 0.3]), 10, "0 and 1"),
    (np.array([0, 1]), np.array([0.2]), 10, "equal length"),
    (np.array([0, 1]), np.array([0.2, 0.3]), 0, "positive integer"),
])
def test_ece_rejects_malformed_input(t, p, n_bins, message):
    with pytest.raises(ValueError, match=message):
        expected_calibration_error(t, p, n_bins)


# ------------------------------------------------------------------------------------- calibrators

def test_calibrator_forms():
    z = np.array([-3.0, 0.0, 2.0])
    temperature = Calibrator.temperature(2.0)
    np.testing.assert_allclose(temperature.probabilities(z), expit(z / 2.0), rtol=0, atol=1e-15)
    assert temperature.probabilities(np.array([0.0]))[0] == 0.5  # no bias: logit 0 stays at 0.5 whatever T is
    platt = Calibrator("platt", 0.5, -1.0)
    np.testing.assert_allclose(platt.probabilities(z), expit(0.5 * z - 1.0), rtol=0, atol=1e-15)
    for cal in (temperature, platt):
        assert Calibrator.from_json(json.loads(json.dumps(cal.to_json()))) == cal


@pytest.mark.parametrize("make, message", [
    (lambda: Calibrator("platt", 0.0, 1.0), "must be > 0"),
    (lambda: Calibrator("platt", -0.5, 1.0), "must be > 0"),
    (lambda: Calibrator("temperature", 1.0, 0.3), "no bias"),
    (lambda: Calibrator("isotonic", 1.0, 0.0), "must be one of"),
    (lambda: Calibrator("platt", math.nan, 0.0), "non-finite"),
    (lambda: Calibrator.temperature(0.0), "> 0"),
])
def test_calibrator_rejects_invalid_parameters(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_temperature_json_must_be_self_consistent():
    d = Calibrator.temperature(1.5).to_json() | {"a": 0.5}
    with pytest.raises(ValueError, match="Temperature calibrator"):
        Calibrator.from_json(d)


def test_calibrated_probabilities_per_column():
    z = np.array([[0.0, 1.0], [2.0, -1.0]])
    cals = {"A": Calibrator.temperature(2.0), "B": Calibrator("platt", 1.0, 1.0)}
    p = calibrated_probabilities(z, cals, ["A", "B"])
    np.testing.assert_allclose(p, np.array([[0.5, expit(2.0)], [expit(1.0), 0.5]]), rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="Calibrators are for"):
        calibrated_probabilities(z, {"A": cals["A"]}, ["A", "B"])
    with pytest.raises(ValueError, match="logits must be"):
        calibrated_probabilities(z[:, :1], cals, ["A", "B"])


# ------------------------------------------------------------------------------------- fits

def test_platt_fit_matches_unpenalised_logistic_regression(settings):
    z, t = _logits_and_labels(seed=1, n=3000, slope=0.6, shift=-1.2)
    cal, fit = fit_platt(z, t, settings)
    ref = LogisticRegression(C=np.inf, solver="lbfgs", tol=1e-12, max_iter=10_000).fit(z[:, None], t)
    assert cal.a == pytest.approx(ref.coef_[0, 0], abs=1e-6)
    assert cal.b == pytest.approx(ref.intercept_[0], abs=1e-6)
    assert fit["max_gradient"] <= settings["max_final_gradient"]


def test_temperature_fit_matches_a_scalar_minimiser(settings):
    z, t = _logits_and_labels(seed=2, n=3000, slope=0.7, shift=0.0)
    cal, _ = fit_temperature(z, t, settings)
    ref = minimize_scalar(lambda temp: mean_bce(t, z / temp), bounds=(0.05, 20.0), method="bounded",
                          options={"xatol": 1e-10})
    assert cal.t == pytest.approx(ref.x, abs=1e-6)
    assert cal.b == 0.0


def test_fits_recover_the_generating_calibration(settings):
    z, t = _logits_and_labels(seed=3, n=200_000, slope=0.65, shift=-0.9)
    platt, _ = fit_platt(z, t, settings)
    assert platt.a == pytest.approx(0.65, abs=0.02) and platt.b == pytest.approx(-0.9, abs=0.03)
    z, t = _logits_and_labels(seed=4, n=200_000, slope=0.5, shift=0.0)
    temperature, _ = fit_temperature(z, t, settings)
    assert temperature.t == pytest.approx(2.0, rel=0.03)


def test_pos_weight_shift_needs_the_bias_term(settings):
    """A model trained with pos_weight w outputs logit(q) + log(w). Platt's bias removes the shift; T cannot."""
    rng = np.random.default_rng(5)
    q = expit(rng.normal(-2.0, 2.0, 100_000))  # true probabilities of a rare class
    t = (rng.random(len(q)) < q).astype(int)
    w = (len(t) - t.sum()) / t.sum()           # negatives / positives, as in training
    z = logit(q) + math.log(w)                 # the pos_weight-shifted logit
    platt, _ = fit_platt(z, t, settings)
    temperature, _ = fit_temperature(z, t, settings)
    assert platt.a == pytest.approx(1.0, abs=0.03)
    assert platt.b == pytest.approx(-math.log(w), abs=0.05)
    ece = {name: expected_calibration_error(t, cal.probabilities(z), 10)
           for name, cal in (("temperature", temperature), ("platt", platt))}
    assert ece["platt"] < 0.01 < 0.05 < ece["temperature"]
    assert choose(ece) == "platt"


def test_choose_prefers_temperature_on_an_exact_tie():
    assert choose({"temperature": 0.02, "platt": 0.02}) == "temperature"
    assert choose({"temperature": 0.02, "platt": 0.019}) == "platt"
    assert choose({"temperature": 0.019, "platt": 0.02}) == "temperature"


def test_logits_that_point_the_wrong_way_fail_loudly(settings):
    z, t = _logits_and_labels(seed=6, n=2000, slope=-0.8, shift=0.0)
    with pytest.raises(RuntimeError, match="no usable ranking"):
        fit_platt(z, t, settings)
    with pytest.raises(RuntimeError, match="no T > 0 gives a lower cross-entropy"):
        fit_temperature(z, t, settings)


def test_temperature_refuses_a_well_ranked_but_heavily_shifted_class(settings):
    """Every logit far above the class rate: no T helps, although the ranking is good and Platt's bias fits it."""
    rng = np.random.default_rng(7)
    t = (rng.random(4000) < 0.1).astype(int)
    z = np.where(t == 1, rng.normal(5.0, 1.5, len(t)), rng.normal(1.0, 1.5, len(t)))
    assert roc_auc_score(t, z) > 0.95 and np.mean((0.5 - t) * z) > 0
    with pytest.raises(RuntimeError, match="shifted far above its rate"):
        fit_temperature(z, t, settings)
    platt, _ = fit_platt(z, t, settings)
    assert platt.b < 0


def test_single_label_class_fails_loudly(settings):
    with pytest.raises(ValueError, match="positives and negatives"):
        fit_platt(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]), settings)


@pytest.mark.parametrize("fit", [fit_temperature, fit_platt])
def test_separable_class_fails_loudly(settings, fit):
    """AUROC 1: the cross-entropy has no minimum, though its gradient fades enough to pass the convergence check."""
    z = np.r_[np.linspace(0.5, 6, 200), np.linspace(-6, -0.5, 200)]
    t = np.r_[np.ones(200, int), np.zeros(200, int)]
    with pytest.raises(ValueError, match="do not overlap"):
        fit(z, t, settings)


def test_order_check_catches_merged_scores():
    z = np.array([40.0, 41.0, -1.0])
    check_order_preserved(z, expit(z * 0.5), "ok")  # distinct, increasing
    with pytest.raises(RuntimeError, match="order and ties"):
        check_order_preserved(z, expit(z), "saturated")  # float64 sigmoid rounds both 40 and 41 to 1.0


# ------------------------------------------------------------------------------------- AUROC unchanged

@pytest.mark.parametrize("seed, slope, shift", [(10, 0.6, -1.5), (11, 1.4, 0.8), (12, 0.3, -3.0), (13, 1.0, 0.0)])
def test_calibration_leaves_auroc_and_auprc_unchanged(settings, seed, slope, shift):
    """Both calibrators are strictly increasing, so per-class AUROC and AUPRC cannot move beyond rounding."""
    z, t = _logits_and_labels(seed, n=2500, slope=slope, shift=shift)
    p0 = expit(z)
    for fit in (fit_temperature, fit_platt):
        cal, _ = fit(z, t, settings)
        p = cal.probabilities(z)
        assert not np.allclose(p, p0)  # the probabilities did change...
        assert roc_auc_score(t, p) == pytest.approx(roc_auc_score(t, p0), abs=RANK_TOLERANCE)  # ...the ranking did not
        assert average_precision_score(t, p) == pytest.approx(average_precision_score(t, p0), abs=RANK_TOLERANCE)
        np.testing.assert_array_equal(np.argsort(p, kind="stable"), np.argsort(z, kind="stable"))


# ------------------------------------------------------------------------------------- loading

RUN = {"run_id": "r1", "checkpoint": "checkpoints/r1/best.pt", "best": {"epoch": 3}}


def _minimal_report(run: dict) -> dict:
    platt = Calibrator("platt", 0.7, -1.0).to_json()
    temperature = Calibrator.temperature(1.4).to_json()
    return {"model": {"run_id": run["run_id"], "checkpoint": run["checkpoint"], "best_epoch": run["best"]["epoch"]},
            "split": "val", "ecg_ids_sha256": "abc", "classes": ["A"],
            "per_class": {"A": {"chosen": "platt", "calibrator": platt,
                                "candidates": {"temperature": temperature | {"ece": 0.09, "fit": {}},
                                               "platt": platt | {"ece": 0.02, "fit": {}}}}}}


def test_load_calibration_checks_the_model_and_records(tmp_path):
    run = RUN
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_minimal_report(run)), encoding="utf-8")
    loaded = load_calibration(path, run, ["A"], "abc")
    assert loaded["calibrators"]["A"] == Calibrator("platt", 0.7, -1.0)
    assert loaded["sha256"] == file_sha256(path)
    with pytest.raises(ValueError, match="fitted for run r1"):
        load_calibration(path, run | {"run_id": "r2"}, ["A"], "abc")
    with pytest.raises(ValueError, match="not fitted on these validation records"):
        load_calibration(path, run, ["A"], "other")
    with pytest.raises(ValueError, match="classes"):
        load_calibration(path, run, ["A", "B"], "abc")
    with pytest.raises(FileNotFoundError, match="python -m src.training.calibrate"):
        load_calibration(tmp_path / "missing.json", run, ["A"], "abc")


@pytest.mark.parametrize("edit, message", [
    (lambda r: r["per_class"]["A"]["calibrator"].update(a=0.1), "edited after fitting"),
    (lambda r: r["per_class"]["A"]["calibrator"].update(b=3.0), "edited after fitting"),
    (lambda r: r["per_class"]["A"].update(chosen="temperature", calibrator=Calibrator.temperature(1.4).to_json()),
     "selection rule picks platt"),
])
def test_load_calibration_rejects_hand_edited_calibrators(tmp_path, edit, message):
    report = _minimal_report(RUN)
    edit(report)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_calibration(path, RUN, ["A"], "abc")


def test_file_hash_ignores_line_endings(tmp_path):
    """Git checks the reports out with CRLF on Windows and LF elsewhere; the recorded hash must match both."""
    (tmp_path / "lf.json").write_bytes(b'{\n  "a": 1\n}\n')
    (tmp_path / "crlf.json").write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    (tmp_path / "other.json").write_bytes(b'{\n  "a": 2\n}\n')
    assert file_sha256(tmp_path / "lf.json") == file_sha256(tmp_path / "crlf.json")
    assert file_sha256(tmp_path / "lf.json") != file_sha256(tmp_path / "other.json")


# ------------------------------------------------------------------------------------- the saved report

@pytest.fixture(scope="module")
def saved(cfg) -> dict:
    reports = project_path(cfg["paths"]["reports_dir"])
    return {"calibration": json.loads((reports / cfg["calibration"]["file"]).read_text(encoding="utf-8")),
            "thresholds": json.loads((reports / cfg["evaluation"]["thresholds_file"]).read_text(encoding="utf-8")),
            "metrics": json.loads((reports / cfg["evaluation"]["val_metrics_file"]).read_text(encoding="utf-8")),
            "calibration_sha256": file_sha256(reports / cfg["calibration"]["file"])}


def test_saved_calibration_leaves_validation_auroc_unchanged(saved):
    """On the real model's validation logits, per-class AUROC (and AUPRC) are the same before and after calibration."""
    report = saved["calibration"]
    for c in report["classes"]:
        cand = report["per_class"][c]["candidates"]
        for k in VARIANTS:
            assert cand[k]["auroc"] == pytest.approx(cand[UNCALIBRATED]["auroc"], abs=RANK_TOLERANCE), (c, k)
            assert cand[k]["auprc"] == pytest.approx(cand[UNCALIBRATED]["auprc"], abs=RANK_TOLERANCE), (c, k)
        # ...and the same as the AUROC the validation metrics report gives for the calibrated probabilities
        assert saved["metrics"]["ranking"]["per_class"][c]["auroc"] == pytest.approx(cand[UNCALIBRATED]["auroc"],
                                                                                     abs=RANK_TOLERANCE)


def test_saved_calibration_choice_and_parameters(cfg, saved):
    report = saved["calibration"]
    assert report["split"] == "val" and report["folds"] == cfg["splits"]["val_folds"]
    assert report["model"]["run_id"] == cfg["evaluation"]["model_run"]
    assert report["classes"] == cfg["labels"]["superclasses"]
    assert report["ece"]["bins"] == cfg["evaluation"]["reliability_bins"]
    for c in report["classes"]:
        r = report["per_class"][c]
        eces = {k: r["candidates"][k]["ece"] for k in ("temperature", "platt")}
        assert r["chosen"] == choose(eces)
        assert r["calibrator"] == {k: v for k, v in r["candidates"][r["chosen"]].items()
                                   if k in ("method", "a", "b", "temperature")}
        Calibrator.from_json(r["calibrator"])  # valid parameters
        # Platt nests temperature scaling, so on the fitted fold its cross-entropy cannot be higher.
        assert r["candidates"]["platt"]["bce"] <= r["candidates"]["temperature"]["bce"] <= r["candidates"][UNCALIBRATED]["bce"]
        assert sum(r["reliability"][UNCALIBRATED]["count"]) == report["records"]
    assert report["macro_ece"]["chosen"] == pytest.approx(
        np.mean([report["per_class"][c]["candidates"][report["per_class"][c]["chosen"]]["ece"] for c in report["classes"]]))


def test_saved_thresholds_are_post_calibration_and_name_this_calibration(saved):
    thresholds, report = saved["thresholds"], saved["calibration"]
    assert thresholds["post_calibration"] is True
    assert thresholds["calibration"]["sha256"] == saved["calibration_sha256"] == saved["metrics"]["calibration"]["sha256"]
    assert thresholds["model"]["run_id"] == report["model"]["run_id"]
    assert thresholds["ecg_ids_sha256"] == report["ecg_ids_sha256"]
    for c in report["classes"]:
        assert thresholds["calibration"]["per_class"][c] == report["per_class"][c]["calibrator"]


def test_calibration_recomputed_from_the_checkpoint_leaves_auroc_unchanged(cfg, saved, monkeypatch):
    """Re-predict fold 9 with the frozen checkpoint, apply reports/calibration.json, and compare per-class AUROC.

    Opens only the validation split. Needs the checkpoint and CUDA, like evaluate_val. The run's own best.pt is not
    committed; where it is missing (a fresh clone) the committed checkpoints/release.pt is used, which load_release
    proves to be a byte copy of it (the manifest pins both files to the same SHA-256).
    """
    from src.evaluation import evaluate_val
    from src.evaluation.evaluate_val import calibration_path, predict_validation
    from src.evaluation.release import load_release
    from src.training.compare_runs import load_run

    run_id = cfg["evaluation"]["model_run"]
    run = load_run(project_path(cfg["paths"]["runs_dir"]), run_id)
    if not project_path(run["checkpoint"]).is_file():
        manifest = load_release(cfg)  # raises unless release.pt has the pinned SHA-256
        assert manifest["run"]["run_id"] == run_id
        assert manifest["source_checkpoint"]["path"] == run["checkpoint"]
        assert manifest["source_checkpoint"]["sha256"] == manifest["release_checkpoint"]["sha256"]
        monkeypatch.setattr(evaluate_val, "load_run",
                            lambda runs_dir, rid: load_run(runs_dir, rid) | {"checkpoint": manifest["release_checkpoint"]["path"]})

    preds = predict_validation(cfg, run_id)
    loaded = load_calibration(calibration_path(cfg), preds.run, preds.run["classes"], ecg_id_digest(preds.ecg_ids))
    assert loaded["sha256"] == saved["calibration_sha256"]
    classes = preds.run["classes"]
    calibrated = calibrated_probabilities(preds.logits, loaded["calibrators"], classes)
    uncalibrated = expit(preds.logits.astype(np.float64))
    n_bins = cfg["evaluation"]["reliability_bins"]
    for j, c in enumerate(classes):
        t = preds.targets[:, j]
        assert roc_auc_score(t, calibrated[:, j]) == pytest.approx(roc_auc_score(t, uncalibrated[:, j]), abs=RANK_TOLERANCE)
        chosen = loaded["report"]["per_class"][c]["chosen"]
        # The saved ECEs are reproduced from the checkpoint: the report holds real numbers, not stale ones.
        assert expected_calibration_error(t, calibrated[:, j], n_bins) == pytest.approx(
            loaded["report"]["per_class"][c]["candidates"][chosen]["ece"], abs=1e-12)
        assert expected_calibration_error(t, uncalibrated[:, j], n_bins) == pytest.approx(
            loaded["report"]["per_class"][c]["candidates"][UNCALIBRATED]["ece"], abs=1e-12)
