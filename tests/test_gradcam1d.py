"""Grad-CAM 1D unit tests. Random weights and random tensors appear only here, as test inputs; the maps in
reports/ come from the trained checkpoint on validation records (src/explain/gradcam_val.py)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.evaluation.calibration import Calibrator
from src.explain.gradcam1d import GradCAM1d, feature_geometry, interpolate, normalise
from src.explain.gradcam_val import (calibrated_head, correlations, draw_sanity_records, load_thresholds, pick_example,
                                     summarise_randomisation)
from src.models.resnet1d import build_model

B, LEADS, SAMPLES = 4, 12, 1000


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return build_model(cfg).double().eval()


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(1)
    return torch.randn(B, LEADS, SAMPLES, dtype=torch.float64)


def test_gradcam_shape_and_range(cfg, model, x):
    for c in range(len(cfg["labels"]["superclasses"])):
        r = GradCAM1d(model)(x, c)
        assert r.maps.shape == (B, SAMPLES)
        assert np.isfinite(r.maps).all()
        assert r.maps.min() >= 0.0 and r.maps.max() <= 1.0
        # Each map with any positive evidence peaks at exactly 1; one without any stays all zero.
        np.testing.assert_array_equal(r.maps.max(axis=1), np.where(r.raw_peak > 0, 1.0, 0.0))


def test_contributions_split_the_logit_over_time(model, x):
    # GAP + linear head: the pre-ReLU map sums over the time positions to exactly logit - bias.
    for c in range(model.n_classes):
        r = GradCAM1d(model)(x, c)
        np.testing.assert_allclose(r.contributions.sum(axis=1), r.logits - model.fc.bias[c].item(), rtol=1e-10, atol=1e-10)


def test_matches_a_computation_without_hooks(model, x):
    # Independent route: forward_features and autograd directly; alpha = w_c / L for this head (Grad-CAM = CAM).
    c = 2
    feats = model.forward_features(x).detach().requires_grad_(True)
    logit = model.fc(model.dropout(feats.mean(dim=-1)))[:, c]
    (grads,) = torch.autograd.grad(logit.sum(), feats)
    alpha = grads.mean(dim=-1)
    torch.testing.assert_close(alpha, (model.fc.weight[c] / feats.shape[-1]).expand_as(alpha).detach())
    cam = torch.relu((alpha[:, :, None] * feats).sum(dim=1)).detach().numpy()
    centres = 32.0 * np.arange(cam.shape[1])  # stride 2 * 2 * 1 * 2 * 2 * 2, centre of position 0 at sample 0
    expected = np.stack([np.interp(np.arange(SAMPLES), centres, row) for row in cam])
    expected /= expected.max(axis=1, keepdims=True)
    np.testing.assert_allclose(GradCAM1d(model)(x, c).maps, expected, rtol=1e-10, atol=1e-12)


def test_positive_affine_calibration_leaves_the_map_unchanged(model, x):
    c = 1
    raw = GradCAM1d(model)(x, c)
    for cal in (Calibrator("platt", 0.672, -0.944), Calibrator.temperature(1.46)):
        calibrated = GradCAM1d(calibrated_head(model, c, cal))(x, c)
        np.testing.assert_allclose(calibrated.logits, cal.a * raw.logits + cal.b, rtol=1e-12)
        np.testing.assert_allclose(calibrated.maps, raw.maps, rtol=0, atol=1e-12)
        np.testing.assert_allclose(calibrated.raw_peak, cal.a * raw.raw_peak, rtol=1e-12)
    # A negative slope is not a calibrator (it reverses the ranking), and it would select other time regions.
    flipped = copy.deepcopy(model)
    with torch.no_grad():
        flipped.fc.weight[c] *= -1.0
    assert not np.allclose(GradCAM1d(flipped)(x, c).maps, raw.maps)


def test_records_in_a_batch_do_not_affect_each_other(model, x):
    batch = GradCAM1d(model)(x, 3)
    alone = GradCAM1d(model)(x[2:3], 3)
    np.testing.assert_allclose(alone.maps[0], batch.maps[2], rtol=1e-10, atol=1e-12)


def test_frozen_weights_and_no_grad_context_give_the_same_map(model, x):
    expected = GradCAM1d(model)(x, 0).maps
    frozen = copy.deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        got = GradCAM1d(frozen)(x, 0).maps
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_no_positive_evidence_gives_an_all_zero_map(model, x):
    # The last stage ends in a ReLU, so A >= 0; with every weight of the class negative, no position adds to it.
    neg = copy.deepcopy(model)
    with torch.no_grad():
        neg.fc.weight[0] = -neg.fc.weight[0].abs()
    r = GradCAM1d(neg)(x, 0)
    assert (r.contributions <= 0).all()
    np.testing.assert_array_equal(r.maps, 0.0)
    np.testing.assert_array_equal(r.raw_peak, 0.0)


def test_feature_geometry_matches_the_input_gradient(model):
    g = feature_geometry(model, SAMPLES)
    assert (g.stride, g.offset, g.length) == (32, 0.0, 32)
    assert g.receptive_field == 859
    # For positions whose receptive field fits inside the record, the input samples that reach the position are
    # exactly the computed receptive field, centred on the computed centre.
    torch.manual_seed(2)
    x = torch.randn(1, LEADS, SAMPLES, dtype=torch.float64, requires_grad=True)
    feats = model.forward_features(x)
    half = (g.receptive_field - 1) // 2
    inside = [j for j, c in enumerate(g.centres()) if c - half >= 0 and c + half <= SAMPLES - 1]
    assert inside
    for j in inside:
        (grad,) = torch.autograd.grad(feats[0, :, j].sum(), x, retain_graph=True)
        reached = np.flatnonzero(grad.abs().sum(dim=1)[0].numpy() > 0)
        assert (reached[0], reached[-1]) == (g.centres()[j] - half, g.centres()[j] + half)


def test_interpolation_holds_the_end_values():
    centres = np.array([0.0, 32.0, 64.0])
    out = interpolate(np.array([[0.0, 1.0, 0.5]]), centres, 70)
    assert out[0, 0] == 0.0 and out[0, 16] == 0.5 and out[0, 32] == 1.0 and out[0, 48] == 0.75
    np.testing.assert_array_equal(out[0, 64:], 0.5)


def test_normalise_divides_by_the_peak_and_keeps_a_positive_minimum():
    maps, peak = normalise(np.array([[2.0, 4.0, 3.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(maps, [[0.5, 1.0, 0.75], [0.0, 0.0, 0.0]])  # min-max would give [0, 1, 0.5]
    np.testing.assert_array_equal(peak, [4.0, 0.0])


def test_a_map_positive_everywhere_is_not_stretched_to_zero(model, x):
    # Every weight of the class positive and A >= 0: every position adds to the logit, so no part of the map is 0.
    pos = copy.deepcopy(model)
    with torch.no_grad():
        pos.fc.weight[0] = pos.fc.weight[0].abs()
    r = GradCAM1d(pos)(x, 0)
    assert (r.contributions > 0).all()
    expected_min = r.contributions.min(axis=1) / r.contributions.max(axis=1)  # interpolation keeps both extremes
    np.testing.assert_allclose(r.maps.min(axis=1), expected_min, rtol=1e-12)
    assert (r.maps.min(axis=1) > 0).all()


def test_normalise_rejects_negative_or_non_finite_maps():
    with pytest.raises(ValueError, match="non-negative"):
        normalise(np.array([[0.0, -0.1]]))
    with pytest.raises(ValueError, match="finite"):
        normalise(np.array([[0.0, np.nan]]))


def test_training_mode_and_bad_class_fail_loudly(cfg, x):
    torch.manual_seed(0)
    m = build_model(cfg).double()
    with pytest.raises(RuntimeError, match="eval"):
        GradCAM1d(m.train())(x, 0)
    with pytest.raises(ValueError, match="class_index"):
        GradCAM1d(m.eval())(x, 5)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        GradCAM1d(m)(x, 0)


def test_the_hook_is_removed_even_when_the_forward_pass_fails(model, x):
    GradCAM1d(model)(x, 0)
    assert len(model.stages[-1]._forward_hooks) == 0
    with pytest.raises(ValueError, match="12"):
        GradCAM1d(model)(x[:, :11], 0)  # the model rejects 11 leads inside the hooked forward pass
    assert len(model.stages[-1]._forward_hooks) == 0


def test_correlations_are_undefined_for_a_constant_map():
    a = np.linspace(0, 1, 50)
    assert correlations(a, np.zeros(50)) == {"pearson": None, "spearman": None}
    got = correlations(a, a ** 2)
    assert got["spearman"] == pytest.approx(1.0) and 0.9 < got["pearson"] < 1.0


# --------------------------------------------------------------------- record selection and the verdict

SEL_CLASSES = ["NORM", "MI", "CD"]
SEL_THRESHOLDS = {"NORM": 0.5, "MI": 0.5, "CD": 0.5}


def test_sanity_records_are_distinct_true_positives_and_reproducible():
    y = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0]])
    p = np.array([[0.9, 0.1, 0.1], [0.8, 0.9, 0.1], [0.1, 0.4, 0.9], [0.1, 0.1, 0.6], [0.1, 0.7, 0.1]])
    for seed in range(20):
        drawn = draw_sanity_records(y, p, SEL_CLASSES, SEL_THRESHOLDS, seed)
        rows = [r for r, _ in drawn]
        assert [j for _, j in drawn] == [0, 1, 2] and len(set(rows)) == 3
        assert all(y[r, j] == 1 and p[r, j] >= 0.5 for r, j in drawn)
        assert rows[1] in (1, 4)  # row 2 has MI but is not flagged for it: a false negative is never drawn
        assert drawn == draw_sanity_records(y, p, SEL_CLASSES, SEL_THRESHOLDS, seed)
    with pytest.raises(RuntimeError, match="No true positive of MI"):
        draw_sanity_records(np.array([[1, 1, 0], [0, 0, 1]]), np.array([[0.9, 0.9, 0.1], [0.1, 0.1, 0.9]]),
                            SEL_CLASSES, SEL_THRESHOLDS, 0)  # the only MI record was already drawn for NORM


def test_example_is_the_most_confident_single_label_record_with_every_decision_correct():
    y = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0]])
    p = np.array([[0.1, 0.80, 0.1],   # eligible
                  [0.1, 0.90, 0.1],   # eligible, most confident
                  [0.1, 0.99, 0.9],   # two labels: not eligible
                  [0.1, 0.95, 0.6],   # CD wrongly flagged: not eligible
                  [0.6, 0.97, 0.1]])  # NORM wrongly flagged: not eligible
    assert pick_example(y, p, SEL_CLASSES, SEL_THRESHOLDS, "MI") == (1, 2)
    with pytest.raises(RuntimeError, match="No single-label CD"):
        pick_example(y, p, SEL_CLASSES, SEL_THRESHOLDS, "CD")


def _records(*rhos):
    return [{"random": {"0": {"spearman": r, "pearson": r}}} for r in rhos]


def test_verdict_uses_the_configured_rule_for_constant_maps():
    recs = _records(0.6, None, 0.4)
    summary, passed = summarise_randomisation(recs, ["0"], 0.5, "exclude")
    assert summary["0"]["mean_abs_spearman"] == {"exclude": pytest.approx(0.5), "zero": pytest.approx(1 / 3)}
    assert summary["0"]["defined"] == 2 and not passed  # 0.5 is not below 0.5
    assert summarise_randomisation(recs, ["0"], 0.5, "zero")[1]
    summary, passed = summarise_randomisation(_records(None, None), ["0"], 0.5, "exclude")
    assert summary["0"]["mean_abs_spearman"]["exclude"] is None and not passed  # nothing to compare is a failure
    assert summarise_randomisation(_records(-0.3, 0.1), ["0"], 0.5, "exclude")[0]["0"]["mean_abs_spearman"]["exclude"] \
        == pytest.approx(0.2)  # absolute values
    with pytest.raises(ValueError, match="undefined_correlations"):
        summarise_randomisation(recs, ["0"], 0.5, "ignore")


def test_thresholds_from_another_run_records_or_calibration_are_refused(cfg, tmp_path):
    import json
    run = {"run_id": "r1", "best": {"epoch": 3}}
    good = {"model": {"run_id": "r1", "best_epoch": 3}, "ecg_ids_sha256": "ids", "calibration": {"sha256": "cal"},
            "operating_points": {"max_f1": {"thresholds": {"NORM": 0.4}}}}
    local = copy.deepcopy(cfg)
    local["paths"]["reports_dir"] = str(tmp_path)
    path = tmp_path / cfg["evaluation"]["thresholds_file"]
    path.write_text(json.dumps(good), encoding="utf-8")
    assert load_thresholds(local, run, {"sha256": "cal", "path": "c"}, "ids", "max_f1") == {"NORM": 0.4}
    for change, message in ((("model", "run_id", "r2"), "run"), (("model", "best_epoch", 4), "run"),
                            (("calibration", "sha256", "other"), "calibration")):
        bad = copy.deepcopy(good)
        bad[change[0]][change[1]] = change[2]
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_thresholds(local, run, {"sha256": "cal", "path": "c"}, "ids", "max_f1")
    path.write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(ValueError, match="validation records"):
        load_thresholds(local, run, {"sha256": "cal", "path": "c"}, "other ids", "max_f1")
