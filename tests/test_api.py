"""The FastAPI backend (src/api/) through FastAPI's TestClient, on the real release model and the held-out test records.

Needs the downloaded dataset, its cache and checkpoints/release.pt, like the other real-data tests. Starting the app
loads and checks the release once per module. Records that tests predict are tracked by the server, so the two truth
tests use records that no other test predicts by id (TRUTH_ROW, UPLOAD_ONLY_ROW).
"""

from __future__ import annotations

import copy
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
import wfdb
from fastapi.testclient import TestClient
from scipy.signal import resample_poly
from starlette.formparsers import MultiPartParser

from src.api import service as service_module
from src.api.main import create_app
from src.api.service import api_device, load_verified_checkpoint
from src.api.upload import resample
from src.config import project_path
from src.data.cache import open_cache
from src.evaluation.calibration import Calibrator
from src.evaluation.evaluate_test import Inputs, calibrate, split_predictions
from src.explain.gradcam1d import GradCAM1d

FRONTEND = "http://localhost:5173"  # api.cors_origins in config/default.yaml
TRUTH_ROW = 5                       # predicted only by test_truth_is_refused_until_the_record_is_predicted
UPLOAD_ONLY_ROW = 7                 # only ever uploaded, never predicted by id
UPLOAD_ROW = 1                      # its files are uploaded and it is predicted by id, to compare the two
EVAL_STRIDE = 50                    # every 50th test record is compared with the evaluation pipeline
THREAD_ROWS = (10, 11, 12, 13)
THREAD_ROUNDS = 5                   # a missing model lock fails one round about 4 times in 5


# ------------------------------------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def upload_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("api_uploads")


@pytest.fixture(scope="module")
def client(cfg, upload_root):
    api_cfg = copy.deepcopy(cfg)
    api_cfg["api"]["upload"]["tmp_dir"] = str(upload_root)  # so the tests can see that uploads are deleted
    with TestClient(create_app(api_cfg)) as c:
        yield c


@pytest.fixture(scope="module")
def service(client):
    return client.app.state.service


@pytest.fixture(scope="module")
def manifest(cfg) -> dict:
    return json.loads((project_path(cfg["paths"]["reports_dir"]) / cfg["release"]["manifest_file"]).read_text("utf-8"))


@pytest.fixture(scope="module")
def test_ids(data) -> list[str]:
    return [str(e) for e in data.ids("test")]


@pytest.fixture(scope="module")
def cache(cfg):
    return open_cache(cfg, build_if_missing=False)


def raw_record(cfg, cache, ecg_id: str) -> Path:
    """data/raw/records100/.../<id>_lr, the WFDB record without extension."""
    rel = cache.index.set_index("ecg_id").loc[int(ecg_id), "record_path"]
    return project_path(cfg["paths"]["raw_dir"]) / rel


def cached_signal(cache, ecg_id: str) -> np.ndarray:
    row = int(np.flatnonzero(cache.index["ecg_id"].to_numpy() == int(ecg_id))[0])
    return np.asarray(cache.signals[row])


def files_of(base: Path) -> dict[str, tuple[str, bytes]]:
    return {ext: (base.name + "." + ext, Path(f"{base}.{ext}").read_bytes()) for ext in ("hea", "dat")}


def write_record(folder: Path, name: str, signal: np.ndarray, fs: float, sig_name, units: str = "mV",
                 adc_gain: float = 1000.0):
    """A WFDB format-16 record of a (samples, leads) signal; by default stored as PTB-XL stores it (1 uV steps)."""
    n = signal.shape[1]
    wfdb.wrsamp(name, fs=fs, units=[units] * n, sig_name=list(sig_name), p_signal=signal, fmt=["16"] * n,
                adc_gain=[adc_gain] * n, baseline=[0] * n, write_dir=str(folder))
    return files_of(folder / name)


def post_files(client, files: dict[str, tuple[str, bytes]]):
    return client.post("/api/predict", files={k: (name, data, "application/octet-stream")
                                              for k, (name, data) in files.items()})


def by_class(body: dict) -> dict[str, dict]:
    return {p["class"]: p for p in body["predictions"]}


def assert_no_upload_left(upload_root: Path) -> None:
    assert list(upload_root.iterdir()) == []


# ------------------------------------------------------------------------------------------------ health, records

def test_health(cfg, client, service, manifest):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": manifest["model_version"],
                        "model_tag": manifest["tag"], "device": str(service.device)}
    assert service.device.type == cfg["api"]["device"]


def test_records_are_the_test_ids_only(client, test_ids):
    r = client.get("/api/records")
    assert r.status_code == 200
    records = r.json()
    assert [rec["id"] for rec in records] == test_ids          # fold 10, labelled, in ecg_id order
    assert len(records) == 2158
    assert all(rec == {"id": rec["id"], "duration_s": 10, "sampling_rate": 100} for rec in records)  # nothing else


def test_signal_is_the_raw_record(cfg, client, cache, test_ids):
    rid = test_ids[2]
    r = client.get(f"/api/records/{rid}/signal")
    assert r.status_code == 200
    body = r.json()
    assert body["leads"] == cache.meta["leads"] and body["sampling_rate"] == 100 and body["units"] == "mV"
    data = np.array(body["data"], dtype=np.float32)
    assert data.shape == (12, 1000)
    np.testing.assert_array_equal(data, cached_signal(cache, rid))      # exactly the cached float32 values
    wfdb_signal, _ = wfdb.rdsamp(str(raw_record(cfg, cache, rid)))
    np.testing.assert_array_equal(data, wfdb_signal.T.astype(np.float32))  # before normalisation: the WFDB file


@pytest.mark.parametrize("record_id", ["1", "abc", "0", "009", "-9", "99999999"])
def test_unknown_records_are_404(client, record_id):
    # "1" is a training record (fold 3): the demo serves held-out test records only
    for path in ("signal", "truth"):
        r = client.get(f"/api/records/{record_id}/{path}")
        assert r.status_code == 404 and "held-out test records" in r.json()["detail"]
    r = client.post("/api/predict", json={"record_id": record_id})
    assert r.status_code == 404


# ------------------------------------------------------------------------------------------------ prediction

def test_prediction_of_a_test_record(cfg, client, service, manifest, test_ids):
    rid = test_ids[3]
    r = client.post("/api/predict", json={"record_id": rid})
    assert r.status_code == 200
    body = r.json()
    classes = cfg["labels"]["superclasses"]
    preds = by_class(body)
    assert [p["class"] for p in body["predictions"]] == classes
    assert preds["NORM"]["threshold"] is None and preds["NORM"]["flagged"] is None  # a probability only

    frozen = manifest["thresholds"]["high_sensitivity"]
    for c in classes:
        p = preds[c]
        assert p["label"] == cfg["labels"]["descriptions"][c]
        expected = Calibrator.from_json(manifest["calibration"]["per_class"][c]).probabilities(np.array([p["logit"]]))
        assert p["probability"] == float(expected[0])  # the frozen calibrator applied to the model's logit
        if c != "NORM":
            assert p["threshold"] == frozen[c]
            assert p["flagged"] == (p["probability"] >= frozen[c])
    n = sum(preds[c]["flagged"] for c in frozen)
    assert body["flagged_count"] == n and body["abnormal_count"] == 4
    assert body["summary"] == f"{n} of 4 abnormal categories flagged"
    assert body["operating_point"] == "high_sensitivity"

    top = max(frozen, key=lambda c: preds[c]["probability"])
    assert body["attention_class"] == top
    attention = np.array(body["attention"])
    assert attention.shape == (1000,) and attention.min() >= 0 and attention.max() <= 1
    x = service.dataset[service.record_ids.index(int(rid))][0].unsqueeze(0).to(service.device)
    cam = GradCAM1d(service.model)(x, classes.index(top))
    np.testing.assert_array_equal(attention, cam.maps[0])  # the Grad-CAM map of that class for this record
    assert body["attention_raw_peak"] == float(cam.raw_peak[0])

    assert body["model_version"] == manifest["model_version"]
    assert body["inference_ms"] > 0 and body["attention_ms"] > 0
    assert body["input"] == {"source": "test_record", "record_id": rid, "sampling_rate_hz": 100.0,
                             "n_samples": 1000, "resampled": False, "checksums_verified": None}
    assert body["signal"] is None


def test_predictions_match_the_one_time_evaluation_pipeline(cfg, client):
    """The API's logits, probabilities and flags for test records equal evaluate_test's own prediction code's.

    split_predictions is the function the one-time evaluation scored fold 10 with (batches of 64); no metric is
    computed here. One record at a time cuDNN may choose another kernel, so with TF32 the logits can differ in about
    the third decimal; explain.logit_tolerance bounds that, as it does for Grad-CAM's one-record forward pass. A
    calibrated probability sigmoid(a z + b) has slope at most a / 4 in z, so it can then move by at most
    a / 4 x logit_tolerance (below 2e-3 here; the largest difference measured over all 2,158 records is 1.6e-4). Flags
    must agree wherever the evaluation's probability is further than that from the threshold.
    """
    inputs = Inputs(cfg)
    logits, _, ecg_ids, _ = split_predictions(cfg, inputs, "test")
    probs = calibrate(logits, inputs.calibrators, inputs.classes)
    tol = cfg["explain"]["logit_tolerance"]
    prob_tol = np.array([inputs.calibrators[c].a for c in inputs.classes]) / 4 * tol
    thresholds = inputs.thresholds[cfg["api"]["operating_point"]]
    batch = cfg["training"]["batch_size"]
    last_batch = len(ecg_ids) // batch * batch
    rows = sorted(set(range(0, len(ecg_ids), EVAL_STRIDE)) | {last_batch - 1, last_batch, len(ecg_ids) - 1})
    compared = 0
    for row in rows:
        body = client.post("/api/predict", json={"record_id": str(ecg_ids[row])}).json()
        np.testing.assert_allclose([p["logit"] for p in body["predictions"]], logits[row], rtol=0, atol=tol)
        assert (np.abs(np.array([p["probability"] for p in body["predictions"]]) - probs[row]) <= prob_tol).all()
        for j, p in enumerate(body["predictions"]):
            if p["class"] in thresholds and abs(probs[row, j] - thresholds[p["class"]]) > prob_tol[j]:
                assert p["flagged"] == bool(probs[row, j] >= thresholds[p["class"]])
                compared += 1
    assert compared > 0.95 * len(rows) * len(thresholds)  # almost no flag is left out as too close to call


def _all_at_once(fn, items: list) -> list:
    """fn over items, one thread each, all released together by a barrier."""
    barrier = threading.Barrier(len(items))

    def call(item):
        barrier.wait()
        return fn(item)

    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        return list(pool.map(call, items))


def test_concurrent_predictions_do_not_interfere(client, service, test_ids):
    """Grad-CAM hooks the shared model, so requests must run one at a time; results equal the sequential ones.

    Without the model lock one round fails about 4 times in 5 (the hook captures another record's activations), so
    several rounds run, on the service and through the HTTP route (run_in_threadpool).
    """
    ids = [int(test_ids[r]) for r in THREAD_ROWS]
    sequential = {e: service.predict_record(e) for e in ids}
    for _ in range(THREAD_ROUNDS):
        for e, result in zip(ids * 2, _all_at_once(service.predict_record, ids * 2)):
            np.testing.assert_array_equal(result.probabilities, sequential[e].probabilities)
            np.testing.assert_array_equal(result.attention, sequential[e].attention)
            assert result.attention_class == sequential[e].attention_class

    def post(e: int) -> dict:
        return client.post("/api/predict", json={"record_id": str(e)}).json()

    for e, body in zip(ids * 2, _all_at_once(post, ids * 2)):
        assert [p["probability"] for p in body["predictions"]] == sequential[e].probabilities.tolist()
        assert body["attention"] == sequential[e].attention.tolist()


# ------------------------------------------------------------------------------------------------ ground truth

def test_truth_is_refused_until_the_record_is_predicted(client, data, test_ids):
    rid = test_ids[TRUTH_ROW]
    r = client.get(f"/api/records/{rid}/truth")
    assert r.status_code == 409
    assert "only after POST /api/predict" in r.json()["detail"]
    assert "labels" not in r.json()

    assert client.post("/api/predict", json={"record_id": rid}).status_code == 200
    r = client.get(f"/api/records/{rid}/truth")
    assert r.status_code == 200
    body = r.json()
    y = dict(zip(data.labels.ecg_ids.tolist(), data.labels.y))[int(rid)]
    expected = [c for c, v in zip(data.labels.classes, y) if v == 1]
    assert body["id"] == rid and body["labels"] == expected
    assert [(c["class"], c["present"]) for c in body["classes"]] == [(c, c in expected) for c in data.labels.classes]


# ------------------------------------------------------------------------------------------------ uploads

def test_upload_of_a_test_record_matches_its_record_id_prediction(cfg, client, cache, test_ids, upload_root):
    rid = test_ids[UPLOAD_ROW]
    r = post_files(client, files_of(raw_record(cfg, cache, rid)))
    assert r.status_code == 200, r.text
    upload = r.json()
    assert_no_upload_left(upload_root)
    assert upload["input"] == {"source": "upload", "record_id": None, "sampling_rate_hz": 100.0, "n_samples": 1000,
                               "resampled": False, "checksums_verified": True}
    np.testing.assert_array_equal(np.array(upload["signal"]["data"], dtype=np.float32), cached_signal(cache, rid))

    by_id = client.post("/api/predict", json={"record_id": rid}).json()
    for key in ("predictions", "attention", "attention_class", "summary"):
        assert upload[key] == by_id[key]  # the same signal through the same model: identical numbers


def test_an_upload_does_not_unlock_the_truth(cfg, client, cache, test_ids):
    rid = test_ids[UPLOAD_ONLY_ROW]
    assert post_files(client, files_of(raw_record(cfg, cache, rid))).status_code == 200
    assert client.get(f"/api/records/{rid}/truth").status_code == 409  # an upload carries no record id


def test_upper_case_extensions_are_accepted(cfg, client, cache, test_ids, tmp_path, upload_root):
    signal, fields = wfdb.rdsamp(str(raw_record(cfg, cache, test_ids[UPLOAD_ROW])))
    files = write_record(tmp_path, "upper", signal, 100, fields["sig_name"])
    files["hea"] = ("upper.HEA", files["hea"][1])
    r = post_files(client, files)
    assert r.status_code == 200, r.text
    assert_no_upload_left(upload_root)


def test_upload_with_leads_reordered_and_renamed_lowercase(cfg, client, cache, test_ids, tmp_path, upload_root):
    rid = test_ids[UPLOAD_ROW]
    signal, fields = wfdb.rdsamp(str(raw_record(cfg, cache, rid)))
    order = list(range(12))[::-1]
    files = write_record(tmp_path, "reordered", signal[:, order], 100, [fields["sig_name"][i].lower() for i in order])
    r = post_files(client, files)
    assert r.status_code == 200, r.text
    assert_no_upload_left(upload_root)
    np.testing.assert_array_equal(np.array(r.json()["signal"]["data"], dtype=np.float32), cached_signal(cache, rid))
    assert r.json()["predictions"] == client.post("/api/predict", json={"record_id": rid}).json()["predictions"]


def test_upload_at_500_hz_is_resampled(cfg, client, cache, test_ids, tmp_path, upload_root):
    """A real test record written at 500 Hz is resampled back to 100 Hz before the model reads it."""
    rid = test_ids[UPLOAD_ROW]
    signal, fields = wfdb.rdsamp(str(raw_record(cfg, cache, rid)))
    files = write_record(tmp_path, "rec500", resample_poly(signal, 5, 1, axis=0, padtype="line"), 500,
                         fields["sig_name"])
    r = post_files(client, files)
    assert r.status_code == 200, r.text
    assert_no_upload_left(upload_root)
    body = r.json()
    assert body["input"] == {"source": "upload", "record_id": None, "sampling_rate_hz": 500.0, "n_samples": 5000,
                             "resampled": True, "checksums_verified": True}
    back = np.array(body["signal"]["data"])
    assert back.shape == (12, 1000)
    # Bounds from the training-record measurement in config/default.yaml (api.upload.resample_padtype)
    error = np.abs(back - cached_signal(cache, rid))
    assert error[:, 10:-10].max() < 0.05 and error.max() < 0.2
    by_id = by_class(client.post("/api/predict", json={"record_id": rid}).json())
    for c, p in by_class(body).items():
        assert abs(p["probability"] - by_id[c]["probability"]) < 0.01


def test_resampling_round_trip_on_training_records(cfg, cache):
    """The measurement behind api.upload.resample_padtype, repeated on 300 of the 3,000 training records."""
    up_cfg = cfg["api"]["upload"]
    rows = np.sort(np.random.default_rng(42).choice(cache.rows("train"), 300, replace=False))
    edge, rest = [], []
    for row in rows:
        x = cache.signals[row].astype(np.float64).T
        up = np.round(resample_poly(x, 5, 1, axis=0, padtype="line") * 1000) / 1000  # stored as WFDB stores it
        d = np.abs(resample(up, 500, 100, up_cfg) - x)
        edge.append(max(d[:10].max(), d[-10:].max()))
        rest.append(d[10:-10].max())
    assert max(edge) < 0.2 and max(rest) < 0.05
    assert resample(x, 100, 100, up_cfg) is x  # nothing is done at the model's own rate


def _bad_uploads(signal: np.ndarray, names: list[str], other: dict, folder: Path):
    """name -> (files, the 400 message must contain this)."""
    out = {
        "eleven leads": (write_record(folder, "eleven", signal[:, :11], 100, names[:11]), "has 11 signals"),
        "five seconds": (write_record(folder, "short", signal[:500], 100, names), "is 5 s long"),
        "unknown lead": (write_record(folder, "v7", signal, 100, names[:11] + ["V7"]), "missing V6"),
        "microvolts": (write_record(folder, "uv", signal, 100, names, units="uV"), "must be in mV"),
        "fractional rate": (write_record(folder, "frac", np.vstack([signal, signal[:5]]), 100.5, names),
                            "whole number of Hz"),
    }
    gap = signal.copy()
    gap[100, 3] = np.nan
    out["missing samples"] = (write_record(folder, "gap", gap, 100, names), "missing samples")
    flat = signal.copy()
    flat[:, 9] = flat[0, 9]
    out["flat lead"] = (write_record(folder, "flat", flat, 100, names), f"Lead(s) {names[9]} are flat lines")
    out["all zero"] = (write_record(folder, "zero", np.zeros_like(signal), 100, names), "are flat lines")
    # Samples 2,000 times too large, as when a header gives gain 1 for values stored at gain 1000
    out["wrong gain"] = (write_record(folder, "gain", signal * 2000, 100, names, adc_gain=1.0),
                         "no training record goes beyond")
    good = write_record(folder, "good", signal, 100, names)
    out["renamed dat"] = ({"hea": good["hea"], "dat": ("other.dat", good["dat"][1])}, "names its samples file")
    out["truncated dat"] = ({"hea": good["hea"], "dat": (good["dat"][0], good["dat"][1][:12000])}, "truncated")
    out["another record's dat"] = ({"hea": good["hea"], "dat": (good["dat"][0], other["dat"][1])}, "checksums")
    out["not a header"] = ({"hea": ("x.hea", b"this is not a WFDB header\n"), "dat": good["dat"]},
                           "not a readable WFDB header")
    out["path in file name"] = ({"hea": ("../good.hea", good["hea"][1]), "dat": good["dat"]}, "plain name")
    out["wrong extension"] = ({"hea": ("good.txt", good["hea"][1]), "dat": good["dat"]}, "ending in .hea")
    return out


BAD_UPLOADS = ["eleven leads", "five seconds", "unknown lead", "microvolts", "fractional rate", "missing samples",
               "flat lead", "all zero", "wrong gain", "renamed dat", "truncated dat", "another record's dat", "not a header", "path in file name",
               "wrong extension"]


@pytest.mark.parametrize("case", BAD_UPLOADS)
def test_bad_upload_is_400_with_a_clear_message(cfg, client, cache, test_ids, tmp_path, upload_root, case):
    signal, fields = wfdb.rdsamp(str(raw_record(cfg, cache, test_ids[UPLOAD_ROW])))
    other = files_of(raw_record(cfg, cache, test_ids[UPLOAD_ROW + 1]))
    files, message = _bad_uploads(signal, list(fields["sig_name"]), other, tmp_path)[case]
    r = post_files(client, files)
    assert r.status_code == 400, r.text
    assert message in r.json()["detail"]
    assert_no_upload_left(upload_root)


def test_upload_form_must_hold_exactly_the_two_files(cfg, client, cache, test_ids, upload_root):
    files = files_of(raw_record(cfg, cache, test_ids[UPLOAD_ROW]))
    r = client.post("/api/predict", files={"hea": files["hea"]})
    assert r.status_code == 400 and "'hea' and 'dat'" in r.json()["detail"]
    r = client.post("/api/predict", files={k: v for k, v in files.items()}, data={"note": "hello"})
    assert r.status_code == 400 and "'hea'" in r.json()["detail"]
    r = client.post("/api/predict", files={**files, "extra": files["dat"]})
    assert r.status_code == 400
    assert_no_upload_left(upload_root)


def test_upload_size_is_capped(cfg, client):
    limit = cfg["api"]["upload"]["max_request_bytes"]
    headers = {"content-type": "multipart/form-data; boundary=x"}
    r = client.post("/api/predict", content=b"x" * (limit + 1), headers=headers)
    assert r.status_code == 413 and f"{limit:,}" in r.json()["detail"]

    def chunks():  # no Content-Length: the cap is enforced while the body streams in
        for _ in range(limit // 65536 + 2):
            yield b"x" * 65536

    r = client.post("/api/predict", content=chunks(), headers=headers)
    assert r.status_code == 413


@pytest.mark.parametrize("body, content_type, status", [
    (b'{"record_id": 9}', "application/json", 400),          # the id must be a string
    (b'{"record_id": "9", "extra": 1}', "application/json", 400),
    (b'{"record": "9"}', "application/json", 400),
    (b"not json", "application/json", 400),
    (b"[" * 200_000, "application/json", 400),               # nested deeper than any recursion limit
    (b"record_id=9", "application/x-www-form-urlencoded", 415),
    (b"9", "text/plain", 415),
], ids=["id not a string", "extra field", "no record_id", "not json", "deeply nested", "form encoded", "plain text"])
def test_malformed_predict_requests(client, body, content_type, status):
    r = client.post("/api/predict", content=body, headers={"content-type": content_type})
    assert r.status_code == status and r.json()["detail"]


# ------------------------------------------------------------------------------------------------ metrics, CORS

def test_metrics_are_served_verbatim(cfg, client):
    r = client.get("/api/metrics")
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    path = project_path(cfg["paths"]["reports_dir"]) / cfg["final_evaluation"]["metrics_file"]
    assert r.content == path.read_bytes()


def test_test_set_figures_are_served_verbatim(cfg, client):
    metrics = json.loads((project_path(cfg["paths"]["reports_dir"]) / cfg["final_evaluation"]["metrics_file"])
                         .read_text(encoding="utf-8"))
    assert sorted(metrics["figures"]) == sorted(cfg["final_evaluation"]["figures"])
    for name, path in metrics["figures"].items():
        r = client.get(f"/api/figures/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.content == project_path(path).read_bytes()


@pytest.mark.parametrize("name", ["training_curves", "roc_curves_val", "roc_curves_test.png", "metrics"])
def test_only_the_figures_metrics_json_lists_are_served(client, name):
    r = client.get(f"/api/figures/{name}")
    assert r.status_code == 404 and "reports/metrics.json lists" in r.json()["detail"]


def test_without_metrics_json_the_demo_runs_and_metrics_answer_404(cfg, tmp_path):
    reports = project_path(cfg["paths"]["reports_dir"])
    for name in ("release.json", "calibration.json", "thresholds.json"):  # no metrics.json: evaluation not run
        shutil.copyfile(reports / name, tmp_path / name)
    no_eval = copy.deepcopy(cfg)
    no_eval["paths"]["reports_dir"] = str(tmp_path)
    with TestClient(create_app(no_eval)) as c:
        assert c.get("/api/health").json()["model_loaded"] is True
        for path in ("/api/metrics", "/api/figures/roc"):
            r = c.get(path)
            assert r.status_code == 404
            assert "python -m src.evaluation.evaluate_test" in r.json()["detail"]


def _metrics_with_figure(cfg, tmp_path, figure: str) -> dict:
    reports = project_path(cfg["paths"]["reports_dir"])
    for name in ("release.json", "calibration.json", "thresholds.json"):
        shutil.copyfile(reports / name, tmp_path / name)
    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    metrics["figures"]["roc"] = figure
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    bad = copy.deepcopy(cfg)
    bad["paths"]["reports_dir"] = str(tmp_path)
    return bad


@pytest.mark.parametrize("figure, error, match", [
    ("reports/figures/no_such_figure_test.png", FileNotFoundError, "does not exist"),
    ("reports/metrics.json", ValueError, "not a .png in"),
    ("reports/figures/../release.json", ValueError, "not a .png in"),
], ids=["missing", "not a png", "outside the figures folder"])
def test_startup_refuses_a_figure_metrics_json_cannot_back(cfg, tmp_path, figure, error, match):
    with pytest.raises(error, match=match):
        _start(_metrics_with_figure(cfg, tmp_path, figure))


def test_cors_allows_only_the_local_frontend(client):
    preflight = {"access-control-request-method": "POST", "access-control-request-headers": "content-type"}
    r = client.options("/api/predict", headers={"origin": FRONTEND, **preflight})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == FRONTEND
    r = client.options("/api/predict", headers={"origin": "http://evil.example", **preflight})
    assert r.status_code == 400 and "access-control-allow-origin" not in r.headers
    r = client.get("/api/health", headers={"origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    assert client.get("/api/health", headers={"origin": FRONTEND}).headers["access-control-allow-origin"] == FRONTEND


# ------------------------------------------------------------------------------------------------ startup checks

def _start(cfg) -> None:
    with TestClient(create_app(cfg)):
        pass


@pytest.mark.parametrize("artifact", ["release_checkpoint", "calibration", "thresholds"])
def test_startup_refuses_artifacts_that_differ_from_the_manifest(cfg, manifest, tmp_path, artifact):
    edited = copy.deepcopy(manifest)
    if artifact == "release_checkpoint":
        edited["release_checkpoint"]["sha256"] = "0" * 64  # release.pt is no longer the pinned file
    else:
        original = project_path(manifest[artifact]["file"])
        copy_path = tmp_path / original.name
        number = {"calibration": str(manifest["calibration"]["per_class"]["MI"]["a"]),
                  "thresholds": str(manifest["thresholds"]["high_sensitivity"]["MI"])}[artifact]
        text = original.read_text(encoding="utf-8")
        assert number in text
        copy_path.write_text(text.replace(number, "0.15"), encoding="utf-8")  # someone edited a frozen number
        edited[artifact]["file"] = str(copy_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / cfg["release"]["manifest_file"]).write_text(json.dumps(edited), encoding="utf-8")
    bad = copy.deepcopy(cfg)
    bad["paths"]["reports_dir"] = str(reports)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        _start(bad)


def test_startup_refuses_metrics_of_another_model(cfg, tmp_path):
    reports = project_path(cfg["paths"]["reports_dir"])
    for name in ("release.json", "calibration.json", "thresholds.json", "metrics.json"):
        shutil.copyfile(reports / name, tmp_path / name)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    metrics["model"]["sha256"] = "0" * 64
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    bad = copy.deepcopy(cfg)
    bad["paths"]["reports_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match="metrics.json was not computed with the frozen release"):
        _start(bad)


def test_checkpoint_bytes_are_hashed_before_loading(manifest, tmp_path):
    fake = tmp_path / "release.pt"
    fake.write_bytes(b"not the release model")
    with pytest.raises(ValueError, match="Refusing to serve"):
        load_verified_checkpoint(fake, manifest["release_checkpoint"]["sha256"], torch.device("cpu"))


def test_startup_refuses_an_upload_cap_above_the_in_memory_spool(cfg):
    bad = copy.deepcopy(cfg)
    bad["api"]["upload"]["max_request_bytes"] = MultiPartParser.spool_max_size + 1
    with pytest.raises(ValueError, match="spool"):
        _start(bad)


def test_cuda_device_without_cuda_fails_loudly(monkeypatch):
    monkeypatch.setattr(service_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="api.device is 'cuda'"):
        api_device("cuda")
