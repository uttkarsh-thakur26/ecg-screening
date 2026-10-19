"""The frozen release model, loaded once for the API, and the held-out test records the demo serves.

The demo serves the held-out test split (fold 10): it lists the records, returns their signals, predicts them one at a
time and reveals a record's labels only after that record has been predicted. It never computes a metric: the test
metrics are reports/metrics.json, written once by the one-time evaluation and served verbatim. This is the only API
module that opens fold 10, and it is on the fold-10 allowlist in tests/test_dataset.py (MAY_UNLOCK_TEST) for that reason.

Loading (ReleaseService.__init__), in this order. Any failure raises, and the server does not start:
1. The frozen inputs, read and checked by the one-time evaluation's own loader (evaluate_test.Inputs): release.pt,
   calibration.json and thresholds.json still have the SHA-256 values pinned in reports/release.json; the calibration
   and thresholds belong to the release run; the thresholds cover exactly the classes they should.
2. api.operating_point's thresholds cover the abnormal classes only and equal the values in the manifest.
3. reports/metrics.json names the same model, calibration and thresholds.
4. release.pt is read into memory once, those bytes are hashed again and compared with the manifest, and the model is
   built from the same bytes, so the weights in memory are exactly the bytes that were checked.
5. The signal cache is the one the model was trained and evaluated on (the checkpoint stores its SHA-256), and the test
   records are normalised with the training statistics by ECGDataset, as in the evaluation's loader. They must be the
   records reports/metrics.json was computed on. The largest absolute amplitude of any training record is read from the
   same cache: uploads above it are refused (src/api/upload.py).
6. One warm-up pass on the first test record (its result is discarded), so the first request's timing does not
   include CUDA start-up.

Inference mirrors evaluate_test.split_predictions for a single record: eval mode, the run's deterministic cuDNN
settings, the logits calibrated with the frozen per-class calibrators. The evaluation ran batches of 64; for one record
cuDNN can pick another kernel, and with TF32 the logits then differ in about the third decimal (measured on fold 9:
up to 1.9e-3; explain.logit_tolerance), which moves a calibrated probability by well under 1e-3.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import project_path
from src.data.cache import Cache, open_cache
from src.data.dataset import ECGDataset
from src.data.norm_stats import ecg_id_digest
from src.evaluation.calibration import calibrated_probabilities
from src.evaluation.evaluate_test import Inputs
from src.explain.gradcam1d import GradCAM1d
from src.models.resnet1d import ResNet1d, build_model
from src.training.train import seed_everything

RECORDS_PER_READ = 1024  # training records read at a time when scanning their amplitudes (about 50 MB)


class UnknownRecordError(KeyError):
    """The ecg_id is not one of the held-out test records the demo serves."""


class TruthLockedError(RuntimeError):
    """The labels of a record were asked for before the record was predicted."""


@dataclass(frozen=True)
class Inference:
    """One record through the release model. Arrays are NumPy, in class order (classes) or sample order (attention)."""

    logits: np.ndarray            # (classes,) float32, the model's raw outputs
    probabilities: np.ndarray     # (classes,) float64, sigmoid(a * logit + b) with the frozen calibrator per class
    attention: np.ndarray         # (samples,) float64 in [0, 1]: Grad-CAM of attention_class
    attention_class: str          # the abnormal class with the highest calibrated probability
    attention_raw_peak: float     # the map's maximum before normalisation; 0 means no position raised the class score
    inference_ms: float           # forward pass and calibration
    attention_ms: float           # Grad-CAM's forward and backward pass


def api_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("api.device is 'cuda' but torch cannot see a CUDA GPU. Check that nvidia-smi lists the GPU "
                           "and that the driver supports this PyTorch build's CUDA version. To serve on the CPU on "
                           "purpose, set api.device: cpu in config/default.yaml.")
    return torch.device(name)


def load_verified_checkpoint(path: Path, sha256: str, device: torch.device) -> tuple[ResNet1d, dict[str, Any]]:
    """The model built from the bytes of `path`, after checking that those bytes have the pinned SHA-256."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Release checkpoint not found: {path}")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise ValueError(f"{path} has SHA-256 {actual}, reports/release.json pins {sha256}. Refusing to serve a model "
                         "that is not the frozen release.")
    ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])  # strict: every tensor must match
    return model.to(device).eval(), ckpt


def training_amplitude_limit(cache: Cache) -> float:
    """The largest absolute value, in mV, of any sample of any training record."""
    rows = cache.rows("train")
    return float(max(np.abs(np.asarray(cache.signals[rows[i:i + RECORDS_PER_READ]])).max()
                     for i in range(0, len(rows), RECORDS_PER_READ)))


def check_metrics_report(metrics: dict[str, Any], manifest: dict[str, Any]) -> None:
    """reports/metrics.json must have been computed with the release model, calibration and thresholds."""
    expected = {
        ("model", "sha256"): manifest["release_checkpoint"]["sha256"],
        ("model", "version"): manifest["model_version"],
        ("model", "run_id"): manifest["run"]["run_id"],
        ("calibration", "sha256"): manifest["calibration"]["sha256"],
        ("thresholds", "sha256"): manifest["thresholds"]["sha256"],
        ("split",): "test",
    }
    wrong = {}
    for keys, value in expected.items():
        got: Any = metrics
        for k in keys:
            got = got.get(k) if isinstance(got, dict) else None
        if got != value:
            wrong[".".join(keys)] = {"metrics.json": got, "release": value}
    if wrong:
        raise ValueError(f"reports/metrics.json was not computed with the frozen release: {wrong}")


class ReleaseService:
    """The release model, its frozen calibrators and thresholds, and the held-out test records. Thread-safe."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        api = cfg["api"]
        inputs = Inputs(cfg)  # 1. SHA-256 of release.pt, calibration.json and thresholds.json against the manifest
        manifest = inputs.manifest
        run = inputs.run
        seed_everything(run["seed"], run["config"]["training"]["deterministic"])  # the evaluation's cuDNN settings

        # 2. The operating point whose thresholds flag the abnormal classes.
        point = api["operating_point"]
        if point not in inputs.thresholds:
            raise ValueError(f"api.operating_point {point!r} is not one of the frozen operating points "
                             f"{sorted(inputs.thresholds)}")
        thresholds = inputs.thresholds[point]
        if sorted(thresholds) != sorted(inputs.abnormal):
            raise ValueError(f"api.operating_point {point!r} has thresholds for {sorted(thresholds)}; the API flags "
                             f"the abnormal classes {inputs.abnormal} only")
        if thresholds != manifest["thresholds"][point]:
            raise ValueError(f"thresholds.json's {point} thresholds differ from reports/release.json's")

        # 3. The test metrics the API serves verbatim.
        metrics_path = project_path(cfg["paths"]["reports_dir"]) / cfg["final_evaluation"]["metrics_file"]
        if not metrics_path.is_file():
            raise FileNotFoundError(f"No {metrics_path}: the one-time test evaluation has not been run")
        self.metrics_bytes = metrics_path.read_bytes()
        metrics = json.loads(self.metrics_bytes)
        check_metrics_report(metrics, manifest)

        # 4. The weights, built from the bytes that were hashed.
        self.device = api_device(api["device"])
        model, ckpt = load_verified_checkpoint(project_path(manifest["release_checkpoint"]["path"]),
                                               manifest["release_checkpoint"]["sha256"], self.device)
        if ckpt["run_id"] != run["run_id"] or ckpt["classes"] != inputs.classes:
            raise ValueError(f"release.pt holds run {ckpt['run_id']} with classes {ckpt['classes']}")

        # 5. The held-out test records, normalised as the evaluation normalised them.
        run_cfg = ckpt["config"]
        cache = open_cache(run_cfg, build_if_missing=False)
        if cache.meta["signals_sha256"] != ckpt["signals_sha256"]:
            raise ValueError("The signal cache on disk is not the one the release model was trained and evaluated on")
        ds = ECGDataset(run_cfg, "test", allow_test=True)  # the demo records; no metric is computed from them
        if list(ds.classes) != inputs.classes:
            raise ValueError(f"Test classes {list(ds.classes)} differ from the model's {inputs.classes}")
        if len(ds) != metrics["records"] or ecg_id_digest(ds.ecg_ids) != metrics["ecg_ids_sha256"]:
            raise ValueError("The test records differ from the ones reports/metrics.json was computed on")

        self.model = model
        self.gradcam = GradCAM1d(model)
        self.dataset = ds
        self.manifest = manifest
        self.run_cfg = run_cfg  # the release run's effective config: its dataset and preprocessing settings
        self.classes: list[str] = list(inputs.classes)
        self.abnormal: list[str] = list(inputs.abnormal)
        self.descriptions: dict[str, str] = dict(cfg["labels"]["descriptions"])
        self.leads: list[str] = list(ds.leads)
        self.operating_point = point
        self.operating_point_rule: str = inputs.thresholds_report["operating_points"][point]["rule"]
        self.thresholds: dict[str, float] = dict(thresholds)
        self.calibrators = inputs.calibrators
        self.model_version: str = manifest["model_version"]
        self.model_tag: str = manifest["tag"]
        self.sampling_rate_hz: int = run_cfg["dataset"]["sampling_rate_hz"]
        self.duration_s: int = run_cfg["dataset"]["record_duration_s"]
        self.n_samples = self.sampling_rate_hz * self.duration_s
        self.logit_tolerance: float = cfg["explain"]["logit_tolerance"]
        self.max_abs_mv = training_amplitude_limit(cache)  # uploads beyond any training record are refused
        self.record_ids: list[int] = [int(e) for e in ds.ecg_ids]
        self._row_of = {e: i for i, e in enumerate(self.record_ids)}
        self._signals = np.load(ds.signals_path, mmap_mode="r")  # the cached records in mV, before normalisation
        self._model_lock = threading.Lock()  # one record at a time: Grad-CAM hooks the shared model
        self._predicted: set[int] = set()
        self._predicted_lock = threading.Lock()

        # 6. Warm-up on a real record; nothing is recorded.
        self._infer(ds[0][0])

    # ------------------------------------------------------------------------------------------- records

    def has_record(self, ecg_id: int) -> bool:
        return ecg_id in self._row_of

    def _row(self, ecg_id: int) -> int:
        if ecg_id not in self._row_of:
            raise UnknownRecordError(ecg_id)
        return self._row_of[ecg_id]

    def raw_signal(self, ecg_id: int) -> np.ndarray:
        """(leads, samples) float32 in mV: the cached record before normalisation."""
        return np.array(self._signals[self.dataset.rows[self._row(ecg_id)]], dtype=np.float32)

    def truth(self, ecg_id: int) -> list[str]:
        """The record's labels (superclasses present), once the record has been predicted; TruthLockedError before."""
        i = self._row(ecg_id)
        with self._predicted_lock:
            predicted = ecg_id in self._predicted
        if not predicted:
            raise TruthLockedError(ecg_id)
        return [c for c, v in zip(self.classes, self.dataset.targets[i]) if v == 1]

    # ------------------------------------------------------------------------------------------- inference

    def predict_record(self, ecg_id: int) -> Inference:
        """A held-out test record through the model; afterwards its labels may be revealed."""
        x, _ = self.dataset[self._row(ecg_id)]
        result = self._infer(x)
        with self._predicted_lock:
            self._predicted.add(ecg_id)
        return result

    def predict_signal(self, signal: np.ndarray) -> Inference:
        """An uploaded record, (leads, samples) float32 mV in the training lead order, already preprocessed."""
        expected = (len(self.leads), self.n_samples)
        if signal.shape != expected or signal.dtype != np.float32:
            raise ValueError(f"Expected a float32 {expected} signal, got {signal.dtype} {signal.shape}")
        return self._infer(self.dataset.normalise(signal))

    def _infer(self, x: torch.Tensor) -> Inference:
        batch = x.unsqueeze(0).to(self.device)
        abnormal_idx = [self.classes.index(c) for c in self.abnormal]
        with self._model_lock:
            start = time.perf_counter()
            with torch.no_grad():
                out = self.model(batch)
            logits = out.float().cpu().numpy()  # copying to the host waits for the GPU, so the timing is complete
            probs = calibrated_probabilities(logits, self.calibrators, self.classes)
            inference_ms = (time.perf_counter() - start) * 1e3

            j = abnormal_idx[int(np.argmax(probs[0, abnormal_idx]))]
            start = time.perf_counter()
            cam = self.gradcam(batch, j)
            attention_ms = (time.perf_counter() - start) * 1e3
        if abs(float(cam.logits[0]) - float(logits[0, j])) > self.logit_tolerance:
            raise RuntimeError(f"Grad-CAM's forward pass gives logit {float(cam.logits[0]):.6f} for {self.classes[j]}, "
                               f"the prediction {float(logits[0, j]):.6f} (explain.logit_tolerance "
                               f"{self.logit_tolerance:.0e})")
        return Inference(logits=logits[0], probabilities=probs[0], attention=cam.maps[0],
                         attention_class=self.classes[j], attention_raw_peak=float(cam.raw_peak[0]),
                         inference_ms=inference_ms, attention_ms=attention_ms)
