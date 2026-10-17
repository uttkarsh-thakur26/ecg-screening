"""Per-class probability calibrators: their form, how to apply them, and reading reports/calibration.json.

src/training/calibrate.py fits them on the validation split. Everything that turns the model's logits into
the probabilities shown or thresholded (evaluate_val now, the API later) applies them through this module.

Both calibrators map one class's logit z (the model output before the sigmoid) to a probability:
- temperature: p = sigmoid(z / T), with T > 0. One parameter. It stretches (T < 1) or squeezes (T > 1) every
  logit towards 0, which makes the model more or less confident, but z = 0 always stays at p = 0.5.
- platt (Platt scaling: temperature plus a bias): p = sigmoid(a z + b), with a > 0. The bias b also moves the
  logit at which p = 0.5, which temperature scaling cannot do.
Both are stored as sigmoid(a z + b): temperature scaling is the case a = 1/T, b = 0.

Because a > 0, both maps are strictly increasing: two records keep their order within a class, so AUROC,
AUPRC and which records any threshold flags are unchanged. Only the numbers attached to the records move,
which is why decision thresholds have to be chosen on the calibrated probabilities.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

METHODS = ("temperature", "platt")


@dataclass(frozen=True)
class Calibrator:
    """p = sigmoid(a * logit + b) for one class. Temperature scaling has a = 1/T and b = 0."""

    method: str
    a: float
    b: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Calibration method must be one of {METHODS}, got {self.method!r}")
        if not (math.isfinite(self.a) and math.isfinite(self.b)):
            raise ValueError(f"{self.method} calibrator has non-finite parameters a={self.a}, b={self.b}")
        if self.a <= 0:
            raise ValueError(f"{self.method} calibrator has slope a={self.a}; it must be > 0, or it would reverse "
                             "the order of the records' scores")
        if self.method == "temperature" and self.b != 0:
            raise ValueError(f"A temperature calibrator has no bias, got b={self.b}")

    @classmethod
    def temperature(cls, t: float) -> Calibrator:
        if not (math.isfinite(t) and t > 0):
            raise ValueError(f"Temperature must be finite and > 0, got {t}")
        return cls("temperature", 1.0 / t, 0.0)

    @property
    def t(self) -> float:
        """The temperature T = 1/a (meaningful for temperature scaling only)."""
        return 1.0 / self.a

    def logits(self, z: np.ndarray) -> np.ndarray:
        """Calibrated logits a * z + b, in float64."""
        return self.a * np.asarray(z, dtype=np.float64) + self.b

    def probabilities(self, z: np.ndarray) -> np.ndarray:
        """Calibrated probabilities sigmoid(a * z + b), in float64 (float32 would round large logits to 1.0)."""
        return expit(self.logits(z))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method, "a": self.a, "b": self.b}
        if self.method == "temperature":
            out["temperature"] = self.t
        return out

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Calibrator:
        """Rebuild from to_json's output. A temperature calibrator is rebuilt from T and must reproduce its a exactly."""
        if d.get("method") == "temperature":
            c = cls.temperature(float(d["temperature"]))
            if c.a != d["a"] or d["b"] != 0:
                raise ValueError(f"Temperature calibrator T={d['temperature']} gives a={c.a}, b=0; the file says "
                                 f"a={d['a']}, b={d['b']}")
            return c
        return cls(str(d.get("method")), float(d["a"]), float(d["b"]))


def calibrated_probabilities(logits: np.ndarray, calibrators: Mapping[str, Calibrator],
                             classes: Sequence[str]) -> np.ndarray:
    """(records, classes) probabilities: column j is calibrators[classes[j]] applied to logits[:, j]."""
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != len(classes):
        raise ValueError(f"logits must be (records, {len(classes)}), got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values")
    if set(calibrators) != set(classes):
        raise ValueError(f"Calibrators are for {sorted(calibrators)}, the classes are {sorted(classes)}")
    return np.stack([calibrators[c].probabilities(logits[:, j]) for j, c in enumerate(classes)], axis=1)


def file_sha256(path: Path) -> str:
    """SHA-256 of a text file with line endings normalised to LF.

    Git stores the reports with LF and checks them out with CRLF on Windows, so hashing the raw bytes would give
    a different hash for the same report on Linux, macOS and Windows.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def choose(eces: Mapping[str, float]) -> str:
    """The selection rule: the calibrator with the lower ECE; temperature on an exact tie (fewer parameters)."""
    return "platt" if eces["platt"] < eces["temperature"] else "temperature"


def load_calibration(path: Path, run: Mapping[str, Any], classes: Sequence[str],
                     ecg_ids_sha256: str) -> dict[str, Any]:
    """Read reports/calibration.json and check it was fitted for this run's checkpoint on these validation records.

    Returns {"path": path, "sha256": file_sha256 of the file, "calibrators": {class: Calibrator},
    "report": the parsed JSON}. Raises if the file is missing, belongs to another model, split or class list, or
    if a class's stored calibrator is not the fitted candidate that the selection rule picks.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No calibration file {path}. Fit it first: python -m src.training.calibrate "
                                f"--run {run['run_id']}")
    report = json.loads(path.read_text(encoding="utf-8"))
    model = report.get("model", {})
    if model.get("run_id") != run["run_id"] or model.get("checkpoint") != run["checkpoint"]:
        raise ValueError(f"{path} was fitted for run {model.get('run_id')} ({model.get('checkpoint')}), not "
                         f"{run['run_id']} ({run['checkpoint']}). Refit: python -m src.training.calibrate "
                         f"--run {run['run_id']}")
    if model.get("best_epoch") != run["best"]["epoch"]:
        raise ValueError(f"{path} was fitted on epoch {model.get('best_epoch')}'s checkpoint; the run's best epoch is "
                         f"{run['best']['epoch']}")
    if report.get("split") != "val" or report.get("ecg_ids_sha256") != ecg_ids_sha256:
        raise ValueError(f"{path} was not fitted on these validation records")
    if report.get("classes") != list(classes):
        raise ValueError(f"{path} has classes {report.get('classes')}, expected {list(classes)}")
    calibrators = {}
    for c in classes:
        r = report["per_class"][c]
        eces = {k: r["candidates"][k]["ece"] for k in METHODS}
        if r["chosen"] != choose(eces):
            raise ValueError(f"{path}: {c} chose {r['chosen']}, but the selection rule picks {choose(eces)} "
                             f"(ECE {eces})")
        fitted = {k: v for k, v in r["candidates"][r["chosen"]].items() if k in ("method", "a", "b", "temperature")}
        if r["calibrator"] != fitted:
            raise ValueError(f"{path}: {c}'s calibrator {r['calibrator']} differs from the fitted {r['chosen']} "
                             f"parameters {fitted}; the file was edited after fitting")
        calibrators[c] = Calibrator.from_json(r["calibrator"])
    return {"path": path, "sha256": file_sha256(path), "calibrators": calibrators, "report": report}
