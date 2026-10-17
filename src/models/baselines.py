"""Prior and feature-based gradient-boosting baselines (PROJECT_SPEC.md section 3), scored on validation only.

    python -m src.models.baselines

Both are fit on the training split (folds 1-8) and scored on the validation split (fold 9); the
results go to reports/baselines_val.json. Nothing is tuned: every setting is fixed in config
baselines. This module asks the cache for the "train" and "val" rows only and never opens the test split.

1. Prior baseline. Every record gets each class's frequency in the training split as its probability.
   All records get the same score, so it cannot rank one above another: its AUROC is exactly 0.5
   (every positive-negative pair is a tie, which counts one half) and its AUPRC is the validation
   prevalence. It is the floor any model has to beat.

2. Feature baseline. 79 features per record, then one gradient-boosted tree classifier per class:
   - rhythm, from beats detected on lead II: the number of beats, the heart rate, and the mean,
     standard deviation, minimum, maximum and RMSSD of the RR intervals (7 features);
   - the amplitude distribution of each of the 12 leads: mean, standard deviation, minimum,
     maximum, skewness and kurtosis (72 features).
   Features come from the cached signals in mV, before the z-score. Trees split on thresholds and
   bin each feature by its quantiles, so a per-lead z-score would not change them.

The beat detector follows Pan & Tompkins (1985) and is described in detect_rpeaks. It was chosen on
training records only, and its quality is measured without annotations: the beat count on lead II
should agree with an independent detection on other leads (config baselines.rpeaks.agreement_leads).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from scipy.signal import find_peaks, sosfiltfilt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.cache import Cache, open_cache
from src.data.preprocess import bandpass_sos
from src.data.splits import assert_no_patient_leakage, assign_splits
from src.evaluation.metrics import ranking_metrics
from src.provenance import git_state

SECONDS_PER_MINUTE = 60.0
RR_FEATURES = ("n_beats", "heart_rate_bpm", "rr_mean_s", "rr_std_s", "rr_min_s", "rr_max_s", "rr_rmssd_s")
LEAD_STATS = ("mean", "std", "min", "max", "skew", "kurtosis")
FIT_SPLIT, EVAL_SPLIT = "train", "val"


def _samples(seconds: float, fs: float) -> int:
    n = round(seconds * fs)
    if n < 1:
        raise ValueError(f"{seconds} s is less than one sample at {fs} Hz")
    return n


def check_rpeak_config(rp: dict[str, Any], fs: float) -> None:
    """Raise on detector settings that would make the output ill-defined."""
    if 2 * _samples(rp["peak_search_s"], fs) >= _samples(rp["refractory_s"], fs):
        raise ValueError("peak_search_s must be under half of refractory_s, or two beats could land on the same sample")
    if not 0 < rp["signal_level_top_fraction"] <= 1:
        raise ValueError(f"signal_level_top_fraction must be in (0, 1], got {rp['signal_level_top_fraction']}")
    if rp["min_rr_intervals"] < max(2, rp["rr_std_ddof"] + 1):
        raise ValueError(f"min_rr_intervals must be at least 2 (RMSSD needs two RR intervals) and above rr_std_ddof, "
                         f"got {rp['min_rr_intervals']}")


def detect_rpeaks(lead: np.ndarray, fs: float, rp: dict[str, Any]) -> np.ndarray:
    """Sample indices of the beats in one lead, shape (samples,) in mV. Pan & Tompkins (1985), run offline.

    1. Band-pass 5-15 Hz (zero-phase), differentiate, square, and average over about one QRS width.
       This "QRS energy" is large on the steep QRS and small on the slower P and T waves.
    2. Candidates: local maxima of the energy at least one refractory period apart.
    3. Threshold: threshold_fraction x the median height of the highest third of the candidates.
    4. In time order, accept candidates above the threshold, except T waves: a candidate soon after a
       beat with much less slope than it (Pan & Tompkins' rule).
    5. Search-back: a gap between beats much longer than the median RR is searched again at a lower
       threshold, and its highest candidate becomes a beat (Pan & Tompkins).
    6. Each beat is placed at the largest band-passed deflection near its energy peak, normally the R
       wave; in a lead where the QRS is mostly negative it is the S or Q wave. RR intervals do not
       depend on which, as long as it is the same wave in every beat.
    A lead with no QRS energy (flat) has no beats.
    """
    x = np.asarray(lead, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected one lead of shape (samples,), got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Lead contains NaN or infinite samples")
    low, high = rp["band_hz"]
    filtered = sosfiltfilt(bandpass_sos(fs, low, high, rp["filter_order"]), x)
    slope = np.gradient(filtered)
    window = _samples(rp["integration_window_s"], fs)
    energy = np.convolve(slope**2, np.ones(window) / window, mode="same")

    refractory = _samples(rp["refractory_s"], fs)
    candidates, _ = find_peaks(energy, distance=refractory)
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    highest = np.sort(energy[candidates])[::-1][: math.ceil(rp["signal_level_top_fraction"] * len(candidates))]
    threshold = rp["threshold_fraction"] * float(np.median(highest))
    if threshold <= 0:
        return np.empty(0, dtype=np.int64)

    half = _samples(rp["peak_search_s"], fs)

    def max_slope(i: int) -> float:
        return float(np.abs(slope[max(0, i - half) : i + half + 1]).max())

    twave = rp["twave_window_s"] * fs
    beats: list[int] = []
    for c in candidates:
        if energy[c] <= threshold:
            continue
        if beats and c - beats[-1] < twave and max_slope(c) < rp["twave_slope_ratio"] * max_slope(beats[-1]):
            continue
        beats.append(int(c))

    if len(beats) > rp["min_rr_intervals"]:  # a median RR needs at least min_rr_intervals intervals
        median_rr = float(np.median(np.diff(beats)))
        missed = []
        for a, b in zip(beats[:-1], beats[1:]):
            if b - a > rp["searchback_rr_ratio"] * median_rr:
                inside = candidates[(candidates - a >= max(refractory, twave)) & (b - candidates >= refractory)]
                inside = inside[energy[inside] > rp["searchback_threshold_ratio"] * threshold]
                if len(inside):
                    missed.append(int(inside[np.argmax(energy[inside])]))
        beats = sorted(beats + missed)

    positions = []
    for c in beats:
        lo, hi = max(0, c - half), min(len(x), c + half + 1)
        positions.append(lo + int(np.argmax(np.abs(filtered[lo:hi]))))
    return np.asarray(positions, dtype=np.int64)


def rr_features(beats: np.ndarray, fs: float, min_rr_intervals: int, rr_std_ddof: int) -> np.ndarray:
    """RR_FEATURES for one record. With fewer than min_rr_intervals RR intervals, all but n_beats are NaN."""
    out = np.full(len(RR_FEATURES), np.nan)
    out[0] = len(beats)
    rr = np.diff(np.asarray(beats, dtype=np.float64)) / fs
    if len(rr) >= min_rr_intervals:
        if (rr <= 0).any():
            raise ValueError(f"Beats must be strictly increasing, got {beats}")
        mean = rr.mean()
        out[1:] = [SECONDS_PER_MINUTE / mean, mean, rr.std(ddof=rr_std_ddof), rr.min(), rr.max(),
                   math.sqrt(np.mean(np.diff(rr) ** 2))]
    return out


def lead_statistics(record: np.ndarray) -> np.ndarray:
    """LEAD_STATS of every lead of one (leads, samples) record, lead by lead: I_mean, I_std, ..., V6_kurtosis.

    Standard deviation, skewness and kurtosis are population moments (scipy's defaults; kurtosis is
    Fisher's, 0 for a normal distribution). Skewness and kurtosis are NaN for a flat lead, where they are undefined.
    """
    x = np.asarray(record, dtype=np.float64)
    std = x.std(axis=1)
    flat = std == 0
    sk = np.full(len(x), np.nan)
    ku = np.full(len(x), np.nan)
    sk[~flat] = skew(x[~flat], axis=1)
    ku[~flat] = kurtosis(x[~flat], axis=1)
    return np.stack([x.mean(axis=1), std, x.min(axis=1), x.max(axis=1), sk, ku], axis=1).ravel()


def feature_names(leads: list[str]) -> list[str]:
    return list(RR_FEATURES) + [f"{lead}_{stat}" for lead in leads for stat in LEAD_STATS]


def _features_chunk(task: tuple[str, np.ndarray, float, list[str], dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Features of some cache rows, and the beat count on the rhythm lead and on each agreement lead."""
    signals_path, rows, fs, leads, rp = task
    signals = np.load(signals_path, mmap_mode="r")
    rhythm = leads.index(rp["lead"])
    others = [leads.index(name) for name in rp["agreement_leads"]]
    features = np.empty((len(rows), len(feature_names(leads))))
    counts = np.empty((len(rows), 1 + len(others)), dtype=np.int64)
    for k, row in enumerate(rows):
        record = np.asarray(signals[row], dtype=np.float64)
        beats = detect_rpeaks(record[rhythm], fs, rp)
        features[k] = np.concatenate([rr_features(beats, fs, rp["min_rr_intervals"], rp["rr_std_ddof"]), lead_statistics(record)])
        counts[k] = [len(beats)] + [len(detect_rpeaks(record[j], fs, rp)) for j in others]
    return features, counts


@dataclass(frozen=True)
class Features:
    X: np.ndarray  # (records, features), NaN where a feature is undefined
    names: list[str]
    beat_counts: np.ndarray  # (records, 1 + agreement leads): rhythm lead first
    seconds: float


def extract_features(cache: Cache, rows: np.ndarray, cfg: dict[str, Any], desc: str) -> Features:
    """Features of the given cache rows, in the same order, computed in parallel worker processes."""
    b = cfg["baselines"]
    rp, fs = b["rpeaks"], cfg["dataset"]["sampling_rate_hz"]
    leads = list(cache.meta["leads"])
    unknown = [name for name in [rp["lead"], *rp["agreement_leads"]] if name not in leads]
    if unknown:
        raise ValueError(f"Leads {unknown} in config baselines.rpeaks are not among the cached leads {leads}")
    if rp["lead"] in rp["agreement_leads"]:
        raise ValueError("The rhythm lead cannot also be an agreement lead")
    check_rpeak_config(rp, fs)

    chunks = np.array_split(rows, math.ceil(len(rows) / b["chunk_size"]))
    tasks = [(str(cache.signals_path), chunk, fs, leads, rp) for chunk in chunks]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=b["workers"]) as pool:
        parts = list(tqdm(pool.map(_features_chunk, tasks), total=len(tasks), desc=desc, unit="chunk"))  # in task order
    X = np.concatenate([p[0] for p in parts])
    counts = np.concatenate([p[1] for p in parts])
    if X.shape != (len(rows), len(feature_names(leads))):
        raise RuntimeError(f"Feature matrix has shape {X.shape}, expected ({len(rows)}, {len(feature_names(leads))})")
    return Features(X, feature_names(leads), counts, time.perf_counter() - start)


class PriorBaseline:
    """Scores every record with each class's frequency in the labels it was fit on."""

    def fit(self, y: np.ndarray) -> PriorBaseline:
        y = np.asarray(y)
        if y.ndim != 2 or len(y) == 0 or not np.isin(y, (0, 1)).all():
            raise ValueError(f"Expected a non-empty (records, classes) 0/1 label matrix, got shape {y.shape}")
        self.prior_ = y.mean(axis=0)
        return self

    def predict_proba(self, n_records: int) -> np.ndarray:
        """(n_records, classes): the same row of training frequencies for every record."""
        return np.tile(self.prior_, (n_records, 1))


class FeatureBaseline:
    """One HistGradientBoostingClassifier per class on the record features; NaN features are handled natively."""

    def __init__(self, params: dict[str, Any], seed: int) -> None:
        if params.get("early_stopping") is not False:
            raise ValueError("gradient_boosting.early_stopping must be false: sklearn's early stopping holds out a random "
                             "share of training records rather than patients, and tuning belongs to the validation split")
        self.params = dict(params)
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureBaseline:
        y = np.asarray(y)
        if y.ndim != 2 or len(y) != len(X):
            raise ValueError(f"Labels {y.shape} do not match features {np.shape(X)}")
        self.models_ = []
        for j in range(y.shape[1]):
            model = HistGradientBoostingClassifier(**self.params, random_state=self.seed).fit(X, y[:, j])
            if list(model.classes_) != [0, 1]:
                raise ValueError(f"Class column {j} has labels {model.classes_.tolist()}, expected both 0 and 1")
            self.models_.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(records, classes): each class's own classifier's probability of the positive label."""
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])


def _beat_summary(features: Features, rp: dict[str, Any]) -> dict[str, Any]:
    counts = features.beat_counts
    rhythm = counts[:, 0]
    rr_missing = int(np.isnan(features.X[:, features.names.index("rr_mean_s")]).sum())
    lead_stats = features.X[:, len(RR_FEATURES):]
    tolerance = rp["agreement_tolerance_beats"]
    agree = {name: float((np.abs(rhythm - counts[:, 1 + i]) <= tolerance).mean()) for i, name in enumerate(rp["agreement_leads"])}
    agree_all = float((np.abs(rhythm[:, None] - counts[:, 1:]) <= tolerance).all(axis=1).mean())
    return {
        "seconds": round(features.seconds, 1),
        "beats_per_record": {"min": int(rhythm.min()), "p1": float(np.percentile(rhythm, 1)), "median": float(np.median(rhythm)),
                             "p99": float(np.percentile(rhythm, 99)), "max": int(rhythm.max())},
        "records_without_rr_features": rr_missing,
        "undefined_lead_statistics": int(np.isnan(lead_stats).sum()),
        "beat_count_agreement": {"tolerance_beats": tolerance, "per_lead": agree, "all_leads": agree_all},
    }


def _split_summary(cache: Cache, rows: np.ndarray, folds: list[int], classes: list[str]) -> dict[str, Any]:
    y = cache.labels[rows]
    return {
        "folds": [int(f) for f in folds],
        "records": len(rows),
        "patients": int(cache.index["patient_id"].iloc[rows].nunique()),
        "positives": {c: int(n) for c, n in zip(classes, y.sum(axis=0))},
        "prevalence": {c: float(p) for c, p in zip(classes, y.mean(axis=0))},
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit both baselines on the training split, score them on the validation split, and return the report."""
    classes = list(cfg["labels"]["superclasses"])
    cache = open_cache(cfg, build_if_missing=False)
    if list(cache.meta["classes"]) != classes:
        raise ValueError(f"Cache classes {cache.meta['classes']} differ from config {classes}")
    index = cache.index.set_index("ecg_id")
    assert_no_patient_leakage(index, assign_splits(index, cfg["splits"]))
    fit_rows, eval_rows = cache.rows(FIT_SPLIT), cache.rows(EVAL_SPLIT)
    y_fit, y_eval = cache.labels[fit_rows], cache.labels[eval_rows]
    seed = cfg["training"]["seed"]
    b, rp = cfg["baselines"], cfg["baselines"]["rpeaks"]

    fit_features = extract_features(cache, fit_rows, cfg, f"features {FIT_SPLIT}")
    eval_features = extract_features(cache, eval_rows, cfg, f"features {EVAL_SPLIT}")

    prior = PriorBaseline().fit(y_fit)
    start = time.perf_counter()
    gbdt = FeatureBaseline(b["gradient_boosting"], seed).fit(fit_features.X, y_fit)
    fit_seconds = time.perf_counter() - start

    return {
        "description": f"Baselines fit on the {FIT_SPLIT} split and scored on the {EVAL_SPLIT} split only "
                       "(PROJECT_SPEC.md section 3). Nothing was tuned. The test split was not opened.",
        "generated_by": "python -m src.models.baselines",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "seed": seed,
        "dataset": {k: cfg["dataset"][k] for k in ("name", "version", "sampling_rate_hz")}
                   | {"preprocessing": cache.meta["preprocessing"]},
        "classes": classes,
        "splits": {
            FIT_SPLIT: _split_summary(cache, fit_rows, cfg["splits"][f"{FIT_SPLIT}_folds"], classes),
            EVAL_SPLIT: _split_summary(cache, eval_rows, cfg["splits"][f"{EVAL_SPLIT}_folds"], classes),
        },
        "baselines": {
            "prior": {
                "description": f"Every record scored with each class's {FIT_SPLIT}-split frequency",
                "scores": {c: float(p) for c, p in zip(classes, prior.prior_)},
                "metrics": ranking_metrics(y_eval, prior.predict_proba(len(eval_rows)), classes),
            },
            "feature_gbdt": {
                "description": f"{len(fit_features.names)} features per record (RR intervals from lead {rp['lead']} beats, "
                               "plus per-lead amplitude statistics), one gradient-boosted tree classifier per class",
                "model": "sklearn.ensemble.HistGradientBoostingClassifier",
                "sklearn_version": sklearn.__version__,
                "params": {**b["gradient_boosting"], "random_state": seed},
                "fit_seconds": round(fit_seconds, 1),
                "features": fit_features.names,
                "rpeak_detector": rp,
                "feature_extraction": {FIT_SPLIT: _beat_summary(fit_features, rp), EVAL_SPLIT: _beat_summary(eval_features, rp)},
                "metrics": ranking_metrics(y_eval, gbdt.predict_proba(eval_features.X), classes),
            },
        },
    }


def format_table(report: dict[str, Any]) -> str:
    """Per-class and macro AUROC and AUPRC of both baselines on the evaluated split, with each class's prevalence."""
    ev = report["splits"][EVAL_SPLIT]
    prior, gbdt = (report["baselines"][k]["metrics"] for k in ("prior", "feature_gbdt"))
    lines = [
        f"{EVAL_SPLIT} split (fold {', '.join(map(str, ev['folds']))}): {ev['records']:,} records, {ev['patients']:,} patients. "
        f"Fit on the {FIT_SPLIT} split: {report['splits'][FIT_SPLIT]['records']:,} records.",
        "",
        f"{'':<30}{'Prior baseline':>16}    {'Feature baseline (GBDT)':>23}",
        f"{'Class':<6}{'Positives':>11}{'Prevalence':>13}{'AUROC':>8}{'AUPRC':>8}    {'AUROC':>11}{'AUPRC':>12}",
    ]
    for c in report["classes"]:
        p, g = prior["per_class"][c], gbdt["per_class"][c]
        lines.append(f"{c:<6}{p['n_positive']:>11,}{p['prevalence']:>13.3f}{p['auroc']:>8.3f}{p['auprc']:>8.3f}"
                     f"    {g['auroc']:>11.3f}{g['auprc']:>12.3f}")
    pm, gm = prior["macro"], gbdt["macro"]
    lines.append(f"{'Macro':<6}{'':>11}{pm['prevalence']:>13.3f}{pm['auroc']:>8.3f}{pm['auprc']:>8.3f}"
                 f"    {gm['auroc']:>11.3f}{gm['auprc']:>12.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the prior and feature baselines and score them on the validation split.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    report = run(cfg)
    out = project_path(cfg["paths"]["reports_dir"]) / cfg["baselines"]["report_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rp = cfg["baselines"]["rpeaks"]
    print()
    for split, summary in report["baselines"]["feature_gbdt"]["feature_extraction"].items():
        bpr, agree = summary["beats_per_record"], summary["beat_count_agreement"]
        per_lead = ", ".join(f"{lead} {share:.1%}" for lead, share in agree["per_lead"].items())
        print(f"{split}: features in {summary['seconds']} s; beats per record on lead {rp['lead']} median {bpr['median']:g} "
              f"(1st-99th percentile {bpr['p1']:g}-{bpr['p99']:g}); {summary['records_without_rr_features']} records "
              f"without RR features; {summary['undefined_lead_statistics']} undefined lead statistics")
        print(f"  lead {rp['lead']} beat count within {agree['tolerance_beats']} of {per_lead}; all at once {agree['all_leads']:.1%}")
    print()
    print(format_table(report))
    print()
    git = report["git"]
    if git["dirty"]:
        print(f"WARNING: tracked files differ from commit {git['commit'][:7]}; the report records dirty=true.")
    print(f"Saved {out} (commit {git['commit'][:7]}{', dirty' if git['dirty'] else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
