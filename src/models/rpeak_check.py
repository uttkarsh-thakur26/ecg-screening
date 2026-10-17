"""Compare beat-detector variants on training records: how the settings in config baselines.rpeaks were chosen.

    python -m src.models.rpeak_check

PTB-XL has no beat annotations, so detectors are compared by consistency instead. The heart beats once
whatever the lead, so a detector run separately on lead II and on the agreement leads (I and V5) should
count the same beats. For each variant this prints the share of records whose lead II count is within
agreement_tolerance_beats of every agreement lead, and the 1st, 50th and 99th percentile of lead II beats
per record. A 10 s record at a normal heart rate has about 10-12 beats, so a low 1st percentile means
records where most beats were missed, and a high 99th percentile means P or T waves counted as beats.

Variants:
    configured           config baselines.rpeaks as it is
    no T-wave rule       twave_slope_ratio = 0, so no candidate is rejected as a T wave
    no search-back       searchback_rr_ratio = infinity, so long gaps are never searched again
    threshold from max   the signal level is the single highest candidate instead of the median of the top third
    wfdb XQRS            wfdb.processing.xqrs_detect, the WFDB library's default QRS detector
    wfdb GQRS            wfdb.processing.gqrs_detect, a port of PhysioNet's gqrs

Only training records are used (a random sample seeded with training.seed); no labels are read.
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from wfdb import processing

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.data.cache import open_cache
from src.models.baselines import check_rpeak_config, detect_rpeaks

VARIANTS = ("configured", "no T-wave rule", "no search-back", "threshold from max", "wfdb XQRS", "wfdb GQRS")


def count_beats(variant: str, lead: np.ndarray, fs: float, rp: dict[str, Any]) -> int:
    """Number of beats one variant finds in one lead."""
    if variant == "configured":
        return len(detect_rpeaks(lead, fs, rp))
    if variant == "no T-wave rule":
        return len(detect_rpeaks(lead, fs, {**rp, "twave_slope_ratio": 0.0}))
    if variant == "no search-back":
        return len(detect_rpeaks(lead, fs, {**rp, "searchback_rr_ratio": math.inf}))
    if variant == "threshold from max":
        # There are fewer candidates than samples, so ceil(fraction * candidates) = 1: the highest candidate alone.
        return len(detect_rpeaks(lead, fs, {**rp, "signal_level_top_fraction": 1 / len(lead)}))
    if variant == "wfdb XQRS":
        return len(processing.xqrs_detect(lead, fs=fs, verbose=False))
    if variant == "wfdb GQRS":
        return len(processing.gqrs_detect(lead, fs=fs))
    raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")


def _count_chunk(task: tuple[str, np.ndarray, float, list[int], dict[str, Any]]) -> np.ndarray:
    """(records, variants, leads) beat counts for some cache rows."""
    signals_path, rows, fs, lead_idx, rp = task
    signals = np.load(signals_path, mmap_mode="r")
    out = np.empty((len(rows), len(VARIANTS), len(lead_idx)), dtype=np.int64)
    for k, row in enumerate(rows):
        record = np.asarray(signals[row], dtype=np.float64)
        for v, variant in enumerate(VARIANTS):
            out[k, v] = [count_beats(variant, record[i], fs, rp) for i in lead_idx]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare beat-detector variants on training records by cross-lead agreement.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    b = cfg["baselines"]
    rp, fs = b["rpeaks"], cfg["dataset"]["sampling_rate_hz"]
    check_rpeak_config(rp, fs)

    cache = open_cache(cfg, build_if_missing=False)
    leads = list(cache.meta["leads"])
    lead_names = [rp["lead"], *rp["agreement_leads"]]
    lead_idx = [leads.index(name) for name in lead_names]
    n = b["rpeak_check"]["records"]
    rows = np.sort(np.random.default_rng(cfg["training"]["seed"]).choice(cache.rows("train"), n, replace=False))

    chunks = np.array_split(rows, math.ceil(len(rows) / b["chunk_size"]))
    tasks = [(str(cache.signals_path), chunk, fs, lead_idx, rp) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=b["workers"]) as pool:
        counts = np.concatenate(list(tqdm(pool.map(_count_chunk, tasks), total=len(tasks), desc="detect", unit="chunk")))

    tol = rp["agreement_tolerance_beats"]
    print(f"\n{len(rows):,} random training records. Agreement: lead {rp['lead']} beat count within {tol} of "
          f"{' and '.join(rp['agreement_leads'])}.\n")
    print(f"{'Variant':<20}{'Agreement':>11}   Lead {rp['lead']} beats per record: 1st / 50th / 99th percentile")
    for v, variant in enumerate(VARIANTS):
        c = counts[:, v]
        agree = (np.abs(c[:, :1] - c[:, 1:]) <= tol).all(axis=1).mean()
        p1, p50, p99 = np.percentile(c[:, 0], [1, 50, 99])
        print(f"{variant:<20}{agree:>10.1%}    {p1:>4g} / {p50:>4g} / {p99:>4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
