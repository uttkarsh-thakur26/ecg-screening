"""Deterministic preprocessing, applied once per record before caching and before computing norm_stats.

The only step is the optional band-pass filter (spec section 2: 0.5-40 Hz, a config flag, off by
default). It removes baseline wander below 0.5 Hz and muscle or mains noise above 40 Hz. It is a
Butterworth filter run forwards and then backwards (scipy's sosfiltfilt), which makes it zero-phase:
P waves, QRS complexes and ST segments stay exactly where they were in time. A single-pass filter
would delay and distort them, and ST-segment shape is what the STTC and MI classes depend on.

The same function must run on training, validation, test and uploaded records, so this module has no
split-specific behaviour. Normalisation and augmentation are not here: they live in dataset.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

RAW_TAG = "raw"


def preprocessing_tag(cfg: dict[str, Any]) -> str:
    """Name of the preprocessing variant, used as its cache folder: "raw" or e.g. "bandpass_0.5-40hz_order2"."""
    bp = cfg["preprocessing"]["bandpass"]
    if not bp["enabled"]:
        return RAW_TAG
    return f"bandpass_{bp['low_hz']:g}-{bp['high_hz']:g}hz_order{bp['order']}"


def bandpass_settings(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The band-pass parameters in effect, or None when it is off (so unused settings never matter)."""
    bp = cfg["preprocessing"]["bandpass"]
    return {k: bp[k] for k in ("low_hz", "high_hz", "order")} if bp["enabled"] else None


def bandpass_sos(fs: float, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    """Second-order sections of a Butterworth band-pass. Raises on cut-offs the sampling rate cannot support."""
    nyquist = fs / 2
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(f"Band-pass needs 0 < low ({low_hz}) < high ({high_hz}) < Nyquist ({nyquist}) Hz")
    if int(order) != order or order < 1:
        raise ValueError(f"Band-pass order must be a positive integer, got {order}")
    return butter(int(order), [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")


def to_cache_layout(signal: np.ndarray) -> np.ndarray:
    """(samples, leads) float64 -> contiguous float32 (leads, samples), exactly as stored in the cache."""
    return np.ascontiguousarray(signal.T, dtype=np.float32)


def preprocess(signal: np.ndarray, fs: float, preprocessing_cfg: dict[str, Any]) -> np.ndarray:
    """Preprocess one record of shape (samples, leads) in mV, as load_record returns it. Returns float64.

    With the band-pass off this is the identity (as float64), so the "raw" cache is the WFDB signal.
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected a (samples, leads) array, got shape {signal.shape}")
    x = np.asarray(signal, dtype=np.float64)
    bp = preprocessing_cfg["bandpass"]
    if not bp["enabled"]:
        return x
    sos = bandpass_sos(fs, bp["low_hz"], bp["high_hz"], bp["order"])
    return sosfiltfilt(sos, x, axis=0)
