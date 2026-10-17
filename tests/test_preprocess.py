"""Band-pass filter: off means untouched; on, it removes wander and high-frequency noise without moving anything in time."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.config import project_path
from src.data.preprocess import RAW_TAG, bandpass_sos, preprocess, preprocessing_tag
from src.data.records import load_record

FS = 100
T = np.arange(1000) / FS
MIDDLE = slice(200, 800)  # away from the edges, where forward-backward filtering pads the signal


def _enabled(cfg: dict) -> dict:
    pre = copy.deepcopy(cfg["preprocessing"])
    pre["bandpass"]["enabled"] = True
    return pre


def _gain_and_lag(pre: dict, freq_hz: float) -> tuple[float, int]:
    x = np.sin(2 * np.pi * freq_hz * T)[:, None]
    y = preprocess(x, FS, pre)
    gain = float(np.sqrt((y[MIDDLE] ** 2).mean() / (x[MIDDLE] ** 2).mean()))
    n = MIDDLE.stop - MIDDLE.start
    lag = int(np.argmax(np.correlate(y[MIDDLE, 0], x[MIDDLE, 0], "full")) - (n - 1))
    return gain, lag


def test_default_is_off_and_leaves_the_record_unchanged(cfg, data):
    assert cfg["preprocessing"]["bandpass"]["enabled"] is False
    assert preprocessing_tag(cfg) == RAW_TAG
    signal, _ = load_record(project_path(cfg["paths"]["raw_dir"]), data.labelled.loc[data.ids("train")[0], "filename_lr"], cfg["dataset"])
    np.testing.assert_array_equal(preprocess(signal, FS, cfg["preprocessing"]), signal)


def test_bandpass_passes_ecg_band_and_removes_wander_and_hf_noise(cfg):
    pre = _enabled(cfg)
    low, high = pre["bandpass"]["low_hz"], pre["bandpass"]["high_hz"]
    # Butterworth is -3 dB at each cut-off; filtering forwards and backwards squares that to a gain of 0.5.
    assert _gain_and_lag(pre, low)[0] == pytest.approx(0.5, abs=0.01)
    assert _gain_and_lag(pre, high)[0] == pytest.approx(0.5, abs=0.01)
    for f in (5, 10, 20):  # QRS energy sits around 5-20 Hz
        assert _gain_and_lag(pre, f)[0] > 0.99
    assert _gain_and_lag(pre, 0.1)[0] < 0.01  # baseline wander (breathing, electrode drift)
    assert _gain_and_lag(pre, 45)[0] < 0.1  # muscle noise


def test_bandpass_is_zero_phase(cfg, data):
    pre = _enabled(cfg)
    for f in (1, 5, 10, 20, 30):
        assert _gain_and_lag(pre, f)[1] == 0
    # On a real record, lead II is not shifted in time either: its cross-correlation with the original peaks at lag 0.
    signal, _ = load_record(project_path(cfg["paths"]["raw_dir"]), data.labelled.loc[data.ids("train")[0], "filename_lr"], cfg["dataset"])
    lead_ii = signal[:, 1] - signal[:, 1].mean()
    filtered = preprocess(signal, FS, pre)[:, 1]
    assert np.argmax(np.correlate(filtered, lead_ii, "full")) - (len(lead_ii) - 1) == 0
    assert abs(filtered.mean()) < abs(lead_ii.std()) * 0.05  # the DC offset is gone


def test_tag_names_the_settings(cfg):
    on = copy.deepcopy(cfg)
    on["preprocessing"]["bandpass"].update(enabled=True, low_hz=0.5, high_hz=40.0, order=2)
    assert preprocessing_tag(on) == "bandpass_0.5-40hz_order2"


@pytest.mark.parametrize("low, high, order", [(0.0, 40.0, 2), (0.5, 50.0, 2), (40.0, 0.5, 2), (0.5, 40.0, 0), (0.5, 40.0, 1.5)])
def test_impossible_settings_fail_loudly(low, high, order):
    with pytest.raises(ValueError):
        bandpass_sos(FS, low, high, order)
