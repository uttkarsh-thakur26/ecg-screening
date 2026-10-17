"""Augmentation unit tests. Random tensors are fine here: these test the transforms, not the data."""

from __future__ import annotations

import copy

import pytest
import torch

from src.data.augment import Augmenter, time_shift

LEADS, SAMPLES, FS = 12, 1000, 100


def _only(cfg: dict, name: str) -> Augmenter:
    aug = copy.deepcopy(cfg["augmentation"])
    for key in aug:
        aug[key]["enabled"] = key == name
    return Augmenter(aug, FS, SAMPLES)


def test_time_shift_moves_the_signal_and_zero_fills_the_gap():
    x = torch.arange(1, 11, dtype=torch.float32).repeat(2, 1)  # strictly increasing in time
    later = time_shift(x, 3)
    assert torch.equal(later[:, :3], torch.zeros(2, 3))
    assert torch.equal(later[:, 3:], x[:, :7])  # same order: nothing is reversed
    earlier = time_shift(x, -3)
    assert torch.equal(earlier[:, :7], x[:, 3:])
    assert torch.equal(earlier[:, 7:], torch.zeros(2, 3))
    assert torch.equal(time_shift(x, 0), x)
    with pytest.raises(ValueError):
        time_shift(x, 10)


def test_time_shift_keeps_at_least_the_window_and_moves_all_leads_together(cfg):
    aug = _only(cfg, "time_shift")
    max_shift = SAMPLES - cfg["augmentation"]["time_shift"]["window_s"] * FS
    assert aug.max_shift == max_shift == 200
    x = torch.zeros(LEADS, SAMPLES)
    x[:, SAMPLES // 2] = 1.0  # an impulse in every lead
    g = torch.Generator().manual_seed(0)
    shifts = []
    for _ in range(500):
        y = aug(x, g)
        where = torch.nonzero(y)
        assert len(where) == LEADS and len(set(where[:, 1].tolist())) == 1  # one shift for all 12 leads
        shifts.append(int(where[0, 1]) - SAMPLES // 2)
    assert min(shifts) >= -max_shift and max(shifts) <= max_shift
    assert min(shifts) < -0.9 * max_shift and max(shifts) > 0.9 * max_shift  # the whole range is used


def test_amplitude_scale_is_one_positive_factor_for_all_leads(cfg):
    aug = _only(cfg, "amplitude_scale")
    low, high = cfg["augmentation"]["amplitude_scale"]["range"]
    g = torch.Generator().manual_seed(0)
    x = torch.rand(LEADS, SAMPLES, generator=g) + 1.0  # values in [1, 2), so the ratio is well defined
    factors = []
    for _ in range(500):
        ratio = aug(x, g) / x
        assert torch.allclose(ratio, ratio[0, 0].expand_as(ratio), rtol=1e-5)  # the same factor everywhere
        factors.append(float(ratio[0, 0]))
    assert low - 1e-6 <= min(factors) and max(factors) <= high + 1e-6
    assert min(factors) < low + 0.05 and max(factors) > high - 0.05
    assert min(factors) > 0  # never a sign inversion


def test_gaussian_noise_has_the_configured_sigma(cfg):
    aug = _only(cfg, "gaussian_noise")
    sigma = cfg["augmentation"]["gaussian_noise"]["sigma"]
    x = torch.zeros(LEADS, SAMPLES)
    noise = aug(x, torch.Generator().manual_seed(0))
    assert float(noise.std()) == pytest.approx(sigma, rel=0.05)
    assert abs(float(noise.mean())) < sigma * 0.05


def test_all_flags_off_means_no_augmentation(cfg):
    aug = copy.deepcopy(cfg["augmentation"])
    for key in aug:
        aug[key]["enabled"] = False
    a = Augmenter(aug, FS, SAMPLES)
    assert not a.enabled
    x = torch.randn(LEADS, SAMPLES, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a(x), x)


def test_same_seed_same_augmentation(cfg):
    a = Augmenter(cfg["augmentation"], FS, SAMPLES)
    x = torch.randn(LEADS, SAMPLES, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a(x, torch.Generator().manual_seed(7)), a(x, torch.Generator().manual_seed(7)))
    assert not torch.equal(a(x, torch.Generator().manual_seed(7)), a(x, torch.Generator().manual_seed(8)))


@pytest.mark.parametrize(
    "key, change",
    [
        ("amplitude_scale", {"range": [-1.2, -0.8]}),  # would invert the ECG
        ("amplitude_scale", {"range": [0.0, 1.2]}),
        ("amplitude_scale", {"range": [1.2, 0.8]}),
        ("time_shift", {"window_s": 10}),  # no room to shift
        ("time_shift", {"window_s": 0}),
        ("gaussian_noise", {"sigma": 0.0}),
    ],
)
def test_invalid_settings_fail_loudly(cfg, key, change):
    aug = copy.deepcopy(cfg["augmentation"])
    aug[key].update(enabled=True, **change)
    with pytest.raises(ValueError):
        Augmenter(aug, FS, SAMPLES)
