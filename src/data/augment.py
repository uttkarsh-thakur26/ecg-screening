"""Training-only augmentation of one normalised record of shape (leads, samples).

Three transforms, each behind its own flag in config augmentation (spec section 2):

- amplitude scaling: one factor from [0.8, 1.2] for the whole record. All 12 leads are scaled
  together because they are projections of the same cardiac activity (lead III = II - I, and the
  augmented leads are derived from I and II); scaling them separately would create a record no
  heart could produce.
- Gaussian noise: independent per sample, small sigma, in normalised units.
- time shift: the record moves up to (10 s - window_s) earlier or later, the same shift for every
  lead because the leads are recorded simultaneously. Vacated samples become 0.

Deliberately absent: time reversal and sign inversion. A reversed ECG runs T wave -> QRS -> P wave,
and an inverted one turns ST elevation into ST depression and upright T waves into inverted ones.
Both change the clinical meaning, so the label would no longer be true of the augmented record.

All randomness comes from torch, so one mechanism controls both shuffling and augmentation: in
DataLoader worker processes torch's RNG is seeded from the loader's generator, and without workers
the Dataset passes its own generator seeded from training.seed (src/data/dataset.py).

Every transform draws its random numbers whether or not it is enabled, always in the same order.
Turning one transform off therefore leaves the draws of the others unchanged, so an ablation run
with the same seed differs from the full run only in the transform it removes.
"""

from __future__ import annotations

from typing import Any

import torch


def time_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Move x (..., samples) `shift` samples later (negative: earlier). Vacated samples are 0; order is kept."""
    n = x.shape[-1]
    if abs(shift) >= n:
        raise ValueError(f"Shift {shift} must be smaller than the record length {n}")
    if shift == 0:
        return x
    out = torch.zeros_like(x)
    if shift > 0:
        out[..., shift:] = x[..., :-shift]
    else:
        out[..., :shift] = x[..., -shift:]
    return out


class Augmenter:
    """Applies the enabled augmentations in a fixed order: amplitude scale, noise, then time shift.

    The shift comes last so the zero-filled gap stays exactly 0 instead of picking up noise.
    """

    def __init__(self, aug_cfg: dict[str, Any], sampling_rate_hz: int, n_samples: int) -> None:
        shift, amp, noise = aug_cfg["time_shift"], aug_cfg["amplitude_scale"], aug_cfg["gaussian_noise"]

        self.max_shift: int | None = None
        if shift["enabled"]:
            window = shift["window_s"] * sampling_rate_hz
            if int(window) != window or not 0 < window < n_samples:
                raise ValueError(f"augmentation.time_shift.window_s = {shift['window_s']} s must be a whole number "
                                 f"of samples, more than 0 and less than the {n_samples / sampling_rate_hz:g} s record")
            self.max_shift = n_samples - int(window)

        self.scale_range: tuple[float, float] | None = None
        if amp["enabled"]:
            low, high = (float(v) for v in amp["range"])
            if not 0 < low <= high:
                raise ValueError(f"augmentation.amplitude_scale.range = {amp['range']} must satisfy 0 < low <= high; "
                                 f"a factor <= 0 would invert or erase the signal")
            self.scale_range = (low, high)

        self.noise_sigma: float | None = None
        if noise["enabled"]:
            sigma = float(noise["sigma"])
            if not sigma > 0:
                raise ValueError(f"augmentation.gaussian_noise.sigma = {noise['sigma']} must be > 0 when enabled")
            self.noise_sigma = sigma

    @property
    def enabled(self) -> bool:
        return any(v is not None for v in (self.max_shift, self.scale_range, self.noise_sigma))

    def describe(self) -> str:
        parts = []
        if self.scale_range is not None:
            parts.append(f"amplitude x{self.scale_range[0]:g}-{self.scale_range[1]:g}")
        if self.noise_sigma is not None:
            parts.append(f"noise sigma {self.noise_sigma:g}")
        if self.max_shift is not None:
            parts.append(f"time shift up to +/-{self.max_shift} samples")
        return ", ".join(parts) if parts else "none enabled"

    def __call__(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        # Drawn in this order every time, used only by the enabled transforms (see the module docstring).
        u_scale = torch.rand((), generator=generator).item()
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
        u_shift = torch.rand((), generator=generator).item()
        if self.scale_range is not None:
            low, high = self.scale_range
            x = x * (low + (high - low) * u_scale)
        if self.noise_sigma is not None:
            x = x + self.noise_sigma * noise
        if self.max_shift is not None:
            # u_shift is in [0, 1), so this is a whole number from -max_shift to +max_shift, each equally likely.
            shift = int(u_shift * (2 * self.max_shift + 1)) - self.max_shift
            x = time_shift(x, shift)
        return x
