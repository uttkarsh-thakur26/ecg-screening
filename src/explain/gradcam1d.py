"""Grad-CAM adapted to 1D (PROJECT_SPEC.md section 5; Selvaraju et al., 2017).

For one target class c and a batch of records:
1. A forward hook on the last residual stage (model.stages[-1]) captures its output A, (records, K, L):
   K = 512 channels at L = 32 time positions for a 10 s record at 100 Hz.
2. The gradient of class c's raw logit y_c with respect to A, dy_c/dA.
3. Channel weights alpha_k = mean over the L positions of dy_c/dA_k.
4. CAM(t) = ReLU(sum_k alpha_k A_k(t)), shape (L,).
5. Linear interpolation from the L positions to every input sample (1000).
6. Division by the map's maximum, so the peak is 1 and the values lie in [0, 1].

The target is the raw logit, before the sigmoid and before calibration. A calibrator turns the logit into
a * y_c + b with a > 0 (temperature scaling: a = 1/T, b = 0; Platt scaling: fitted a and b). Its gradient is
a * dy_c/dA, so every alpha_k is multiplied by a, and since ReLU(a s) = a ReLU(s) for a > 0, the map is
multiplied by a. Interpolation is linear, and step 6 divides that factor out again: the normalised map is
identical. b is a constant and has no gradient. With a < 0 the ReLU would keep the opposite time regions,
which is one reason the calibrators require a > 0. The sigmoid would add another positive per-record factor,
p (1 - p), which also cancels, but it underflows for confident records, so the logit is the safer target.

This network ends in global average pooling and one linear layer, y_c = sum_k w_ck mean_t A_k(t) + b_c. So
dy_c/dA_k(t) = w_ck / L at every t, alpha_k = w_ck / L, and the map before the ReLU, sum_k alpha_k A_k(t),
sums over the L positions to exactly y_c - b_c: it splits the logit into one additive share per position.
Grad-CAM here equals CAM (Zhou et al., 2016). tests/test_gradcam1d.py checks this identity.

Why step 6 divides by the maximum instead of stretching min..max to 0..1: after the ReLU, 0 means "this time
position adds nothing to the class score". Dividing by the peak keeps that meaning and keeps a nearly flat map
looking flat; min-max stretching would turn small differences into full contrast. A map with no positive
value anywhere (every position lowers the class score) stays all zero, and raw_peak = 0 says so.

Time alignment. Every convolution and the max pool use padding kernel // 2 with odd kernels, so feature
position j is centred on input sample stride * j, with stride 32 here (2 x 2 x 1 x 2 x 2 x 2): positions
0, 32, ..., 992. feature_geometry derives this from the layers instead of assuming it. Interpolating between
those centres keeps the band aligned with the waveform; past the last centre (samples 993-999) the last value
is held. The map has one value every 0.32 s, and each position's receptive field spans 859 samples (8.6 s), so
the band is coarse: it says around when, not which beat or wave.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.models.resnet1d import ResBlock1d, ResNet1d

POINTWISE = (nn.BatchNorm1d, nn.ReLU, nn.Dropout, nn.Identity)  # no effect on where a time position sits


@dataclass(frozen=True)
class Geometry:
    """Where the hooked layer's time positions sit in the input, in samples."""

    stride: int               # input samples between neighbouring feature positions
    offset: float             # input sample at the centre of feature position 0
    receptive_field: int      # input samples that can reach one feature position
    length: int               # feature positions for an input of n_samples

    def centres(self) -> np.ndarray:
        """(length,) input sample at the centre of each feature position."""
        return self.offset + self.stride * np.arange(self.length, dtype=np.float64)


class _Tracker:
    """Composes the centre, spacing, receptive field and output length of a chain of Conv1d / MaxPool1d layers."""

    def __init__(self, n_samples: int) -> None:
        self.jump, self.start, self.rf, self.length = 1, 0.0, 1, n_samples

    def copy(self) -> _Tracker:
        t = _Tracker(self.length)
        t.jump, t.start, t.rf = self.jump, self.start, self.rf
        return t

    def same_place(self, other: _Tracker) -> bool:
        return (self.jump, self.start, self.length) == (other.jump, other.start, other.length)

    def layer(self, m: nn.Module) -> None:
        if isinstance(m, POINTWISE):
            return
        if not isinstance(m, (nn.Conv1d, nn.MaxPool1d)):
            raise TypeError(f"feature_geometry does not know how {type(m).__name__} moves time positions")
        k, s, p, d = (int(np.atleast_1d(v)[0]) for v in (m.kernel_size, m.stride, m.padding, m.dilation))
        if isinstance(m, nn.MaxPool1d) and m.ceil_mode:
            raise ValueError("feature_geometry assumes MaxPool1d with ceil_mode=False")
        self.start += (d * (k - 1) / 2 - p) * self.jump  # centre of output 0, in input samples
        self.rf += d * (k - 1) * self.jump
        self.jump *= s
        self.length = (self.length + 2 * p - d * (k - 1) - 1) // s + 1
        if self.length < 1:
            raise ValueError(f"The input is too short: {type(m).__name__} leaves no time positions")

    def block(self, b: ResBlock1d) -> None:
        main, skip = self.copy(), self.copy()
        for m in (b.conv1, b.bn1, b.dropout, b.conv2, b.bn2):
            main.layer(m)
        skip.layer(b.skip)
        if not main.same_place(skip):
            raise ValueError("A residual block's main and skip paths put the time positions in different places")
        self.jump, self.start, self.length = main.jump, main.start, main.length
        self.rf = max(main.rf, skip.rf)


def feature_geometry(model: ResNet1d, n_samples: int) -> Geometry:
    """Centre spacing, offset, receptive field and length of the last residual stage's time positions."""
    t = _Tracker(n_samples)
    for m in model.stem:
        t.layer(m)
    for stage in model.stages:
        for b in stage:
            if not isinstance(b, ResBlock1d):
                raise TypeError(f"Expected ResBlock1d in the stages, got {type(b).__name__}")
            t.block(b)
    return Geometry(stride=t.jump, offset=t.start, receptive_field=t.rf, length=t.length)


@dataclass(frozen=True)
class GradCAMResult:
    """Grad-CAM of one target class for a batch of records. Arrays are float64 NumPy, one row per record."""

    class_index: int
    maps: np.ndarray            # (records, samples) in [0, 1]: the normalised map at every input sample
    contributions: np.ndarray   # (records, L) sum_k alpha_k A_k(t) before the ReLU; sums to logit - bias
    raw_peak: np.ndarray        # (records,) max of the ReLU'd, interpolated map before step 6; 0: no positive evidence
    logits: np.ndarray          # (records,) the target class's raw logit (uncalibrated)
    centres: np.ndarray         # (L,) input sample at the centre of each feature position
    receptive_field: int        # input samples that can reach one feature position


def interpolate(cam: np.ndarray, centres: np.ndarray, n_samples: int) -> np.ndarray:
    """(records, L) values at the feature centres -> (records, n_samples), linear, held constant past either end."""
    t = np.arange(n_samples, dtype=np.float64)
    return np.stack([np.interp(t, centres, row) for row in np.asarray(cam, dtype=np.float64)])


def normalise(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide each non-negative row by its maximum. Returns (maps in [0, 1], the maxima). An all-zero row stays zero."""
    maps = np.asarray(maps, dtype=np.float64)
    if (maps < 0).any() or not np.isfinite(maps).all():
        raise ValueError("Grad-CAM maps must be finite and non-negative before normalisation")
    peak = maps.max(axis=1)
    scale = np.where(peak > 0, peak, 1.0)  # an all-zero row is divided by 1, i.e. left at zero
    return maps / scale[:, None], peak


class GradCAM1d:
    """Grad-CAM on the last residual stage of a ResNet1d. Call it with a batch and a class index."""

    def __init__(self, model: ResNet1d) -> None:
        if not isinstance(model, ResNet1d):
            raise TypeError(f"GradCAM1d expects a ResNet1d, got {type(model).__name__}")
        self.model = model
        self.layer = model.stages[-1]
        self._activations: torch.Tensor | None = None

    def _capture(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        if not output.requires_grad:  # frozen weights: start the graph here, so the head can still be differentiated
            output = output.detach().requires_grad_(True)
        self._activations = output
        return output

    def __call__(self, x: torch.Tensor, class_index: int) -> GradCAMResult:
        model = self.model
        if model.training:
            raise RuntimeError("Grad-CAM needs model.eval(): in training mode dropout is random and BatchNorm "
                               "mixes the records of the batch, so one record's map would depend on the others")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Grad-CAM needs gradients; call it outside torch.inference_mode()")
        if not 0 <= class_index < model.n_classes:
            raise ValueError(f"class_index {class_index} is outside 0..{model.n_classes - 1}")
        if x.ndim != 3:
            raise ValueError(f"Expected a batch (records, leads, samples), got shape {tuple(x.shape)}")
        n_samples = x.shape[-1]
        geometry = feature_geometry(model, n_samples)

        self._activations = None
        handle = self.layer.register_forward_hook(self._capture)
        try:
            with torch.enable_grad():
                logits = model(x)
                acts = self._activations
                if acts is None:
                    raise RuntimeError("The forward hook on the last residual stage did not fire")
                # Records are independent in eval mode, so the gradient of the summed logits w.r.t. A gives
                # each record's own dy_c/dA.
                (grads,) = torch.autograd.grad(logits[:, class_index].sum(), acts)
        finally:
            handle.remove()
            self._activations = None

        if acts.shape[-1] != geometry.length:
            raise RuntimeError(f"The last stage has {acts.shape[-1]} time positions; feature_geometry expected "
                               f"{geometry.length}")
        with torch.no_grad():
            alpha = grads.mean(dim=-1, keepdim=True)            # (records, K, 1): step 3
            contributions = (alpha * acts).sum(dim=1)            # (records, L): step 4 before the ReLU
            cam = torch.relu(contributions)                      # step 4
        cam_np = cam.double().cpu().numpy()
        centres = geometry.centres()
        maps, peak = normalise(interpolate(cam_np, centres, n_samples))  # steps 5 and 6
        return GradCAMResult(
            class_index=class_index,
            maps=maps,
            contributions=contributions.double().cpu().numpy(),
            raw_peak=peak,
            logits=logits[:, class_index].detach().double().cpu().numpy(),
            centres=centres,
            receptive_field=geometry.receptive_field,
        )
