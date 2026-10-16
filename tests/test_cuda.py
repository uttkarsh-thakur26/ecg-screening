"""The training GPU is visible and computes correctly. Run this before any training run.

Fails, not skips, without CUDA: on the ASUS TUF A15 that usually means Armoury Crate GPU Mode is Eco.
"""

from __future__ import annotations

import torch

from src.config import load_config

CFG = load_config()


def test_cuda_is_available():
    assert torch.cuda.is_available(), (
        "torch cannot see a CUDA GPU. If the RTX 3050 is powered off, set Armoury Crate GPU Mode to Standard."
    )


def test_stem_conv_on_gpu_matches_cpu():
    ds, model, training = CFG["dataset"], CFG["model"], CFG["training"]
    torch.manual_seed(0)
    x = torch.randn(training["batch_size"], ds["n_leads"], ds["sampling_rate_hz"] * ds["record_duration_s"])
    conv = torch.nn.Conv1d(
        ds["n_leads"], model["stem_channels"], model["stem_kernel"],
        stride=model["stem_stride"], padding=model["stem_kernel"] // 2,
    )
    expected = conv(x)
    actual = conv.cuda()(x.cuda()).cpu()
    # Loose tolerance: cuDNN may use TF32 on Ampere GPUs such as the RTX 3050.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
