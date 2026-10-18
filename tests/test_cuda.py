"""The training GPU is visible and computes correctly. Run this before any training run.

Fails, not skips, without CUDA, so a missing GPU is never mistaken for a pass. On a machine without an NVIDIA GPU,
run the rest of the suite with `pytest --ignore=tests/test_cuda.py`.
"""

from __future__ import annotations

import torch

from src.config import load_config

CFG = load_config()


def test_cuda_is_available():
    assert torch.cuda.is_available(), (
        "torch cannot see a CUDA GPU. Check that nvidia-smi lists the GPU and that the driver supports this "
        "PyTorch build's CUDA version."
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
    # Loose tolerance: cuDNN may use TF32 on Ampere and newer GPUs.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
