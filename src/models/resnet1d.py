"""1D residual CNN for 12-lead ECGs (PROJECT_SPEC.md section 3).

    python -m src.models.resnet1d      # layer output shapes and the parameter count

Input (batch, 12 leads, 1000 samples), output 5 logits, one per superclass. The sigmoid is applied
outside the model (BCEWithLogitsLoss in training, torch.sigmoid at inference), one per class, because
a record can belong to several superclasses at once.

    Stem      Conv1d(12 -> 64, k=7, stride 2) -> BN -> ReLU -> MaxPool
    Stage 1   2 x ResBlock(64),  stride 1
    Stage 2   2 x ResBlock(128), stride 2
    Stage 3   2 x ResBlock(256), stride 2
    Stage 4   2 x ResBlock(512), stride 2
    Head      global average pooling over time -> Dropout(0.3) -> Linear(512 -> 5)

Every size comes from config `model`. For a 10 s record at 100 Hz the time axis shrinks
1000 -> 500 (stem conv) -> 250 (max pool) -> 250, 125, 63, 32 (stages 1-4).

ResBlock: Conv1d(k=5, pad=2) -> BN -> ReLU -> Dropout -> Conv1d(k=5, pad=2) -> BN, added to the
skip path, then ReLU. The skip path is the identity, or a 1x1 convolution when the block changes
the channel count or the stride, as the spec says. There is no BatchNorm on that 1x1 convolution
(torchvision's ResNet has one; Ribeiro et al. 2020, a residual network for 12-lead ECGs, does not),
and it has no bias because the sum already gets a per-channel shift from the main path's last BN.
In each stage the first block's first convolution carries the stride.

Global average pooling is kept before the classifier because Grad-CAM (section 5) weights the
last stage's feature maps, (batch, 512, 32), by the gradients that reach them through the average.
forward_features returns exactly those maps, and `stages[-1]` is the module to hook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.config import DEFAULT_CONFIG_PATH, load_config


class ResBlock1d(nn.Module):
    """Two k-wide convolutions with BatchNorm, dropout between them, and a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, dropout: float) -> None:
        super().__init__()
        if kernel % 2 != 1:
            raise ValueError(f"model.block_kernel = {kernel} must be odd so that padding kernel // 2 keeps the length")
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels == out_channels and stride == 1:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ResNet1d(nn.Module):
    """Stem, residual stages, global average pooling, dropout and a linear layer giving one logit per class."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        *,
        stem_channels: int,
        stem_kernel: int,
        stem_stride: int,
        stem_pool_kernel: int,
        stem_pool_stride: int,
        stage_channels: list[int],
        stage_strides: list[int],
        blocks_per_stage: int,
        block_kernel: int,
        block_dropout: float,
        head_dropout: float,
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(stage_strides):
            raise ValueError(f"model.stage_channels {stage_channels} and model.stage_strides {stage_strides} "
                             "must have one entry per stage")
        if blocks_per_stage < 1:
            raise ValueError(f"model.blocks_per_stage = {blocks_per_stage} must be at least 1")
        for name, k in (("stem_kernel", stem_kernel), ("stem_pool_kernel", stem_pool_kernel)):
            if k % 2 != 1:
                raise ValueError(f"model.{name} = {k} must be odd so that padding k // 2 centres the window")
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, stem_kernel, stride=stem_stride, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(),
            nn.MaxPool1d(stem_pool_kernel, stride=stem_pool_stride, padding=stem_pool_kernel // 2),
        )
        stages = []
        channels = stem_channels
        for out_channels, stride in zip(stage_channels, stage_strides):
            blocks = [ResBlock1d(channels, out_channels, block_kernel, stride, block_dropout)]
            blocks += [ResBlock1d(out_channels, out_channels, block_kernel, 1, block_dropout)
                       for _ in range(blocks_per_stage - 1)]
            stages.append(nn.Sequential(*blocks))
            channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(channels, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        # He initialisation for convolutions followed by ReLU (He et al. 2015), as torchvision's ResNet does;
        # BatchNorm starts as the identity (weight 1, bias 0) and the linear layer keeps PyTorch's default.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, leads, samples) -> the last residual stage's feature maps (batch, channels, time)."""
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input of shape (batch, {self.in_channels}, samples), got {tuple(x.shape)}")
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, leads, samples) -> (batch, n_classes) logits. No sigmoid here."""
        pooled = self.forward_features(x).mean(dim=-1)  # global average pooling over time
        return self.fc(self.dropout(pooled))


def build_model(cfg: dict[str, Any]) -> ResNet1d:
    """The model described by config `model`, for config `dataset.n_leads` inputs and one output per superclass."""
    m = cfg["model"]
    return ResNet1d(
        cfg["dataset"]["n_leads"],
        len(cfg["labels"]["superclasses"]),
        stem_channels=m["stem_channels"],
        stem_kernel=m["stem_kernel"],
        stem_stride=m["stem_stride"],
        stem_pool_kernel=m["stem_pool_kernel"],
        stem_pool_stride=m["stem_pool_stride"],
        stage_channels=list(m["stage_channels"]),
        stage_strides=list(m["stage_strides"]),
        blocks_per_stage=m["blocks_per_stage"],
        block_kernel=m["block_kernel"],
        block_dropout=m["block_dropout"],
        head_dropout=m["head_dropout"],
    )


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def summary(cfg: dict[str, Any]) -> str:
    """Output shape and parameter count of each part, for one record of the configured length."""
    ds = cfg["dataset"]
    n_samples = ds["sampling_rate_hz"] * ds["record_duration_s"]
    # Shapes are traced on the meta device: no values are computed and no signal, real or made up, is involved.
    with torch.device("meta"):
        model = build_model(cfg).eval()
        x = torch.empty(1, ds["n_leads"], n_samples)
        rows = [("Input", tuple(x.shape), 0)]
        x = model.stem(x)
        rows.append(("Stem", tuple(x.shape), count_parameters(model.stem)))
        for i, stage in enumerate(model.stages, start=1):
            x = stage(x)
            rows.append((f"Stage {i}", tuple(x.shape), count_parameters(stage)))
        x = model.fc(model.dropout(x.mean(dim=-1)))
        rows.append(("Head (GAP, dropout, linear)", tuple(x.shape), count_parameters(model.fc)))
    total = count_parameters(model)
    if sum(r[2] for r in rows) != total:
        raise RuntimeError("Per-part parameter counts do not add up to the model total")
    lines = [f"{'Part':<30}{'Output shape':>18}{'Parameters':>14}"]
    lines += [f"{name:<30}{str(shape):>18}{params:>14,}" for name, shape, params in rows]
    lines.append(f"{'Total':<30}{'':>18}{total:>14,}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the 1D ResNet's layer shapes and parameter count.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    print(summary(load_config(args.config)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
