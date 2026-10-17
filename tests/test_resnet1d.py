"""ResNet1d unit tests. Random tensors are fine here: these test the architecture, not the data."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.resnet1d import ResBlock1d, build_model, count_parameters, summary

B, LEADS, SAMPLES, CLASSES = 4, 12, 1000, 5


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return build_model(cfg).eval()


def test_output_is_one_logit_per_class_and_features_come_from_the_last_stage(cfg, model):
    x = torch.randn(B, LEADS, SAMPLES)
    assert model(x).shape == (B, CLASSES)
    assert model.forward_features(x).shape == (B, cfg["model"]["stage_channels"][-1], 32)


def test_parameter_count_is_in_the_spec_range(model):
    n = count_parameters(model)
    assert 5_000_000 <= n <= 10_000_000  # spec section 3: "roughly 5-10M parameters"
    assert n == 6_290_821  # printed by `python -m src.models.resnet1d`; changes only if the architecture does


def test_summary_parts_add_up(cfg, model):
    text = summary(cfg)
    assert text.splitlines()[-1].split()[-1] == f"{count_parameters(model):,}"


def test_layers_match_the_spec(cfg, model):
    m = cfg["model"]
    conv, bn, relu, pool = model.stem
    assert (conv.in_channels, conv.out_channels) == (LEADS, m["stem_channels"]) == (12, 64)
    assert (conv.kernel_size, conv.stride) == ((7,), (2,))
    assert isinstance(bn, nn.BatchNorm1d) and isinstance(relu, nn.ReLU) and isinstance(pool, nn.MaxPool1d)
    assert [stage[0].conv1.out_channels for stage in model.stages] == [64, 128, 256, 512]
    assert [stage[0].conv1.stride[0] for stage in model.stages] == [1, 2, 2, 2]
    for stage in model.stages:
        assert len(stage) == 2
        for block in stage:
            for c in (block.conv1, block.conv2):
                assert c.kernel_size == (5,) and c.padding == (2,)
            assert block.dropout.p == m["block_dropout"]
    assert model.dropout.p == 0.3
    assert (model.fc.in_features, model.fc.out_features) == (512, CLASSES)


def test_no_softmax_or_sigmoid_inside_the_model(model):
    assert not any(isinstance(m, (nn.Softmax, nn.Sigmoid, nn.LogSoftmax)) for m in model.modules())


def test_skip_path_is_identity_unless_channels_or_stride_change():
    same = ResBlock1d(64, 64, 5, 1, 0.0)
    assert isinstance(same.skip, nn.Identity)
    for in_ch, out_ch, stride in [(64, 128, 2), (64, 128, 1), (64, 64, 2)]:
        block = ResBlock1d(in_ch, out_ch, 5, stride, 0.0)
        assert isinstance(block.skip, nn.Conv1d)
        assert block.skip.kernel_size == (1,) and block.skip.stride == (stride,)
        x = torch.randn(2, in_ch, 125)
        assert block(x).shape == block.skip(x).shape


def test_identity_block_adds_its_input():
    block = ResBlock1d(8, 8, 5, 1, 0.0).eval()
    for p in (block.bn2.weight, block.bn2.bias):
        nn.init.zeros_(p)  # main path now outputs 0, so the block reduces to ReLU(x)
    x = torch.randn(2, 8, 50)
    torch.testing.assert_close(block(x), torch.relu(x))


def test_head_is_global_average_pooling_then_linear(model):
    x = torch.randn(B, LEADS, SAMPLES)
    with torch.no_grad():
        expected = model.fc(model.forward_features(x).mean(dim=-1))  # dropout is inactive in eval mode
        torch.testing.assert_close(model(x), expected)


def test_wrong_number_of_leads_fails_loudly(model):
    with pytest.raises(ValueError, match="12"):
        model(torch.randn(B, 11, SAMPLES))


def test_every_parameter_gets_a_gradient(cfg):
    torch.manual_seed(0)
    model = build_model(cfg).train()
    model(torch.randn(B, LEADS, SAMPLES)).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing
