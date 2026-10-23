"""Tests for the weight-only int8 linear layers."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.int8_linear import Int8Linear, quantize_mlp_linears


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs CUDA")
def test_int8_linear_matches_quantized_dequantized_linear() -> None:
    torch.manual_seed(0)
    weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    layer = Int8Linear(weight)

    dequant = (
        layer.qweight_t.to(torch.bfloat16) * layer.scales.to(torch.bfloat16)
    ).t()
    expected = torch.nn.functional.linear(x, dequant)
    actual = layer(x)

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert cosine > 0.9999


def test_quantize_mlp_linears_replaces_all_three_projections() -> None:
    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.gate_proj = nn.Linear(8, 16, bias=False)
            self.mlp.up_proj = nn.Linear(8, 16, bias=False)
            self.mlp.down_proj = nn.Linear(16, 8, bias=False)

    model = SimpleNamespace(layers=[Layer(), Layer()])

    assert quantize_mlp_linears(model) == 6
    for layer in model.layers:
        assert isinstance(layer.mlp.gate_proj, Int8Linear)
        assert isinstance(layer.mlp.up_proj, Int8Linear)
        assert isinstance(layer.mlp.down_proj, Int8Linear)
    assert quantize_mlp_linears(model) == 0
