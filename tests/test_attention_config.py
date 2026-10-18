from types import SimpleNamespace

import pytest
import torch

from breeze_infer.runtime import resolve_attn_implementation
from models.attention_utils import (
    force_attention_implementation,
    is_flash_attention_implementation,
)
from models.breeze import _resolve_text_encoder_attn_implementation
from models.cudagraph.backbone_graph import BackboneGraph


def _config(*, caller=None, preferred=None):
    return SimpleNamespace(
        _attn_implementation=caller,
        text_encoder_config=SimpleNamespace(
            preferred_attn_implementation=preferred
        ),
    )


def test_explicit_attention_choice_overrides_checkpoint_preference() -> None:
    config = _config(caller="eager", preferred="flash_attention_2")

    assert _resolve_text_encoder_attn_implementation(config) == "eager"


def test_checkpoint_attention_preference_is_used_without_explicit_choice() -> None:
    config = _config(caller=None, preferred="sdpa")

    assert _resolve_text_encoder_attn_implementation(config) == "sdpa"


def test_text_encoder_attention_keeps_legacy_default() -> None:
    config = _config(caller=None, preferred=None)

    assert _resolve_text_encoder_attn_implementation(config) == "flash_attention_2"


def test_flash_attention_implementation_detection() -> None:
    assert is_flash_attention_implementation("flash_attention_2")
    assert is_flash_attention_implementation("flash_attention_3")
    assert is_flash_attention_implementation("kernels-community/flash-attn")
    assert not is_flash_attention_implementation("eager")
    assert not is_flash_attention_implementation("sdpa")
    assert not is_flash_attention_implementation(None)


def test_force_attention_implementation_updates_config_and_layers() -> None:
    class Attention:
        def __init__(self, config):
            self.config = config

    class Layer:
        def __init__(self, config):
            self.self_attn = Attention(config)

    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        layers=[
            Layer(SimpleNamespace(_attn_implementation="flash_attention_2")),
            Layer(SimpleNamespace(_attn_implementation="flash_attention_2")),
        ],
    )

    force_attention_implementation(module, "eager")

    assert module.config._attn_implementation == "eager"
    for layer in module.layers:
        assert layer.self_attn.config._attn_implementation == "eager"


def test_force_attention_implementation_handles_shared_config() -> None:
    class Attention:
        def __init__(self, config):
            self.config = config

    class Layer:
        def __init__(self, config):
            self.self_attn = Attention(config)

    shared = SimpleNamespace(_attn_implementation="flash_attention_2")
    module = SimpleNamespace(config=shared, layers=[Layer(shared)])

    force_attention_implementation(module, "eager")

    assert shared._attn_implementation == "eager"


def test_resolve_attn_implementation_prefers_flash_when_available(monkeypatch) -> None:
    monkeypatch.setattr("breeze_infer.runtime.flash_attention_available", lambda: True)
    assert resolve_attn_implementation() == "flash_attention_2"
    assert resolve_attn_implementation("eager") == "eager"


def test_resolve_attn_implementation_falls_back_to_eager(monkeypatch) -> None:
    monkeypatch.setattr("breeze_infer.runtime.flash_attention_available", lambda: False)

    assert resolve_attn_implementation() == "eager"
    with pytest.raises(ImportError, match="flash-attn"):
        resolve_attn_implementation("flash_attention_2")


def _flash_backbone_graph() -> BackboneGraph:
    graph = BackboneGraph.__new__(BackboneGraph)
    graph.model = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="flash_attention_2")
    )
    graph.batch_size = 2
    graph.max_seq_len = 8
    graph.device = torch.device("cpu")
    graph.dtype = torch.float32
    graph._pad_lens = torch.tensor([0, 2])
    graph._kv_indices = torch.arange(8)
    graph.attn_mask = torch.ones(2, 8, dtype=torch.bool)
    return graph


def test_backbone_graph_flash_decode_mask_is_2d_validity() -> None:
    graph = _flash_backbone_graph()

    graph._set_attention_mask(4)

    expected = torch.tensor(
        [
            [True, True, True, True, True, False, False, False],
            [False, False, True, True, True, False, False, False],
        ]
    )
    assert graph.attn_mask.dtype == torch.bool
    assert torch.equal(graph.attn_mask, expected)


def test_backbone_graph_eager_decode_mask_stays_additive_4d() -> None:
    graph = BackboneGraph.__new__(BackboneGraph)
    graph.model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
    graph.batch_size = 2
    graph._pad_lens = torch.tensor([0, 2])
    graph._kv_indices = torch.arange(8)
    graph.attn_mask = torch.zeros(2, 1, 1, 8, dtype=torch.float32)
    graph._mask_min_val = torch.finfo(torch.float32).min

    graph._set_attention_mask(4)

    assert graph.attn_mask.shape == (2, 1, 1, 8)
    assert graph.attn_mask[0, 0, 0, :5].tolist() == [0.0] * 5
    assert (graph.attn_mask == torch.finfo(torch.float32).min).sum().item() == 8


def test_backbone_graph_capture_falls_back_to_eager_for_flash() -> None:
    layer_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    graph = BackboneGraph.__new__(BackboneGraph)
    graph.model = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(config=layer_config),
            )
        ],
    )

    graph._use_static_mask_attention()

    assert graph.model.config._attn_implementation == "eager"
    assert layer_config._attn_implementation == "eager"
