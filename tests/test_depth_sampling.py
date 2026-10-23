"""Light-weight tests for the depth decoder sampling configuration."""

from types import SimpleNamespace

import pytest
import torch

from models.cudagraph.depth_decoder_graph import DepthDecoderGraph


def _bare_graph(*, top_p_active: bool, half: int = 2) -> SimpleNamespace:
    graph = SimpleNamespace(
        _top_p_active=top_p_active,
        half=half,
        top_p_buf=torch.zeros(half, 1),
    )
    graph.set_top_p = DepthDecoderGraph.set_top_p.__get__(graph)
    return graph


def test_top_p_below_one_is_rejected_when_graph_was_captured_without_it() -> None:
    graph = _bare_graph(top_p_active=False)

    graph.set_top_p(1.0)
    assert torch.equal(graph.top_p_buf, torch.zeros(2, 1))

    with pytest.raises(RuntimeError, match="top_p"):
        graph.set_top_p(0.9)


def test_top_p_below_one_updates_buffer_when_captured_with_it() -> None:
    graph = _bare_graph(top_p_active=True)

    graph.set_top_p(0.9)

    assert torch.allclose(graph.top_p_buf, torch.full((2, 1), 0.9))
