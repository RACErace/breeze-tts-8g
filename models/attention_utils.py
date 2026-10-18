"""Attention-backend helpers shared by the eager runtime and CUDA-graph stages.

The inference runtime can be loaded with ``attn_implementation="flash_attention_2"``.
That backend works for the eager streaming path (text encoder / backbone
prefill / backbone decode), but two inference components must keep a
mask-compatible backend:

* CUDA-graph stages capture static shapes. FlashAttention's padding-mask path
  unpads dynamically (``torch.nonzero``), so it cannot be captured in a graph.
  Those stages fall back to the pre-existing eager backend.
* The depth decoder decodes at most ``max_seq_len`` (dozens of) tokens. Flash
  attention has no advantage there, and the static-mask graph/batch paths need
  the eager backend.

Both are handled by switching the affected modules in place with
:func:`force_attention_implementation`.
"""

from __future__ import annotations

from typing import Any


def is_flash_attention_implementation(implementation: str | None) -> bool:
    """Return whether ``implementation`` selects a FlashAttention kernel."""
    if not implementation:
        return False
    implementation = str(implementation)
    if implementation.startswith("flash_attention"):
        return True
    return "flash-attn" in implementation


def force_attention_implementation(module: Any, implementation: str) -> None:
    """Switch a decoder stack to ``implementation`` in place.

    Sets both the config used for causal-mask creation (``module.config``) and
    the per-layer attention configs that attention modules read at call time.
    Encoder-style models (T5Gemma2) share one config object between the module
    and its layers, which makes the layer update a no-op there.
    """
    config = getattr(module, "config", None)
    if config is not None:
        config._attn_implementation = implementation
    for layer in getattr(module, "layers", ()) or ():
        attention = getattr(layer, "self_attn", None)
        layer_config = getattr(attention, "config", None)
        if layer_config is not None and layer_config is not config:
            layer_config._attn_implementation = implementation
