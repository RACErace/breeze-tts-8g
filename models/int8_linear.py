"""Weight-only int8 linear layers backed by a Triton GEMM kernel.

The TTS decode path multiplies tiny activation batches (1-2 rows) by large
weight matrices, so those GEMMs are weight-bandwidth bound. Storing weights as
int8 with one scale per output channel halves the bytes streamed from HBM;
the kernel dequantizes tiles to bf16 and uses tensor cores for the dot product.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _int8_weight_gemm(
    x_ptr,
    w_ptr,
    scale_ptr,
    out_ptr,
    M,
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        x = tl.load(
            x_ptr + rows[:, None] * K + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + ks[:, None] * N + cols[None, :],
            mask=k_mask[:, None] & col_mask[None, :],
            other=0,
        )
        scale = tl.load(scale_ptr + cols, mask=col_mask, other=0.0)
        wf = w.to(tl.bfloat16) * scale[None, :].to(tl.bfloat16)
        acc = tl.dot(x, wf, acc)
    out = acc.to(tl.bfloat16)
    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :],
        out,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric int8 quantization, transposed for the kernel."""
    scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.t().contiguous(), scale.float().contiguous()


def _tuned_launch_config(k: int) -> tuple[int, int, int, int]:
    """Block shape / occupancy tuned per reduction size (measured on RTX 4060).

    Long-K projections (the MLP down projections) reach ~200 GB/s with a wider
    K tile and a third pipeline stage; gate/up projections are fastest with the
    flatter default tile.
    """
    if os.environ.get("BREEZE_INT8_TUNED", "1") == "0":
        return 64, 64, 4, 3
    if k >= 4096:
        return 64, 256, 4, 3
    return 64, 64, 4, 2


def _launch_int8_gemm(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    M, K = x.shape
    N = qweight_t.shape[1]
    block_n, block_k, num_warps, num_stages = _tuned_launch_config(K)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, block_n))
    _int8_weight_gemm[grid](
        x,
        qweight_t,
        scales,
        out,
        M,
        K,
        N,
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if bias is not None:
        out = out + bias
    return out


@torch.library.custom_op("breeze_int8::linear", mutates_args=())
def int8_linear(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Opaque op kept outside torch.compile graphs (Triton kernel inside)."""
    return _launch_int8_gemm(x, qweight_t, scales, bias)


@int8_linear.register_fake
def _(x, qweight_t, scales, bias):
    return x.new_empty((*x.shape[:-1], qweight_t.shape[1]))


class Int8Linear(nn.Module):
    """Drop-in inference replacement for nn.Linear (bf16 activations)."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        qweight_t, scales = quantize_weight(weight.detach())
        self.register_buffer("qweight_t", qweight_t, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone(), persistent=False)
        else:
            self.bias = None
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.shape[-1] != self.in_features:
            # Fall back for unexpected dtypes/shapes (not used by the TTS path).
            weight = (
                self.qweight_t.to(torch.bfloat16) * self.scales.to(torch.bfloat16)
            ).t()
            out = x @ weight
            if self.bias is not None:
                out = out + self.bias
            return out
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = int8_linear(flat, self.qweight_t, self.scales, self.bias)
        return out.view(*shape[:-1], self.out_features)


def quantize_mlp_linears(model) -> int:
    """Replace gate/up/down projections of every decoder layer's MLP with int8.

    Only the large MLP projections are quantized: for the tiny-batch decode
    shapes they are faster than bf16, while the small QKV/O projections are
    faster in bf16.
    """
    replaced = 0
    for layer in getattr(model, "layers", ()) or ():
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(mlp, name, None)
            if isinstance(linear, nn.Linear):
                setattr(mlp, name, Int8Linear(linear.weight, linear.bias))
                replaced += 1
    return replaced
