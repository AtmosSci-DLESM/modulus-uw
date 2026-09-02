# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused isolatitude HEALPix pad in native NCHW/NHWC layout.

The ATen gather path permutes folded faces to ``[B, C, F*H*W]``, clones, then
(optionally) copies to channels-last. On v2-12_det that permute/gather/scatter/copy
chain is a large fraction of CUDA-graph time. These Triton kernels gather (fwd)
and scatter-add (bwd) using the same precomputed indices, writing the output
directly in the requested memory format so the extra copies are unnecessary.

``index1[o] < 0`` means a single source (``out = x[index0]``); otherwise
``out = 0.5 * (x[index0] + x[index1])``.
"""

from __future__ import annotations

import torch as th

_HPX_FACES = 12

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:
    _HAVE_TRITON = False


def isolatitude_pad_cuda_available() -> bool:
    """True when the fused CUDA kernel can run (Triton + GPU)."""
    return _HAVE_TRITON and th.cuda.is_available()


if _HAVE_TRITON:

    @triton.jit
    def _isolatitude_pad_fwd_kernel(
        x_ptr,
        y_ptr,
        index0_ptr,
        index1_ptr,
        B,
        C,
        H,
        W,
        Hp,
        Wp,
        stride_xn,
        stride_xc,
        stride_xh,
        stride_xw,
        stride_yn,
        stride_yc,
        stride_yh,
        stride_yw,
        BLOCK_C: tl.constexpr,
        FACES: tl.constexpr,
    ):
        b = tl.program_id(0)
        o = tl.program_id(1)
        c0 = tl.program_id(2) * BLOCK_C
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        face_out = Hp * Wp
        in_area = H * W
        f = o // face_out
        rem = o - f * face_out
        hp = rem // Wp
        wp = rem - hp * Wp

        s0 = tl.load(index0_ptr + o)
        s1 = tl.load(index1_ptr + o)

        f0 = s0 // in_area
        hw0 = s0 - f0 * in_area
        h0 = hw0 // W
        w0 = hw0 - h0 * W
        n0 = b * FACES + f0
        n_out = b * FACES + f

        x0 = x_ptr + n0 * stride_xn + h0 * stride_xh + w0 * stride_xw + offs_c * stride_xc
        v = tl.load(x0, mask=mask_c, other=0)

        blend = s1 >= 0
        f1 = s1 // in_area
        hw1 = s1 - f1 * in_area
        h1 = hw1 // W
        w1 = hw1 - h1 * W
        n1 = b * FACES + f1
        x1 = x_ptr + n1 * stride_xn + h1 * stride_xh + w1 * stride_xw + offs_c * stride_xc
        v1 = tl.load(x1, mask=mask_c & blend, other=0)
        v = tl.where(blend, (v + v1) * 0.5, v)

        y = y_ptr + n_out * stride_yn + hp * stride_yh + wp * stride_yw + offs_c * stride_yc
        tl.store(y, v, mask=mask_c)

    @triton.jit
    def _isolatitude_pad_bwd_kernel(
        gy_ptr,
        gx_ptr,
        index0_ptr,
        index1_ptr,
        B,
        C,
        H,
        W,
        Hp,
        Wp,
        stride_gxn,
        stride_gxc,
        stride_gxh,
        stride_gxw,
        stride_gyn,
        stride_gyc,
        stride_gyh,
        stride_gyw,
        BLOCK_C: tl.constexpr,
        FACES: tl.constexpr,
    ):
        b = tl.program_id(0)
        o = tl.program_id(1)
        c0 = tl.program_id(2) * BLOCK_C
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        face_out = Hp * Wp
        in_area = H * W
        f = o // face_out
        rem = o - f * face_out
        hp = rem // Wp
        wp = rem - hp * Wp
        n_out = b * FACES + f

        gy = gy_ptr + n_out * stride_gyn + hp * stride_gyh + wp * stride_gyw + offs_c * stride_gyc
        g = tl.load(gy, mask=mask_c, other=0)

        s0 = tl.load(index0_ptr + o)
        s1 = tl.load(index1_ptr + o)
        blend = s1 >= 0
        g0 = tl.where(blend, g * 0.5, g)

        f0 = s0 // in_area
        hw0 = s0 - f0 * in_area
        h0 = hw0 // W
        w0 = hw0 - h0 * W
        n0 = b * FACES + f0
        gx0 = gx_ptr + n0 * stride_gxn + h0 * stride_gxh + w0 * stride_gxw + offs_c * stride_gxc
        tl.atomic_add(gx0, g0, mask=mask_c)

        f1 = s1 // in_area
        hw1 = s1 - f1 * in_area
        h1 = hw1 // W
        w1 = hw1 - h1 * W
        n1 = b * FACES + f1
        gx1 = gx_ptr + n1 * stride_gxn + h1 * stride_gxh + w1 * stride_gxw + offs_c * stride_gxc
        tl.atomic_add(gx1, g0, mask=mask_c & blend)


def _block_c(channels: int) -> int:
    if channels <= 32:
        return 32
    if channels <= 64:
        return 64
    if channels <= 128:
        return 128
    return 256


def _launch_fwd(x: th.Tensor, y: th.Tensor, index0: th.Tensor, index1: th.Tensor) -> None:
    BF, C, H, W = x.shape
    _, _, Hp, Wp = y.shape
    B = BF // _HPX_FACES
    nout = _HPX_FACES * Hp * Wp
    block_c = _block_c(C)
    grid = (B, nout, triton.cdiv(C, block_c))
    _isolatitude_pad_fwd_kernel[grid](
        x,
        y,
        index0,
        index1,
        B,
        C,
        H,
        W,
        Hp,
        Wp,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        BLOCK_C=block_c,
        FACES=_HPX_FACES,
    )


def _launch_bwd(gy: th.Tensor, gx: th.Tensor, index0: th.Tensor, index1: th.Tensor) -> None:
    BF, C, H, W = gx.shape
    _, _, Hp, Wp = gy.shape
    B = BF // _HPX_FACES
    nout = _HPX_FACES * Hp * Wp
    block_c = _block_c(C)
    grid = (B, nout, triton.cdiv(C, block_c))
    _isolatitude_pad_bwd_kernel[grid](
        gy,
        gx,
        index0,
        index1,
        B,
        C,
        H,
        W,
        Hp,
        Wp,
        gx.stride(0),
        gx.stride(1),
        gx.stride(2),
        gx.stride(3),
        gy.stride(0),
        gy.stride(1),
        gy.stride(2),
        gy.stride(3),
        BLOCK_C=block_c,
        FACES=_HPX_FACES,
    )


class IsolatitudePadTritonFunction(th.autograd.Function):
    """Linear isolatitude pad with a fused gather fwd / scatter-add bwd."""

    @staticmethod
    def forward(
        ctx,
        data: th.Tensor,
        index0: th.Tensor,
        index1: th.Tensor,
        padding: int,
        enable_nhwc: bool,
    ) -> th.Tensor:
        BF, C, H, W = data.shape
        Hp = H + 2 * int(padding)
        Wp = W + 2 * int(padding)
        mem = th.channels_last if enable_nhwc else th.contiguous_format
        out = th.empty(
            (BF, C, Hp, Wp),
            device=data.device,
            dtype=data.dtype,
            memory_format=mem,
        )
        idx0 = index0 if index0.device == data.device else index0.to(device=data.device, non_blocking=True)
        idx1 = index1 if index1.device == data.device else index1.to(device=data.device, non_blocking=True)
        _launch_fwd(data, out, idx0, idx1)
        ctx.save_for_backward(idx0, idx1)
        ctx.input_shape = (BF, C, H, W)
        ctx.input_channels_last = bool(
            enable_nhwc or data.is_contiguous(memory_format=th.channels_last)
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: th.Tensor):
        index0, index1 = ctx.saved_tensors
        BF, C, H, W = ctx.input_shape
        mem = th.channels_last if ctx.input_channels_last else th.contiguous_format
        grad_data = th.empty(
            (BF, C, H, W),
            device=grad_output.device,
            dtype=grad_output.dtype,
            memory_format=mem,
        ).zero_()
        _launch_bwd(grad_output.contiguous(memory_format=mem), grad_data, index0, index1)
        return grad_data, None, None, None, None


def isolatitude_pad_triton(
    data: th.Tensor,
    index0: th.Tensor,
    index1: th.Tensor,
    padding: int,
    enable_nhwc: bool,
) -> th.Tensor:
    """Apply fused isolatitude pad. ``index1`` uses ``-1`` where there is no second source."""
    if not isolatitude_pad_cuda_available():
        raise RuntimeError("isolatitude_pad_triton requires Triton and CUDA")
    if data.ndim != 4:
        raise ValueError(f"expected [N*12, C, H, W], got {tuple(data.shape)}")
    if data.shape[0] % _HPX_FACES != 0:
        raise ValueError(
            f"Folded batch {data.shape[0]} is not divisible by {_HPX_FACES} HEALPix faces"
        )
    return IsolatitudePadTritonFunction.apply(
        data, index0, index1, int(padding), bool(enable_nhwc)
    )
