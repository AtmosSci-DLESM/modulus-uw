# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused isolatitude HEALPix pad in native NCHW/NHWC layout.

The ATen gather path permutes folded faces to ``[B, C, F*H*W]``, clones, then
(optionally) copies to channels-last. On v2-12_det that permute/gather/scatter/copy
chain is a large fraction of CUDA-graph time. These Triton kernels gather (fwd)
and scatter-add (bwd) using the same precomputed indices, writing the output
directly in the requested memory format so the extra copies are unnecessary.

Each program covers a ``[BLOCK_O, BLOCK_C]`` tile of output pixels × channels.
One-pixel programs were launch-bound at nside=64 (~50k spatial programs).

``index1[o] < 0`` means a single source (``out = x[index0]``); otherwise
``out = 0.5 * (x[index0] + x[index1])``.

``index0[o] < 0`` means the main programs leave that output alone. The pole
correction uses this for the 24 halo slots it fills itself (``pole_dst`` [24]):
slot ``s`` is ``sum_k pole_w[s, k] * x[pole_src[s, k]]`` over 12 input pixels. The
slot programs are extra programs in the same launch (the first ``N_POLE`` ids on
grid axis 1, so their latency overlaps the pad tiles instead of forming the tail), so the correction adds no kernel launch and no pass over the
padded tensor. In the backward each of the 12 source pixels of a pole gathers
``sum_s pole_w[s, k] * g[slot s]`` over the 12 slots of that pole and adds it with
one atomic, rather than 12 slots each scattering to the same 12 addresses.
"""

from __future__ import annotations

import torch as th

_HPX_FACES = 12
# Output pixels per program. H100 nside=64 C=256 NHWC bf16: 1→0.46ms fwd, 8→0.15ms;
# 64+ slows down; 256 regresses past the one-pixel grid.
_BLOCK_O = 8

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
    def _pad_fwd_body(
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
        nout,
        stride_xn,
        stride_xc,
        stride_xh,
        stride_xw,
        stride_yn,
        stride_yc,
        stride_yh,
        stride_yw,
        BLOCK_O: tl.constexpr,
        BLOCK_C: tl.constexpr,
        FACES: tl.constexpr,
        N_POLE: tl.constexpr,
    ):
        b = tl.program_id(0)
        o0 = (tl.program_id(1) - N_POLE) * BLOCK_O
        c0 = tl.program_id(2) * BLOCK_C
        offs_o = o0 + tl.arange(0, BLOCK_O)
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_o = offs_o < nout
        mask_c = offs_c < C
        mask = mask_o[:, None] & mask_c[None, :]

        face_out = Hp * Wp
        in_area = H * W
        f = offs_o // face_out
        rem = offs_o - f * face_out
        hp = rem // Wp
        wp = rem - hp * Wp

        s0 = tl.load(index0_ptr + offs_o, mask=mask_o, other=0)
        mask = mask & (s0 >= 0)[:, None]  # index0 < 0: slot owned by the pole programs
        s1 = tl.load(index1_ptr + offs_o, mask=mask_o, other=-1)

        f0 = s0 // in_area
        hw0 = s0 - f0 * in_area
        h0 = hw0 // W
        w0 = hw0 - h0 * W
        n0 = b * FACES + f0
        n_out = b * FACES + f

        x0 = (
            x_ptr
            + n0[:, None] * stride_xn
            + h0[:, None] * stride_xh
            + w0[:, None] * stride_xw
            + offs_c[None, :] * stride_xc
        )
        v = tl.load(x0, mask=mask, other=0)

        blend = (s1 >= 0)[:, None]
        f1 = s1 // in_area
        hw1 = s1 - f1 * in_area
        h1 = hw1 // W
        w1 = hw1 - h1 * W
        n1 = b * FACES + f1
        x1 = (
            x_ptr
            + n1[:, None] * stride_xn
            + h1[:, None] * stride_xh
            + w1[:, None] * stride_xw
            + offs_c[None, :] * stride_xc
        )
        v1 = tl.load(x1, mask=mask & blend, other=0)
        v = tl.where(blend, (v + v1) * 0.5, v)

        y = (
            y_ptr
            + n_out[:, None] * stride_yn
            + hp[:, None] * stride_yh
            + wp[:, None] * stride_yw
            + offs_c[None, :] * stride_yc
        )
        tl.store(y, v, mask=mask)

    @triton.jit
    def _pad_bwd_body(
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
        nout,
        stride_gxn,
        stride_gxc,
        stride_gxh,
        stride_gxw,
        stride_gyn,
        stride_gyc,
        stride_gyh,
        stride_gyw,
        BLOCK_O: tl.constexpr,
        BLOCK_C: tl.constexpr,
        FACES: tl.constexpr,
        N_POLE: tl.constexpr,
    ):
        b = tl.program_id(0)
        o0 = (tl.program_id(1) - N_POLE) * BLOCK_O
        c0 = tl.program_id(2) * BLOCK_C
        offs_o = o0 + tl.arange(0, BLOCK_O)
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_o = offs_o < nout
        mask_c = offs_c < C
        mask = mask_o[:, None] & mask_c[None, :]

        face_out = Hp * Wp
        in_area = H * W
        f = offs_o // face_out
        rem = offs_o - f * face_out
        hp = rem // Wp
        wp = rem - hp * Wp
        n_out = b * FACES + f

        gy = (
            gy_ptr
            + n_out[:, None] * stride_gyn
            + hp[:, None] * stride_gyh
            + wp[:, None] * stride_gyw
            + offs_c[None, :] * stride_gyc
        )
        g = tl.load(gy, mask=mask, other=0)

        s0 = tl.load(index0_ptr + offs_o, mask=mask_o, other=0)
        mask = mask & (s0 >= 0)[:, None]  # index0 < 0: slot owned by the pole programs
        s1 = tl.load(index1_ptr + offs_o, mask=mask_o, other=-1)
        blend = (s1 >= 0)[:, None]
        g0 = tl.where(blend, g * 0.5, g)

        f0 = s0 // in_area
        hw0 = s0 - f0 * in_area
        h0 = hw0 // W
        w0 = hw0 - h0 * W
        n0 = b * FACES + f0
        gx0 = (
            gx_ptr
            + n0[:, None] * stride_gxn
            + h0[:, None] * stride_gxh
            + w0[:, None] * stride_gxw
            + offs_c[None, :] * stride_gxc
        )
        tl.atomic_add(gx0, g0, mask=mask)

        f1 = s1 // in_area
        hw1 = s1 - f1 * in_area
        h1 = hw1 // W
        w1 = hw1 - h1 * W
        n1 = b * FACES + f1
        gx1 = (
            gx_ptr
            + n1[:, None] * stride_gxn
            + h1[:, None] * stride_gxh
            + w1[:, None] * stride_gxw
            + offs_c[None, :] * stride_gxc
        )
        tl.atomic_add(gx1, g0, mask=mask & blend)

    @triton.jit
    def _pole_fwd_body(
        x_ptr,
        y_ptr,
        pole_dst_ptr,
        pole_src_ptr,
        pole_w_ptr,
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
        NSRC: tl.constexpr,
    ):
        """y[pole_dst[s]] = sum_k pole_w[s, k] * x[pole_src[s, k]] for slot s = program_id(1)."""
        b = tl.program_id(0)
        slot = tl.program_id(1)
        offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        in_area = H * W
        d = tl.load(pole_dst_ptr + slot)
        f = d // (Hp * Wp)
        rem = d - f * (Hp * Wp)
        hp = rem // Wp
        wp = rem - hp * Wp
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for k in tl.static_range(NSRC):
            sk = tl.load(pole_src_ptr + slot * NSRC + k)
            wk = tl.load(pole_w_ptr + slot * NSRC + k)
            fk = sk // in_area
            hw = sk - fk * in_area
            hk = hw // W
            wk_ = hw - hk * W
            xk = (
                x_ptr
                + (b * FACES + fk) * stride_xn
                + hk * stride_xh
                + wk_ * stride_xw
                + offs_c * stride_xc
            )
            acc += tl.load(xk, mask=mask_c, other=0).to(tl.float32) * wk
        yp = (
            y_ptr
            + (b * FACES + f) * stride_yn
            + hp * stride_yh
            + wp * stride_yw
            + offs_c * stride_yc
        )
        tl.store(yp, acc.to(y_ptr.dtype.element_ty), mask=mask_c)

    @triton.jit
    def _pole_bwd_body(
        gy_ptr,
        gx_ptr,
        pole_dst_ptr,
        pole_src_ptr,
        pole_w_ptr,
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
        NSRC: tl.constexpr,
    ):
        """Program (pole, k) = divmod(program_id(1), 12): gx[src k of the pole] += sum_s w[s, k] * gy[slot s of the pole].

        The 12 slots of a pole share the same 12 source pixels, so the rows of ``pole_src``
        and ``pole_dst`` of a pole are read from its first slot.
        """
        b = tl.program_id(0)
        idx = tl.program_id(1)
        pole = idx // NSRC
        k = idx - pole * NSRC
        offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        in_area = H * W
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for s in tl.static_range(NSRC):
            slot = pole * NSRC + s
            d = tl.load(pole_dst_ptr + slot)
            f = d // (Hp * Wp)
            rem = d - f * (Hp * Wp)
            hp = rem // Wp
            wp = rem - hp * Wp
            gp = (
                gy_ptr
                + (b * FACES + f) * stride_gyn
                + hp * stride_gyh
                + wp * stride_gyw
                + offs_c * stride_gyc
            )
            wsk = tl.load(pole_w_ptr + slot * NSRC + k)
            acc += tl.load(gp, mask=mask_c, other=0).to(tl.float32) * wsk
        sk = tl.load(pole_src_ptr + pole * NSRC * NSRC + k)
        fk = sk // in_area
        hw = sk - fk * in_area
        hk = hw // W
        wk = hw - hk * W
        gxk = (
            gx_ptr
            + (b * FACES + fk) * stride_gxn
            + hk * stride_gxh
            + wk * stride_gxw
            + offs_c * stride_gxc
        )
        tl.atomic_add(gxk, acc.to(gx_ptr.dtype.element_ty), mask=mask_c)

    @triton.jit
    def _isolatitude_pad_fwd_kernel(
        x_ptr,
        y_ptr,
        index0_ptr,
        index1_ptr,
        pole_dst_ptr,
        pole_src_ptr,
        pole_w_ptr,
        B,
        C,
        H,
        W,
        Hp,
        Wp,
        nout,
        stride_xn,
        stride_xc,
        stride_xh,
        stride_xw,
        stride_yn,
        stride_yc,
        stride_yh,
        stride_yw,
        BLOCK_O: tl.constexpr,
        BLOCK_C: tl.constexpr,
        FACES: tl.constexpr,
        N_POLE: tl.constexpr,
        NSRC: tl.constexpr,
    ):
        if N_POLE > 0:
            if tl.program_id(1) < N_POLE:
                _pole_fwd_body(
                    x_ptr, y_ptr, pole_dst_ptr, pole_src_ptr, pole_w_ptr, C, H, W, Hp, Wp,
                    stride_xn, stride_xc, stride_xh, stride_xw, stride_yn, stride_yc, stride_yh, stride_yw,
                    BLOCK_C, FACES, NSRC,
                )
            else:
                _pad_fwd_body(
                    x_ptr, y_ptr, index0_ptr, index1_ptr, B, C, H, W, Hp, Wp, nout,
                    stride_xn, stride_xc, stride_xh, stride_xw, stride_yn, stride_yc, stride_yh, stride_yw,
                    BLOCK_O, BLOCK_C, FACES, N_POLE,
                )
        else:
            _pad_fwd_body(
                x_ptr, y_ptr, index0_ptr, index1_ptr, B, C, H, W, Hp, Wp, nout,
                stride_xn, stride_xc, stride_xh, stride_xw, stride_yn, stride_yc, stride_yh, stride_yw,
                BLOCK_O, BLOCK_C, FACES, N_POLE,
            )

    @triton.jit
    def _isolatitude_pad_bwd_kernel(
        gy_ptr,
        gx_ptr,
        index0_ptr,
        index1_ptr,
        pole_dst_ptr,
        pole_src_ptr,
        pole_w_ptr,
        B,
        C,
        H,
        W,
        Hp,
        Wp,
        nout,
        stride_gxn,
        stride_gxc,
        stride_gxh,
        stride_gxw,
        stride_gyn,
        stride_gyc,
        stride_gyh,
        stride_gyw,
        BLOCK_O: tl.constexpr,
        BLOCK_C: tl.constexpr,
        FACES: tl.constexpr,
        N_POLE: tl.constexpr,
        NSRC: tl.constexpr,
    ):
        if N_POLE > 0:
            if tl.program_id(1) < N_POLE:
                _pole_bwd_body(
                    gy_ptr, gx_ptr, pole_dst_ptr, pole_src_ptr, pole_w_ptr, C, H, W, Hp, Wp,
                    stride_gxn, stride_gxc, stride_gxh, stride_gxw, stride_gyn, stride_gyc, stride_gyh, stride_gyw,
                    BLOCK_C, FACES, NSRC,
                )
            else:
                _pad_bwd_body(
                    gy_ptr, gx_ptr, index0_ptr, index1_ptr, B, C, H, W, Hp, Wp, nout,
                    stride_gxn, stride_gxc, stride_gxh, stride_gxw, stride_gyn, stride_gyc, stride_gyh, stride_gyw,
                    BLOCK_O, BLOCK_C, FACES, N_POLE,
                )
        else:
            _pad_bwd_body(
                gy_ptr, gx_ptr, index0_ptr, index1_ptr, B, C, H, W, Hp, Wp, nout,
                stride_gxn, stride_gxc, stride_gxh, stride_gxw, stride_gyn, stride_gyc, stride_gyh, stride_gyw,
                BLOCK_O, BLOCK_C, FACES, N_POLE,
            )


def _block_c(channels: int) -> int:
    if channels <= 32:
        return 32
    if channels <= 64:
        return 64
    if channels <= 128:
        return 128
    return 256


def _launch_fwd(x, y, index0, index1, pole=None) -> None:
    BF, C, H, W = x.shape
    _, _, Hp, Wp = y.shape
    B = BF // _HPX_FACES
    nout = _HPX_FACES * Hp * Wp
    block_c = _block_c(C)
    block_o = int(_BLOCK_O)
    n_main = triton.cdiv(nout, block_o)
    pole_dst, pole_src, pole_w = pole if pole is not None else (index0, index0, index0)
    n_pole = pole_dst.numel() if pole is not None else 0
    grid = (B, n_pole + n_main, triton.cdiv(C, block_c))
    _isolatitude_pad_fwd_kernel[grid](
        x, y, index0, index1, pole_dst, pole_src, pole_w,
        B, C, H, W, Hp, Wp, nout,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_O=block_o, BLOCK_C=block_c, FACES=_HPX_FACES,
        N_POLE=n_pole, NSRC=pole_src.shape[1] if pole is not None else 1,
    )


def _launch_bwd(gy, gx, index0, index1, pole=None) -> None:
    BF, C, H, W = gx.shape
    _, _, Hp, Wp = gy.shape
    B = BF // _HPX_FACES
    nout = _HPX_FACES * Hp * Wp
    block_c = _block_c(C)
    block_o = int(_BLOCK_O)
    n_main = triton.cdiv(nout, block_o)
    pole_dst, pole_src, pole_w = pole if pole is not None else (index0, index0, index0)
    n_pole = pole_dst.numel() if pole is not None else 0
    grid = (B, n_pole + n_main, triton.cdiv(C, block_c))
    _isolatitude_pad_bwd_kernel[grid](
        gy, gx, index0, index1, pole_dst, pole_src, pole_w,
        B, C, H, W, Hp, Wp, nout,
        gx.stride(0), gx.stride(1), gx.stride(2), gx.stride(3),
        gy.stride(0), gy.stride(1), gy.stride(2), gy.stride(3),
        BLOCK_O=block_o, BLOCK_C=block_c, FACES=_HPX_FACES,
        N_POLE=n_pole, NSRC=pole_src.shape[1] if pole is not None else 1,
    )


class IsolatitudePadTritonFunction(th.autograd.Function):
    """Linear isolatitude pad with a fused gather fwd / scatter-add bwd.

    With pole tables (``pole_dst`` [S], ``pole_src`` [S, K], ``pole_w`` [S, K]) the S = 24
    halo slots in ``pole_dst`` are not gathered (``index0`` must be ``-1`` there); the extra
    programs write slot ``s`` as ``sum_k pole_w[s, k] * x[pole_src[s, k]]``. Slots 0-11 are
    one pole and 12-23 the other; the 12 slots of a pole share the same K = 12 sources.
    """

    @staticmethod
    def forward(
        ctx,
        data: th.Tensor,
        index0: th.Tensor,
        index1: th.Tensor,
        padding: int,
        enable_nhwc: bool,
        pole_dst: th.Tensor | None = None,
        pole_src: th.Tensor | None = None,
        pole_w: th.Tensor | None = None,
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
        ctx.has_pole = pole_dst is not None
        pole = (pole_dst, pole_src, pole_w) if ctx.has_pole else None
        _launch_fwd(data, out, idx0, idx1, pole)
        if ctx.has_pole:
            ctx.save_for_backward(idx0, idx1, pole_dst, pole_src, pole_w)
        else:
            ctx.save_for_backward(idx0, idx1)
        ctx.input_shape = (BF, C, H, W)
        ctx.input_channels_last = bool(
            enable_nhwc or data.is_contiguous(memory_format=th.channels_last)
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: th.Tensor):
        index0, index1, *pole = ctx.saved_tensors
        BF, C, H, W = ctx.input_shape
        mem = th.channels_last if ctx.input_channels_last else th.contiguous_format
        grad_data = th.empty(
            (BF, C, H, W),
            device=grad_output.device,
            dtype=grad_output.dtype,
            memory_format=mem,
        ).zero_()
        _launch_bwd(
            grad_output.contiguous(memory_format=mem), grad_data, index0, index1,
            tuple(pole) if ctx.has_pole else None,
        )
        return grad_data, None, None, None, None, None, None, None


def isolatitude_pad_triton(
    data: th.Tensor,
    index0: th.Tensor,
    index1: th.Tensor,
    padding: int,
    enable_nhwc: bool,
    pole_dst: th.Tensor | None = None,
    pole_src: th.Tensor | None = None,
    pole_w: th.Tensor | None = None,
) -> th.Tensor:
    """Apply fused isolatitude pad. ``index1`` uses ``-1`` where there is no second source.

    ``pole_*`` (all three or none) add the pole-correction slots; see
    :class:`IsolatitudePadTritonFunction`.
    """
    if not isolatitude_pad_cuda_available():
        raise RuntimeError("isolatitude_pad_triton requires Triton and CUDA")
    if data.ndim != 4:
        raise ValueError(f"expected [N*12, C, H, W], got {tuple(data.shape)}")
    if data.shape[0] % _HPX_FACES != 0:
        raise ValueError(
            f"Folded batch {data.shape[0]} is not divisible by {_HPX_FACES} HEALPix faces"
        )
    return IsolatitudePadTritonFunction.apply(
        data, index0, index1, int(padding), bool(enable_nhwc), pole_dst, pole_src, pole_w
    )
