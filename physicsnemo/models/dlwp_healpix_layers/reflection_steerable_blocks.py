# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ReflectionSteerable HEALPix blocks (config-swappable with standard blocks).

Used when ``reflection_equivariance_mode: structural``. Each block sets
``reflection_steerable = True`` so the encoder/decoder can pass ``sin_lat_gate``
and keep even|odd channel layout. Drop-in Hydra replacements for
``SymmetricConvNeXtBlock`` / ``ConvGRUBlock`` / ``BasicConvBlock`` /
``SmoothedInterpolateConv``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch as th
import torch.nn as nn

from .healpix_blocks import HEALPixLayer, SmoothedInterpolate
from .healpix_paddings import make_hpx_padding_layer, warn_deprecated_enable_healpixpad
from .normalization import ConditionalLayerNorm
from .reflection_ops import bank_sizes, merge_banks, split_banks
from physicsnemo.models.layers.activations import Tanh

from .reflection_steerable_conv import (
    ParitySplitActivation,
    ReflectionSteerableConv1x1,
    ReflectionSteerableConv2d,
    require_reflection_steerable_tanh_activation,
)


def parity_skip_concat(x: th.Tensor, skip: th.Tensor, odd_fraction: float) -> th.Tensor:
    """Concatenate UNet skip so channel layout stays ``[even|odd]``."""
    ne_x, _ = bank_sizes(x.shape[1], odd_fraction)
    ne_s, _ = bank_sizes(skip.shape[1], odd_fraction)
    # Single cat of four slices (vs two cats + merge) cuts peak allocations.
    return th.cat(
        [x[:, :ne_x], skip[:, :ne_s], x[:, ne_x:], skip[:, ne_s:]],
        dim=1,
    )


class _PaddedReflectionSteerableConv(nn.Module):
    """HEALPix pad + ReflectionSteerableConv2d (kernel_size > 1)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        odd_fraction: float = 0.25,
        hpx_padding_mode: str = "isolatitude",
        compile_padding: bool = False,
        nside: int | None = None,
        enable_nhwc: bool = False,
        in_even: int | None = None,
        out_even: int | None = None,
    ):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        layers = []
        if padding > 0:
            pad = make_hpx_padding_layer(
                padding=padding,
                hpx_padding_mode=hpx_padding_mode,
                enable_nhwc=enable_nhwc,
                nside=nside,
            )
            if compile_padding:
                pad = th.compile(pad)
            layers.append(pad)
        layers.append(
            ReflectionSteerableConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                odd_fraction=odd_fraction,
                dilation=dilation,
                in_even=in_even,
                out_even=out_even,
            )
        )
        self.net = nn.Sequential(*layers)
        self.needs_sin_lat = False

    def forward(self, x, sin_lat=None):
        return self.net(x)


class _ReflectionSteerableConv1x1Wrap(nn.Module):
    def __init__(self, *args, use_sin_lat_gate: bool = False, **kwargs):
        super().__init__()
        # Default: same-parity 1×1 (single cuDNN gemm). Cross-parity is carried by 3×3
        # odd kernels; optional sin_lat gating remains available if explicitly enabled.
        self.conv = ReflectionSteerableConv1x1(*args, use_sin_lat_gate=use_sin_lat_gate, **kwargs)
        self.needs_sin_lat = bool(use_sin_lat_gate)

    def forward(self, x, sin_lat=None):
        return self.conv(x, sin_lat=sin_lat)


class ReflectionSteerableSymmetricConvNeXtBlock(nn.Module):
    """SymmetricConvNeXt with ReflectionSteerable convolutions, unified tanh, and optional CLN.

    CLN placement matches ``SymmetricConvNeXtBlock`` (after the first three
    convolutions when ``conditional_layer_norm_once`` is false). Instantiated CLN
    modules receive ``n_even`` for the current channel bank so LayerNorm stays
    ρ-equivariant (per-bank norm, odd β forced to zero).
    """

    reflection_steerable = True

    def __init__(
        self,
        geometry_layer=None,  # unused; kept for Hydra signature compatibility
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        upscale_factor: int = 4,
        activation: nn.Module = None,
        enable_nhwc: bool = False,
        use_block_skip_connection: bool = True,
        hpx_padding_mode: str | None = None,
        compile_padding: bool = False,
        nside: int | None = None,
        dropout: float = 0.0,
        conditional_layer_norm=None,
        conditional_layer_norm_once: bool = False,
        enable_healpixpad: bool | None = None,
        odd_fraction: float = 0.25,
        in_even: int | None = None,
        out_even: int | None = None,
        **kwargs,
    ):
        super().__init__()
        hpx_padding_mode = warn_deprecated_enable_healpixpad(enable_healpixpad, hpx_padding_mode)
        if hpx_padding_mode is None:
            hpx_padding_mode = "isolatitude"
        self.odd_fraction = odd_fraction
        self.use_block_skip_connection = use_block_skip_connection
        self.cln_enabled = conditional_layer_norm is not None

        if in_even is None:
            in_even, _ = bank_sizes(in_channels, odd_fraction)
        if out_even is None:
            out_even, _ = bank_sizes(out_channels, odd_fraction)
        lat_even, _ = bank_sizes(int(latent_channels), odd_fraction)
        up_ch = int(latent_channels * upscale_factor)
        up_even, _ = bank_sizes(up_ch, odd_fraction)

        # Unified tanh on both banks: odd bank needs an odd map for ρ-equivariance;
        # applying the same odd act to the even bank enables a single-kernel path.
        # Config may pass activation=None (default tanh) or Tanh; anything else errors.
        require_reflection_steerable_tanh_activation(
            activation, where="ReflectionSteerableSymmetricConvNeXtBlock", allow_none=True
        )

        def _act():
            return Tanh()

        def _cln(channel_depth: int, n_even_ch: int):
            # Pass n_even so CLN uses bank-separate LN and zeros odd β.
            return conditional_layer_norm(channel_depth=int(channel_depth), n_even=int(n_even_ch))

        if use_block_skip_connection:
            if in_channels == int(out_channels) and in_even == out_even:
                self.skip_module = None
            else:
                self.skip_module = _ReflectionSteerableConv1x1Wrap(
                    in_channels, out_channels, odd_fraction=odd_fraction, in_even=in_even, out_even=out_even
                )
        else:
            self.skip_module = None

        # Entry CLN (optional once-mode) uses the input bank layout.
        if conditional_layer_norm_once:
            if conditional_layer_norm is not None:
                self.entry_norm = _cln(in_channels, in_even)
            else:
                self.entry_norm = None
        else:
            self.entry_norm = None

        # Match SymmetricConvNeXtBlock: CLN after in→lat, lat→up, up→lat (not after final 3×3).
        layers: list[nn.Module] = [
            _PaddedReflectionSteerableConv(
                in_channels,
                int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                odd_fraction=odd_fraction,
                hpx_padding_mode=hpx_padding_mode,
                compile_padding=compile_padding,
                nside=nside,
                enable_nhwc=enable_nhwc,
                in_even=in_even,
                out_even=lat_even,
            ),
        ]
        if conditional_layer_norm is not None and not conditional_layer_norm_once:
            layers.append(_cln(latent_channels, lat_even))
        layers.append(ParitySplitActivation(int(latent_channels), odd_fraction, _act(), _act(), unified=True))

        layers.append(
            _ReflectionSteerableConv1x1Wrap(
                int(latent_channels),
                up_ch,
                odd_fraction=odd_fraction,
                in_even=lat_even,
                out_even=up_even,
            )
        )
        if conditional_layer_norm is not None and not conditional_layer_norm_once:
            layers.append(_cln(up_ch, up_even))
        layers.append(ParitySplitActivation(up_ch, odd_fraction, _act(), _act(), unified=True))

        layers.append(
            _ReflectionSteerableConv1x1Wrap(
                up_ch,
                int(latent_channels),
                odd_fraction=odd_fraction,
                in_even=up_even,
                out_even=lat_even,
            )
        )
        if conditional_layer_norm is not None and not conditional_layer_norm_once:
            layers.append(_cln(latent_channels, lat_even))
        layers.append(ParitySplitActivation(int(latent_channels), odd_fraction, _act(), _act(), unified=True))

        layers.extend(
            [
                _PaddedReflectionSteerableConv(
                    int(latent_channels),
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    odd_fraction=odd_fraction,
                    hpx_padding_mode=hpx_padding_mode,
                    compile_padding=compile_padding,
                    nside=nside,
                    enable_nhwc=enable_nhwc,
                    in_even=lat_even,
                    out_even=out_even,
                ),
                ParitySplitActivation(out_channels, odd_fraction, _act(), _act(), unified=True),
            ]
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, x, conditions_cln=None, sin_lat_gate=None):
        residual = 0
        if self.use_block_skip_connection:
            residual = x if self.skip_module is None else self.skip_module(x, sin_lat=sin_lat_gate)

        if self.entry_norm is not None:
            if conditions_cln is None:
                raise ValueError("Conditional inputs are required when entry CLN is enabled")
            x = self.entry_norm(x, conditions=conditions_cln)

        for layer in self.layers:
            if isinstance(layer, ConditionalLayerNorm):
                if conditions_cln is None:
                    raise ValueError("Conditional inputs are required for layers with CLN enabled")
                x = layer(x, conditions=conditions_cln)
            elif getattr(layer, "needs_sin_lat", False):
                x = layer(x, sin_lat=sin_lat_gate)
            else:
                x = layer(x)
        return x + residual


class ReflectionSteerableMulti_SymmetricConvNeXtBlock(nn.Module):
    """Serial stack of ReflectionSteerableSymmetricConvNeXtBlock."""

    reflection_steerable = True

    def __init__(
        self,
        geometry_layer=None,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        n_layers: int = 1,
        activation: nn.Module = None,
        enable_nhwc: bool = False,
        hpx_padding_mode: str | None = None,
        compile_padding: bool = False,
        nside: int | None = None,
        dropout: float = 0.0,
        conditional_layer_norm=None,
        conditional_layer_norm_once: bool = False,
        enable_healpixpad: bool | None = None,
        odd_fraction: float = 0.25,
        in_even: int | None = None,
        out_even: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.odd_fraction = odd_fraction
        self.cln_enabled = conditional_layer_norm is not None
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            curr_in = in_channels if i == 0 else out_channels
            curr_in_even = in_even if i == 0 else out_even
            self.blocks.append(
                ReflectionSteerableSymmetricConvNeXtBlock(
                    geometry_layer=geometry_layer,
                    in_channels=curr_in,
                    latent_channels=latent_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    upscale_factor=upscale_factor,
                    activation=activation,
                    enable_nhwc=enable_nhwc,
                    hpx_padding_mode=hpx_padding_mode,
                    compile_padding=compile_padding,
                    nside=nside,
                    dropout=0.0,
                    conditional_layer_norm=conditional_layer_norm,
                    conditional_layer_norm_once=conditional_layer_norm_once,
                    odd_fraction=odd_fraction,
                    in_even=curr_in_even,
                    out_even=out_even,
                )
            )

    def forward(self, x, conditions_cln=None, sin_lat_gate=None):
        for block in self.blocks:
            x = block(x, conditions_cln=conditions_cln, sin_lat_gate=sin_lat_gate)
        return x


class ReflectionSteerableConvGRUBlock(nn.Module):
    """ConvGRU with typed even|odd hidden state and ReflectionSteerable candidate."""

    reflection_steerable = True

    def __init__(
        self,
        geometry_layer=None,
        in_channels: int = 3,
        kernel_size: int = 1,
        enable_nhwc: bool = False,
        hpx_padding_mode: str | None = None,
        compile_padding: bool = False,
        nside: int | None = None,
        enable_healpixpad: bool | None = None,
        odd_fraction: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        self.channels = in_channels
        self.odd_fraction = odd_fraction
        self.n_even, self.n_odd = bank_sizes(in_channels, odd_fraction)
        # Gates from even banks only (ρ-invariant) via plain Conv2d — same cost as
        # baseline GRU gates on 2*n_even channels. Candidate is same-parity ReflectionSteerable.
        self.conv_gates = nn.Conv2d(2 * self.n_even, 2 * self.n_even, kernel_size=1, bias=True)
        self.conv_can = ReflectionSteerableConv1x1(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            odd_fraction=odd_fraction,
            in_even=2 * self.n_even,
            out_even=self.n_even,
            use_sin_lat_gate=False,
        )
        self.h = th.zeros(1, 1, 1, 1)

    def forward(self, inputs: th.Tensor, sin_lat_gate: Optional[th.Tensor] = None) -> th.Tensor:
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)

        xe, xo = split_banks(inputs, self.n_even)
        he, ho = split_banks(self.h, self.n_even)

        # sin_lat_gate unused (API kept for encoder wiring).
        gates = self.conv_gates(th.cat([xe, he], dim=1))
        reset_e, update_e = th.sigmoid(gates[:, : self.n_even]), th.sigmoid(gates[:, self.n_even :])
        if self.n_odd > 0:
            gate_scalar = update_e.mean(dim=1, keepdim=True)
            reset_scalar = reset_e.mean(dim=1, keepdim=True)
            reset_o = reset_scalar.expand(-1, self.n_odd, -1, -1)
            update_o = gate_scalar.expand(-1, self.n_odd, -1, -1)
        else:
            reset_o = reset_e[:, :0]
            update_o = update_e[:, :0]

        combined_can = merge_banks(th.cat([xe, reset_e * he], dim=1), th.cat([xo, reset_o * ho], dim=1))
        cnm = th.tanh(self.conv_can(combined_can))

        update_gate = merge_banks(update_e, update_o)
        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next
        return inputs + h_next

    def reset(self):
        self.h = th.zeros_like(self.h)


class ReflectionSteerableBasicConvBlock(nn.Module):
    """Output / basic stack of ReflectionSteerable convs."""

    reflection_steerable = True

    def __init__(
        self,
        geometry_layer=None,
        in_channels: int = 3,
        out_channels: int = 1,
        kernel_size: int = 1,
        dilation: int = 1,
        n_layers: int = 1,
        latent_channels: int = None,
        activation: nn.Module = None,
        enable_nhwc: bool = False,
        hpx_padding_mode: str | None = None,
        compile_padding: bool = False,
        nside: int | None = None,
        enable_healpixpad: bool | None = None,
        odd_fraction: float = 0.25,
        in_even: int | None = None,
        out_even: int | None = None,
        **kwargs,
    ):
        super().__init__()
        hpx_padding_mode = warn_deprecated_enable_healpixpad(enable_healpixpad, hpx_padding_mode)
        if hpx_padding_mode is None:
            hpx_padding_mode = "isolatitude"
        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)
        self.odd_fraction = odd_fraction
        layers = []
        ch_in = in_channels
        ie = in_even
        for n in range(n_layers):
            ch_out = out_channels if n == n_layers - 1 else latent_channels
            oe = out_even if n == n_layers - 1 else None
            if kernel_size == 1:
                layers.append(
                    _ReflectionSteerableConv1x1Wrap(
                        ch_in, ch_out, odd_fraction=odd_fraction, in_even=ie, out_even=oe
                    )
                )
            else:
                layers.append(
                    _PaddedReflectionSteerableConv(
                        ch_in,
                        ch_out,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        odd_fraction=odd_fraction,
                        hpx_padding_mode=hpx_padding_mode,
                        compile_padding=compile_padding,
                        nside=nside,
                        enable_nhwc=enable_nhwc,
                        in_even=ie,
                        out_even=oe,
                    )
                )
            if activation is not None and n < n_layers - 1:
                require_reflection_steerable_tanh_activation(
                    activation, where="ReflectionSteerableBasicConvBlock", allow_none=False
                )
                layers.append(
                    ParitySplitActivation(ch_out, odd_fraction, Tanh(), Tanh(), unified=True)
                )
            ch_in = ch_out
            ie = oe if oe is not None else bank_sizes(ch_out, odd_fraction)[0]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, sin_lat_gate=None):
        for layer in self.layers:
            if getattr(layer, "needs_sin_lat", False):
                x = layer(x, sin_lat=sin_lat_gate)
            elif isinstance(layer, ParitySplitActivation):
                x = layer(x)
            else:
                x = layer(x)
        return x


class ReflectionSteerableSmoothedInterpolateConv(nn.Module):
    """Z₂-equivariant ``SmoothedInterpolateConv`` for ReflectionSteerable decoders.

    The HPX four-point smoother is depthwise with an ``R``-symmetric stencil, so it
    preserves typed even|odd banks. The post-upsample map is ``ReflectionSteerableConv2d``.
    """

    reflection_steerable = True

    def __init__(
        self,
        geometry_layer=HEALPixLayer,
        in_channels: int = 3,
        out_channels: int = 3,
        kernel_size: int = 3,
        dilation: int = 1,
        scale_factor: int = 2,
        mode: str = "nearest",
        activation: nn.Module = None,
        odd_fraction: float = 0.25,
        enable_nhwc: bool = False,
        hpx_padding_mode: str | None = "isolatitude",
        compile_padding: bool = False,
        nside: int = 64,
        enable_healpixpad: bool | None = None,
        **kwargs,
    ):
        super().__init__()
        hpx_padding_mode = warn_deprecated_enable_healpixpad(enable_healpixpad, hpx_padding_mode)
        if dilation > 1:
            raise ValueError(
                f"dilation > 1 is not supported for parity hpx resize convolutions, got {dilation}"
            )
        self.odd_fraction = float(odd_fraction)
        trim_size = 1
        self.interp = geometry_layer(
            layer=SmoothedInterpolate,
            in_channels=in_channels,
            scale_factor=scale_factor,
            mode=mode,
            trim_size=trim_size,
            enable_nhwc=enable_nhwc,
            hpx_padding_mode=hpx_padding_mode,
            compile_padding=compile_padding,
            nside=nside,
        )
        in_even, _ = bank_sizes(in_channels, self.odd_fraction)
        out_even, _ = bank_sizes(out_channels, self.odd_fraction)
        self.conv = _PaddedReflectionSteerableConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            odd_fraction=self.odd_fraction,
            hpx_padding_mode=hpx_padding_mode,
            compile_padding=compile_padding,
            nside=nside * scale_factor,
            enable_nhwc=enable_nhwc,
            in_even=in_even,
            out_even=out_even,
        )
        self.act = None
        if activation is not None:
            require_reflection_steerable_tanh_activation(
                activation, where="ReflectionSteerableSmoothedInterpolateConv", allow_none=False
            )
            self.act = ParitySplitActivation(
                out_channels, self.odd_fraction, Tanh(), Tanh(), unified=True
            )

    def forward(self, x, sin_lat_gate=None) -> th.Tensor:
        x = self.interp(x)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x
