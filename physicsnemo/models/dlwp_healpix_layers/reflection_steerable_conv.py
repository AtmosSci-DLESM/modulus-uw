# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ReflectionSteerable convolutions for Z₂ equatorial reflection equivariance.

**ReflectionSteerable** means kernels (and optional 1×1 maps) are constrained to
the even/odd subspace under the in-face reflection ``R``, so each layer
satisfies ``L(ρx) = ρ L(x)``. Used when ``reflection_equivariance_mode`` is
``structural`` (legacy alias: ``steerable``).

Performance notes
-----------------
Hot path is plain ``nn.Conv2d`` (same apply cost as unconstrained). Constraints
(even/odd 3×3 blocks; block-diagonal 1×1) are enforced by:
* init / ``eval()`` weight projection onto the parity subspace, and
* **gradient hooks** so Adam stays on-manifold (``R`` not in the forward graph).

Optional ``sin_lat`` cross-parity 1×1 remains available but is off by default;
spatial cross-parity still comes from odd 3×3 kernels.
"""

from __future__ import annotations

import math
from typing import Optional

import torch as th
import torch.nn.functional as F
from torch import nn

from .reflection_ops import (
    apply_R_kernel,
    bank_sizes,
    merge_banks,
    split_banks,
)


def _project_parity_conv_weight_(
    weight: th.Tensor,
    out_even: int,
    in_even: int,
) -> None:
    """In-place even/odd block projection on a full ``[O, I, kH, kW]`` kernel."""
    oe, ie = out_even, in_even
    oo, ii = weight.shape[0] - oe, weight.shape[1] - ie

    def _even_(block: th.Tensor) -> None:
        # K <- 0.5 * (K + R(K)); R(...) is evaluated before in-place update.
        block.add_(apply_R_kernel(block)).mul_(0.5)

    def _odd_(block: th.Tensor) -> None:
        # K <- 0.5 * (K - R(K))
        block.sub_(apply_R_kernel(block)).mul_(0.5)

    if oe and ie:
        _even_(weight[:oe, :ie])
    if oe and ii:
        _odd_(weight[:oe, ie:])
    if oo and ie:
        _odd_(weight[oe:, :ie])
    if oo and ii:
        _even_(weight[oe:, ie:])


def _project_block_diag_1x1_(weight: th.Tensor, out_even: int, in_even: int) -> None:
    """In-place: zero cross-parity blocks of a 1×1 weight (same-parity subspace)."""
    oe, ie = out_even, in_even
    if oe and weight.shape[1] > ie:
        weight[:oe, ie:] = 0
    if weight.shape[0] > oe and ie:
        weight[oe:, :ie] = 0


def _project_parity_conv_grad_(grad: th.Tensor, out_even: int, in_even: int) -> th.Tensor:
    """Project weight gradients onto the even/odd subspace (in-place)."""
    _project_parity_conv_weight_(grad, out_even, in_even)
    return grad


def _project_block_diag_1x1_grad_(grad: th.Tensor, out_even: int, in_even: int) -> th.Tensor:
    _project_block_diag_1x1_(grad, out_even, in_even)
    return grad


class ReflectionSteerableConv2d(nn.Module):
    """Local conv with even/odd channel banks and even/odd kernel constraints.

    Implemented as ``nn.Conv2d`` + projected weights so the hot path matches
    unconstrained convolution. Channel layout is ``[even | odd]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        odd_fraction: float = 0.25,
        bias: bool = True,
        dilation: int = 1,
        stride: int = 1,
        in_even: int | None = None,
        out_even: int | None = None,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.odd_fraction = odd_fraction
        self.dilation = dilation
        self.stride = stride
        if in_even is None:
            self.in_even, self.in_odd = bank_sizes(in_channels, odd_fraction)
        else:
            self.in_even, self.in_odd = in_even, in_channels - in_even
        if out_even is None:
            self.out_even, self.out_odd = bank_sizes(out_channels, odd_fraction)
        else:
            self.out_even, self.out_odd = out_even, out_channels - out_even
        if self.in_even < 0 or self.in_odd < 0 or self.out_even < 0 or self.out_odd < 0:
            raise ValueError("Invalid even/odd bank sizes")

        # Odd-bank bias is kept at 0 by projection; even bias lives in conv.bias[:out_even].
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
            bias=bias,
        )
        self.reset_parameters()
        # Projected GD on the linear even/odd subspace: project grads so Adam stays
        # on-manifold without reprojecting weights every forward (R is paid once/backward).
        self.conv.weight.register_hook(
            lambda g, oe=self.out_even, ie=self.in_even: _project_parity_conv_grad_(g, oe, ie)
        )
        if self.conv.bias is not None and self.out_odd > 0:
            oe = self.out_even

            def _bias_hook(g, oe=oe):
                g[oe:] = 0
                return g

            self.conv.bias.register_hook(_bias_hook)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.conv.bias is not None:
            fan_in = max(self.in_channels * self.kernel_size * self.kernel_size, 1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv.bias, -bound, bound)
        self._project_params_()

    def train(self, mode: bool = True):
        was_training = self.training
        out = super().train(mode)
        if was_training and not mode:
            self._project_params_()
        return out

    def _project_params_(self) -> None:
        with th.no_grad():
            _project_parity_conv_weight_(self.conv.weight, self.out_even, self.in_even)
            if self.conv.bias is not None and self.out_odd > 0:
                self.conv.bias[self.out_even :] = 0

    @property
    def bias_even(self) -> Optional[th.Tensor]:
        if self.conv.bias is None or self.out_even == 0:
            return None
        return self.conv.bias[: self.out_even]

    def materialize_weight(self) -> th.Tensor:
        """Return the on-manifold full weight (alias of ``conv.weight``)."""
        return self.conv.weight

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.conv(x)


class ReflectionSteerableConv1x1(nn.Module):
    """Pointwise ReflectionSteerable map.

    Default (``use_sin_lat_gate=False``): ``nn.Conv2d`` 1×1 projected to the
    block-diagonal same-parity subspace (hot path ≈ unconstrained 1×1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        odd_fraction: float = 0.25,
        bias: bool = True,
        use_sin_lat_gate: bool = False,
        in_even: int | None = None,
        out_even: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.odd_fraction = odd_fraction
        self.use_sin_lat_gate = use_sin_lat_gate
        if in_even is None:
            self.in_even, self.in_odd = bank_sizes(in_channels, odd_fraction)
        else:
            self.in_even, self.in_odd = in_even, in_channels - in_even
        if out_even is None:
            self.out_even, self.out_odd = bank_sizes(out_channels, odd_fraction)
        else:
            self.out_even, self.out_odd = out_even, out_channels - out_even

        if use_sin_lat_gate:
            # Slow expressivity path: separate bank gemms + sin_lat cross terms.
            self.conv = None
            self.W_ee = nn.Parameter(th.empty(self.out_even, self.in_even)) if self.out_even and self.in_even else None
            self.W_oo = nn.Parameter(th.empty(self.out_odd, self.in_odd)) if self.out_odd and self.in_odd else None
            self.W_eo = nn.Parameter(th.empty(self.out_even, self.in_odd)) if self.out_even and self.in_odd else None
            self.W_oe = nn.Parameter(th.empty(self.out_odd, self.in_even)) if self.out_odd and self.in_even else None
            if bias and self.out_even > 0:
                self.bias_even = nn.Parameter(th.empty(self.out_even))
            else:
                self.register_parameter("bias_even", None)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            self.W_ee = self.W_oo = self.W_eo = self.W_oe = None
            self.register_parameter("bias_even", None)

        self.reset_parameters()
        if self.conv is not None:
            self.conv.weight.register_hook(
                lambda g, oe=self.out_even, ie=self.in_even: _project_block_diag_1x1_grad_(
                    g, oe, ie
                )
            )
            if self.conv.bias is not None and self.out_odd > 0:
                oe = self.out_even

                def _bias_hook(g, oe=oe):
                    g[oe:] = 0
                    return g

                self.conv.bias.register_hook(_bias_hook)

    def reset_parameters(self) -> None:
        if self.conv is not None:
            nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
            if self.conv.bias is not None:
                fan_in = max(self.in_channels, 1)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.conv.bias, -bound, bound)
            self._project_params_()
        else:
            for w in (self.W_ee, self.W_oo, self.W_eo, self.W_oe):
                if w is not None:
                    nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.bias_even is not None:
                fan_in = max(self.in_channels, 1)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias_even, -bound, bound)

    def train(self, mode: bool = True):
        was_training = self.training
        out = super().train(mode)
        if was_training and not mode and self.conv is not None:
            self._project_params_()
        return out

    def _project_params_(self) -> None:
        if self.conv is None:
            return
        with th.no_grad():
            _project_block_diag_1x1_(self.conv.weight, self.out_even, self.in_even)
            if self.conv.bias is not None and self.out_odd > 0:
                self.conv.bias[self.out_even :] = 0

    def forward(self, x: th.Tensor, sin_lat: Optional[th.Tensor] = None) -> th.Tensor:
        if not self.use_sin_lat_gate:
            return self.conv(x)

        if sin_lat is None:
            raise ValueError("ReflectionSteerableConv1x1 with use_sin_lat_gate=True requires sin_lat")
        if sin_lat.shape[0] != x.shape[0]:
            raise ValueError(f"sin_lat batch {sin_lat.shape[0]} != x batch {x.shape[0]}")
        if sin_lat.shape[-2:] != x.shape[-2:]:
            raise ValueError(f"sin_lat spatial {sin_lat.shape[-2:]} != x spatial {x.shape[-2:]}")

        x_e, x_o = split_banks(x, self.in_even)
        y_e = x.new_zeros(x.shape[0], self.out_even, x.shape[2], x.shape[3])
        y_o = x.new_zeros(x.shape[0], self.out_odd, x.shape[2], x.shape[3])
        if self.W_ee is not None:
            y_e = y_e + F.conv2d(x_e, self.W_ee[:, :, None, None])
        if self.W_oo is not None:
            y_o = y_o + F.conv2d(x_o, self.W_oo[:, :, None, None])
        if self.W_eo is not None:
            y_e = y_e + F.conv2d(sin_lat * x_o, self.W_eo[:, :, None, None])
        if self.W_oe is not None:
            y_o = y_o + F.conv2d(sin_lat * x_e, self.W_oe[:, :, None, None])
        if self.bias_even is not None and y_e.shape[1] > 0:
            y_e = y_e + self.bias_even.view(1, -1, 1, 1)
        return merge_banks(y_e, y_o)


def require_reflection_steerable_tanh_activation(
    activation: Optional[nn.Module],
    *,
    where: str,
    allow_none: bool = True,
) -> Optional[nn.Module]:
    """Reject non-``tanh`` activations from Hydra/config for ReflectionSteerable blocks.

    Structural layers use a *unified* nonlinearity on even and odd banks. That
    function must be odd so the odd bank stays odd under ρ. This codebase
    standardizes on ``physicsnemo.models.layers.activations.Tanh`` (a
    ``torch.nn.Tanh`` subclass). ``None`` means "no activation" when ``allow_none``.
    """
    if activation is None:
        if allow_none:
            return None
        raise ValueError(
            f"{where}: activation is required and must be "
            "physicsnemo.models.layers.activations.Tanh"
        )
    if not isinstance(activation, nn.Tanh):
        raise TypeError(
            f"{where}: ReflectionSteerable blocks only allow tanh "
            f"(got {type(activation).__name__}). Set activation._target_ to "
            "physicsnemo.models.layers.activations.Tanh in the config."
        )
    return activation


class ParitySplitActivation(nn.Module):
    """Apply ``even_act`` on the even bank and ``odd_act`` on the odd bank.

    "Parity" here means even/odd channel typing, not the ReflectionSteerable
    product name. Production structural path uses unified ``nn.Tanh`` on both
    banks. Odd-bank equivariance needs an odd map; this module requires
    ``nn.Tanh`` unless ``allow_non_tanh=True`` (ablations only).

    When both banks share the same odd nonlinearity (``unified=True``), apply one
    kernel to all channels.
    """

    def __init__(
        self,
        n_channels: int,
        odd_fraction: float,
        even_act: nn.Module,
        odd_act: nn.Module,
        *,
        unified: bool | None = None,
        allow_non_tanh: bool = False,
    ):
        super().__init__()
        self.n_even, self.n_odd = bank_sizes(n_channels, odd_fraction)
        if not allow_non_tanh:
            if not isinstance(even_act, nn.Tanh) or not isinstance(odd_act, nn.Tanh):
                raise TypeError(
                    "ParitySplitActivation requires nn.Tanh on both banks "
                    f"(got even={type(even_act).__name__}, odd={type(odd_act).__name__}). "
                    "Pass allow_non_tanh=True only for explicit ablations."
                )
        self.even_act = even_act
        self.odd_act = odd_act
        if unified is None:
            unified = type(even_act) is type(odd_act) and not hasattr(even_act, "cap")
        self.unified = bool(unified)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.n_odd == 0:
            return self.even_act(x)
        if self.n_even == 0:
            return self.odd_act(x)
        if self.unified:
            return self.even_act(x)
        y = th.empty_like(x)
        y[:, : self.n_even] = self.even_act(x[:, : self.n_even])
        y[:, self.n_even :] = self.odd_act(x[:, self.n_even :])
        return y
