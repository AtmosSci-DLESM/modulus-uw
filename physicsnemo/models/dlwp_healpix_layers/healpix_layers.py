# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HEALPix convolution / interpolation wrapper.

Builds a small ``Sequential`` that optionally prepends a HEALPix-aware padding module,
then the user-supplied base layer (e.g. ``Conv2d``). Inputs are face tensors with
12 HEALPix faces; see ``healpix_paddings`` for face ordering and padding modes.
"""

import torch as th
import logging

logger = logging.getLogger(__name__)

from .healpix_paddings import (
    make_hpx_padding_layer,
    pop_deprecated_enable_healpixpad_from_kwargs,
    warn_deprecated_enable_healpixpad,
)


class HEALPixLayer(th.nn.Module):
    """
    Apply a base ``torch.nn.Module`` on data laid out as HEALPix faces.

    Expected layout includes 12 HEALPix faces, typically ``[N, 12, C, H, W]`` 
    (any leading batch dimensions are allowed). When the base layer is a 
    convolution with ``kernel_size > 1`` or an interpolation layer, native
    ``padding`` is disabled for convolutions and a HEALPix padding module is 
    inserted so boundary values come from the correct neighboring faces.
    """

    def __init__(
        self,
        layer,
        hpx_padding_mode=None,
        nside: int | None = None,
        compile_padding: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        layer : type or torch.nn.Module
            Layer class (e.g. ``torch.nn.Conv2d``) or module; must match the
            detection logic for convolution vs interpolation vs other.
        hpx_padding_mode : str, optional
            Which padding implementation to use (``None`` means omitted; default ``earth2grid``):
            - ``"earth2grid"`` — ``earth2grid.healpix.pad`` (default).
            - ``"karlbauer"`` — Karlbauer et al. (2024) face stitching, same result as earth2grid but slower.
            - ``"isolatitude"`` — alternate padding scheme which preserves isolatitude signals.
        nside : int or None, optional
            Native resolution of each HEALPix face (height = width). Required when
            ``hpx_padding_mode=="isolatitude"``.
        compile_padding : bool, optional
            Whether to wrap isolatitude padding in ``_CompilePaddingWrapper``. Only
            supported when ``hpx_padding_mode="isolatitude"``.
        **kwargs
            Forwarded to ``layer`` after removing ``enable_nhwc`` and deprecated
            ``enable_healpixpad`` (e.g. ``in_channels``, ``out_channels``, ``kernel_size``,
            ``dilation``, ``enable_nhwc``). If ``nside`` or ``compile_padding`` appears
            here (e.g. Hydra), it is consumed and overrides the corresponding argument.
        """
        super().__init__()
        layers = []

        legacy_enable_healpixpad = pop_deprecated_enable_healpixpad_from_kwargs(kwargs)
        hpx_padding_mode = warn_deprecated_enable_healpixpad(
            legacy_enable_healpixpad, hpx_padding_mode
        )

        if "nside" in kwargs:
            _ns = kwargs.pop("nside")
            nside = int(_ns) if _ns is not None else None
        if "compile_padding" in kwargs:
            compile_padding = bool(kwargs.pop("compile_padding"))

        if "enable_nhwc" in kwargs:
            enable_nhwc = kwargs["enable_nhwc"]
            del kwargs["enable_nhwc"]
        else:
            enable_nhwc = False

        kernel_size = 3 if "kernel_size" not in kwargs else kwargs["kernel_size"]
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = ((kernel_size - 1) // 2) * dilation

        # Define a HEALPixPadding layer if padding is necessary
        if padding > 0:
            # Disable native padding for conv layers
            if layer.__bases__[0] is th.nn.modules.conv._ConvNd:
                kwargs["padding"] = 0
            padding_layer = make_hpx_padding_layer(
                padding=padding,
                hpx_padding_mode=hpx_padding_mode,
                enable_nhwc=enable_nhwc,
                nside=nside,
            )
            if compile_padding:
                padding_layer = th.compile(padding_layer)
            layers.append(padding_layer)

        layers.append(layer(**kwargs))
        self.layers = th.nn.Sequential(*layers)

        if enable_nhwc:
            self.layers = self.layers.to(memory_format=th.channels_last)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Run padding (if configured) and the wrapped layer.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B*F, C, H, W).

        Returns
        -------
        torch.Tensor
            Output of the composed ``Sequential`` of shape (B*F, C', H', W').
        """
        return self.layers(x)


# Modules whose forward draws from the RNG. Gradient checkpointing recomputes the
# forward during backward, so anything listed here produces a *different* random
# number on recompute unless the RNG state is restored.
_RNG_CONSUMER_TYPES = (
    th.nn.Dropout,
    th.nn.Dropout1d,
    th.nn.Dropout2d,
    th.nn.Dropout3d,
    th.nn.AlphaDropout,
    th.nn.FeatureAlphaDropout,
)

def check_for_rng_consumers(module: th.nn.Module, where: str) -> bool:
    """Refuse to checkpoint a module whose forward consumes RNG.

    The checkpoint calls in the encoder and decoder pass ``preserve_rng_state=False``
    because the default (``True``) calls ``CUDAGeneratorImpl::current_seed``, which is
    not capturable and either hard-fails under ``graph_mode: train_eval`` or suffers
    a large performance penalty.

    With no RNG consumer in the region that is exact -- recompute is bitwise identical.
    With one, the recomputed forward draws a *different* mask than the forward whose
    output was used, so the gradients are wrong. Measured with
    ``Dropout2d(p=0.5)``: max relative error 0.909 on the input gradient and 0.843 on
    the weight gradient, against a bitwise-exact match when the state is preserved.

    This cannot be fixed by saving and restoring the state around the checkpoint call.
    The cost would be negligible (~3.3 us host-side, ~0.4 us via the graph-safe
    generator), but every snapshot API is blocked inside a CUDA graph capture:
    ``get_rng_state()`` raises on ``current_seed`` and ``clone_state()`` raises on
    ``clone_impl``. ``graphsafe_get_state()`` is capturable but returns a live handle
    rather than a snapshot, so restoring it is a no-op. CUDA graphs can *advance* RNG
    across replays, not *rewind* it within a capture, and rewinding is exactly what
    checkpoint recompute needs.

    So the we check for RNG consumers in the module and enable preserve_rng_state accordingly.
    We suffer the performance penalty of preserving the RNG state only if there are RNG consumers.

    Parameters
    ----------
    module: th.nn.Module
        The module that would be wrapped in ``checkpoint``.
    where: str
        Human-readable location, used in the warning message.

    Returns
    ------
    bool
        True if ``module`` contains any RNG-consuming submodule, False otherwise.
    """
    offenders = [
        f"{name or '<root>'} ({type(m).__name__}, p={getattr(m, 'p', '?')})"
        for name, m in module.named_modules()
        if isinstance(m, _RNG_CONSUMER_TYPES) and getattr(m, "p", 0.0) > 0.0
    ]
    if offenders:
        logger.warning(f"Module {module.__class__.__name__} contains RNG-consuming submodules: {offenders} " \
                        "Enabling preserve_rng_state, this will cause a performance penalty during activation checkpointing.")
        return True
    return False