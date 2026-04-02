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
    HEALPixPadding,
    HEALPixPaddingIsolatitude,
    HEALPixPaddingv2,
    have_earth2grid,
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
        nside: int = 64,
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
        nside : int, optional
            Native resolution of each HEALPix face (height = width = ``nside``).
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
            nside = int(kwargs.pop("nside"))
        if "compile_padding" in kwargs:
            compile_padding = bool(kwargs.pop("compile_padding"))

        if "enable_nhwc" in kwargs:
            enable_nhwc = kwargs["enable_nhwc"]
            del kwargs["enable_nhwc"]
        else:
            enable_nhwc = False

        # Define a HEALPixPadding layer if the given layer is a convolution or
        # interpolation layer
        if layer.__bases__[0] is th.nn.modules.conv._ConvNd:
            layer_type = "conv"
        elif "Interpolate" in layer.__name__:
            layer_type = "interp"
        else:
            layer_type = "other"

        padding_layer = None

        if (
            (layer_type == "conv" and kwargs["kernel_size"] > 1)
            or (layer_type == "interp")
        ):
            if layer_type == "conv":
                # HEALPix padding replaces symmetric conv padding on each face.
                kwargs["padding"] = 0
            kernel_size = 3 if "kernel_size" not in kwargs else kwargs["kernel_size"]
            dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
            padding = ((kernel_size - 1) // 2) * dilation

            if hpx_padding_mode == "earth2grid":
                if (
                    have_earth2grid
                    and th.cuda.is_available()
                    and not enable_nhwc
                ):  # pragma: no cover
                    padding_layer = HEALPixPaddingv2(padding=padding)
                else:
                    raise ValueError(
                        "hpx_padding_mode=earth2grid requires earth2grid import, "
                        "CUDA, and enable_nhwc=False."
                    )
            elif hpx_padding_mode == "karlbauer":
                padding_layer = HEALPixPadding(padding=padding, enable_nhwc=enable_nhwc)
            elif hpx_padding_mode == "isolatitude":
                padding_layer =  HEALPixPaddingIsolatitude(
                    padding=padding,
                    nside=nside,
                    enable_nhwc=enable_nhwc,
                )
            else:
                raise ValueError(
                    f"Unsupported hpx_padding_mode={hpx_padding_mode!r}."
                )

        if padding_layer is not None:
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
