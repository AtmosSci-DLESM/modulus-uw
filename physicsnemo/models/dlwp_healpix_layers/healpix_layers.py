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

from .healpix_paddings import (
    HEALPixPadding,
    HEALPixPaddingIsolatitude,
    HEALPixPaddingIsolatitudeReference,
    HEALPixPaddingv2,
    have_earth2grid,
    pop_deprecated_enable_healpixpad_from_kwargs,
)


class _CompilePaddingWrapper(th.nn.Module):
    """
    A wrapper around HEALPix padding modules that ensures padding is always applied with a fixed dtype.

    This is useful for cases where the model may use mixed-precision training (e.g., float16/bfloat16),
    but the padding operation requires a specific dtype (like float32 or float16). The input tensor is 
    casted to the required dtype, the padding is applied, and the result is casted back to the original 
    dtype if necessary.

    Parameters
    ----------
    padding : int
        Amount of padding to apply to the spatial dimensions.
    healpix_face_size : int
        The size (height/width) of each HEALPix face.
    enable_nhwc : bool
        Whether to use channels-last (NHWC) format.
    pad_dtype : torch.dtype
        The dtype to use for the padding operation.
    compile_inner : bool, optional
        Whether to use torch.compile on the inner padding module (default: True).
    """

    def __init__(
        self,
        padding: int,
        healpix_face_size: int,
        enable_nhwc: bool,
        compile_inner: bool = True,
        fixed_pad_dtype: th.dtype = th.float32,
    ):
        super().__init__()
        inner = HEALPixPaddingIsolatitude(
            padding=padding,
            enable_nhwc=enable_nhwc,
            healpix_face_size=healpix_face_size,
        )
        self.inner = th.compile(inner) if compile_inner else inner
        self.fixed_pad_dtype = fixed_pad_dtype

    def forward(self, data: th.Tensor) -> th.Tensor:
        orig_dtype = data.dtype
        if data.dtype != self.fixed_pad_dtype:
            data = data.to(dtype=self.fixed_pad_dtype)
        out = self.inner(data)
        if out.dtype != orig_dtype:
            out = out.to(dtype=orig_dtype)
        return out


class HEALPixLayer(th.nn.Module):
    """
    Apply a base ``torch.nn.Module`` on data laid out as HEALPix faces.

    Expected layout includes 12 HEALPix faces, typically ``[N, 12, C, H, W]`` (any
    leading batch dimensions are allowed). When the base layer
    is a convolution with ``kernel_size > 1`` or an interpolation layer, native
    ``padding`` is disabled for convolutions and a HEALPix padding module is inserted
    so boundary values come from the correct neighboring faces.

    Notes
    -----
    Pass ``enable_nhwc`` in ``kwargs`` to use channels-last memory format for the
    padding path and the instantiated layer. The default ``earth2grid`` mode requires
    ``earth2grid``, a CUDA device, and ``enable_nhwc=False``. For CPU-only execution,
    set ``hpx_padding_mode`` to ``karlbauer`` or an isolatitude mode.
    """

    def __init__(
        self,
        layer,
        hpx_padding_mode="earth2grid",
        nside: int = 64,
        **kwargs,
    ):
        """
        Parameters
        ----------
        layer : type or torch.nn.Module
            Layer class (e.g. ``torch.nn.Conv2d``) or module; must match the
            detection logic for convolution vs interpolation vs other.
        hpx_padding_mode : str, optional
            Which padding implementation to use:

            - ``"earth2grid"`` — ``earth2grid.healpix.pad`` (CUDA, non-NHWC; default).
            - ``"karlbauer"`` — Karlbauer et al. (2024) face stitching.
            - ``"isolatitude_reference"`` — isolatitude rules, reference implementation.
            - ``"isolatitude"`` — same numerics as reference, gather-based forward.
        nside : int, optional
            Native resolution of each HEALPix face (height = width = ``nside``). Passed
            as ``healpix_face_size`` to ``HEALPixPaddingIsolatitude`` so gather indices
            are built at module init. Ignored for other padding modes.
        **kwargs
            Forwarded to ``layer`` after removing ``enable_nhwc`` and deprecated
            ``enable_healpixpad`` (e.g. ``in_channels``, ``out_channels``, ``kernel_size``,
            ``dilation``, ``enable_nhwc``). If ``nside`` appears here (e.g. Hydra), it is
            consumed and overrides the ``nside`` argument.
        """
        super().__init__()
        layers = []

        pop_deprecated_enable_healpixpad_from_kwargs(kwargs)

        if "nside" in kwargs:
            nside = int(kwargs.pop("nside"))

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
                    layers.append(
                        HEALPixPaddingv2(
                            padding=padding,
                        )
                    )
                else:
                    raise ValueError(
                        "hpx_padding_mode=earth2grid requires earth2grid import, "
                        "CUDA, and enable_nhwc=False."
                    )
            elif hpx_padding_mode == "karlbauer":
                layers.append(
                    HEALPixPadding(padding=padding, enable_nhwc=enable_nhwc)
                )
            elif hpx_padding_mode == "isolatitude_reference":
                layers.append(
                    HEALPixPaddingIsolatitudeReference(
                        padding=padding, enable_nhwc=enable_nhwc
                    )
                )
            elif hpx_padding_mode == "isolatitude":
                layers.append(
                    _CompilePaddingWrapper(
                        padding=padding,
                        healpix_face_size=nside,
                        enable_nhwc=enable_nhwc,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported hpx_padding_mode={hpx_padding_mode!r}."
                )

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
            Tensor whose face axis has size 12 (e.g. ``[N, 12, C, H, W]``).

        Returns
        -------
        torch.Tensor
            Output of the composed ``Sequential`` (same leading dimensions as ``x``,
            except where the inner layer changes channel or spatial size).
        """
        return self.layers(x)
