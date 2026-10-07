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
HEALPix padding modules (earth2grid, karlbauer, isolatitude).

         HEALPix                              Face order                 3D array representation
                                                                            -----------------
--------------------------               //\\  //\\  //\\  //\\             |   |   |   |   |
|| 0  |  1  |  2  |  3  ||              //  \\//  \\//  \\//  \\            |0  |1  |2  |3  |
|\\  //\\  //\\  //\\  //|             /\\0 //\\1 //\\2 //\\3 //            -----------------
| \\//  \\//  \\//  \\// |            // \\//  \\//  \\//  \\//             |   |   |   |   |
|4//\\5 //\\6 //\\7 //\\4|            \\4//\\5 //\\6 //\\7 //\\             |4  |5  |6  |7  |
|//  \\//  \\//  \\//  \\|             \\/  \\//  \\//  \\//  \\            -----------------
|| 8  |  9  |  10 |  11  |              \\8 //\\9 //\\10//\\11//            |   |   |   |   |
--------------------------               \\//  \\//  \\//  \\//             |8  |9  |10 |11 |
                                                                            -----------------
                                    "\\" are top and bottom, whereas
                                    "//" are left and right borders

Details on the HEALPix can be found at https://iopscience.iop.org/article/10.1086/427976
"""

from __future__ import annotations

import logging
import math
import os

import torch as th

logger = logging.getLogger(__name__)

# True if ``from earth2grid.healpix import pad`` succeeded; ``HEALPixPaddingv2`` and
# ``HEALPixLayer(..., hpx_padding_mode='earth2grid')`` require this.
have_earth2grid = True
try:
    from earth2grid.healpix import pad as healpix_pad
except ImportError:
    logger.warning("Could not import pad from earth2grid.healpix.")
    have_earth2grid = False

try:
    from .isolatitude_pad_triton import (
        isolatitude_pad_cuda_available,
        isolatitude_pad_triton,
    )
except ImportError:
    isolatitude_pad_cuda_available = lambda: False  # noqa: E731
    isolatitude_pad_triton = None

_ENABLE_HEALPIXPAD_DEPRECATION_MSG = (
    "enable_healpixpad is deprecated; use hpx_padding_mode instead "
    "(e.g. hpx_padding_mode='earth2grid' for earth2grid CUDA padding, "
    "hpx_padding_mode='karlbauer' for the pure PyTorch HEALPixPadding implementation). "
    "To reproduce the same behavior as the legacy enable_healpixpad=True, "
    "set hpx_padding_mode='earth2grid' in the model config."
)


def warn_deprecated_enable_healpixpad(
    enable_healpixpad: bool | None, hpx_padding_mode: str | None = None
) -> str:
    """
    Resolve ``hpx_padding_mode`` for module construction, accounting for deprecated
    ``enable_healpixpad``.

    ``hpx_padding_mode`` uses ``None`` in signatures to mean \"omitted\" (e.g. Hydra
    default). Omitted values are treated as the implicit default ``earth2grid`` except
    when ``enable_healpixpad`` is set and ``hpx_padding_mode`` was omitted: then
    ``False`` maps to ``karlbauer`` (legacy behavior) and ``True`` to ``earth2grid``.

    If ``hpx_padding_mode`` was explicitly provided (non-``None``), it wins over
    ``enable_healpixpad`` after logging the deprecation warning.

    Parameters
    ----------
    enable_healpixpad : bool or None
        Deprecated flag from config or kwargs; ``None`` means not set.
    hpx_padding_mode : str or None
        Requested mode, or ``None`` if omitted (normalized to ``earth2grid`` before
        applying legacy mapping).

    Returns
    -------
    str
        Resolved padding mode to store and pass to child modules.
    """
    hpx_padding_mode_explicit = hpx_padding_mode is not None
    if hpx_padding_mode is None:
        hpx_padding_mode = "earth2grid"
    if enable_healpixpad is None:
        return hpx_padding_mode
    msg = _ENABLE_HEALPIXPAD_DEPRECATION_MSG
    msg = f"{msg} Current hpx_padding_mode={hpx_padding_mode!r}, enable_healpixpad={enable_healpixpad!r}."
    logger.warning(f"WARNING: {msg}")
    if hpx_padding_mode_explicit:
        return hpx_padding_mode
    if enable_healpixpad:
        return "earth2grid"
    return "karlbauer"


def pop_deprecated_enable_healpixpad_from_kwargs(kwargs: dict) -> bool | None:
    """Remove ``enable_healpixpad`` from ``kwargs`` if present; return its value for legacy resolution."""
    if "enable_healpixpad" in kwargs:
        return bool(kwargs.pop("enable_healpixpad"))
    return None


POLE_CORRECTION_MODE = "isolatitude_pole_correction"


def make_hpx_padding_layer(
    padding: int,
    hpx_padding_mode: str,
    enable_nhwc: bool,
    nside: int | None = None,
) -> th.nn.Module:
    """
    Construct the HEALPix padding submodule for a given mode.

    Parameters
    ----------
    padding : int
        Symmetric pad width on each face edge (``p >= 1``).
    hpx_padding_mode : str
        One of ``"earth2grid"``, ``"karlbauer"``, ``"isolatitude"``, or
        ``"isolatitude_pole_correction"`` (isolatitude with the pole halo slots
        refilled by :class:`HEALPixPaddingIsolatitude`'s ``pole_correction``).
    enable_nhwc : bool
        Passed to padding modules that support channels-last output.
    nside : int or None, optional
        Native face height/width. Required for the isolatitude modes; ignored otherwise.

    Returns
    -------
    torch.nn.Module
        ``HEALPixPaddingv2``, ``HEALPixPadding``, or ``HEALPixPaddingIsolatitude``.

    Raises
    ------
    ValueError
        Unknown mode, an isolatitude mode without ``nside``, ``isolatitude_pole_correction``
        with ``padding != 1``, or earth2grid when earth2grid is unavailable or CUDA is
        not available.
    """
    if hpx_padding_mode == "earth2grid":
        if not have_earth2grid or not th.cuda.is_available():
            raise ValueError(
                "hpx_padding_mode=earth2grid requires earth2grid import and CUDA "
                f"(have_earth2grid={have_earth2grid}, "
                f"th.cuda.is_available()={th.cuda.is_available()})."
            )
        return HEALPixPaddingv2(padding=padding, enable_nhwc=enable_nhwc)
    if hpx_padding_mode == "karlbauer":
        return HEALPixPadding(padding=padding, enable_nhwc=enable_nhwc)
    if hpx_padding_mode in ("isolatitude", POLE_CORRECTION_MODE):
        if nside is None:
            raise ValueError(
                f'hpx_padding_mode="{hpx_padding_mode}" requires nside (positive int, '
                "native face height/width) to build gather indices."
            )
        return HEALPixPaddingIsolatitude(
            padding=padding,
            nside=nside,
            enable_nhwc=enable_nhwc,
            pole_correction=hpx_padding_mode == POLE_CORRECTION_MODE,
        )
    raise ValueError(
        f"Unsupported hpx_padding_mode={hpx_padding_mode!r}; "
        "expected one of 'earth2grid', 'karlbauer', 'isolatitude', "
        "'isolatitude_pole_correction'."
    )


class HEALPixFoldFaces(th.nn.Module):
    """Merge the HEALPix face dimension into the batch dimension (NCHW layout)."""

    def __init__(self, enable_nhwc: bool = False):
        """
        Parameters
        ----------
        enable_nhwc: bool, optional
            Use nhwc format instead of nchw format
        """
        super().__init__()
        self.enable_nhwc = enable_nhwc

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """
        Reshape ``[B, F, C, H, W]`` to ``[B * F, C, H, W]``.

        Parameters
        ----------
        tensor : torch.Tensor
            Five-dimensional tensor with ``F`` HEALPix faces (typically ``F == 12``).

        Returns
        -------
        torch.Tensor
            Four-dimensional tensor with faces stacked on the batch axis. If
            ``enable_nhwc`` is True, result uses channels-last memory format.
        """
        N, F, C, H, W = tensor.shape
        tensor = th.reshape(tensor, shape=(N * F, C, H, W))

        if self.enable_nhwc:
            tensor = tensor.to(memory_format=th.channels_last)

        return tensor


class HEALPixUnfoldFaces(th.nn.Module):
    """Split the batch axis so the HEALPix face index is its own dimension."""

    def __init__(self, num_faces: int = 12, enable_nhwc: bool = False):
        """
        Parameters
        ----------
        num_faces: int, optional
            The number of faces on the grid, default 12
        enable_nhwc: bool, optional
            If nhwc format is being used, default False
        """
        super().__init__()
        self.num_faces = num_faces
        self.enable_nhwc = enable_nhwc

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """
        Reshape ``[B * F, C, H, W]`` to ``[B, F, C, H, W]``.

        Parameters
        ----------
        tensor : torch.Tensor
            Folded layout produced by ``HEALPixFoldFaces`` or equivalent; leading size
            must be divisible by ``num_faces``.

        Returns
        -------
        torch.Tensor
            Five-dimensional tensor with explicit face dimension ``F``.
        """
        NF, C, H, W = tensor.shape
        tensor = th.reshape(tensor, shape=(-1, self.num_faces, C, H, W))

        return tensor


class HEALPixPaddingv2(th.nn.Module):
    """
    Padding layer for data on a HEALPix sphere. This version uses a faster method to calculate the padding.
    The requirements for using this layer are as follows:
    - The last three dimensions are (face=12, height, width)
    - The first four indices in the faces dimension [0, 1, 2, 3] are the faces on the northern hemisphere
    - The second four indices in the faces dimension [4, 5, 6, 7] are the faces on the equator
    - The last four indices in the faces dimension [8, 9, 10, 11] are the faces on the southern hemisphere

    Orientation and arrangement of the HEALPix faces are outlined above.

    TODO: Missing library to use this class. Need to see if we can get it, if not needs to be removed
    """

    def __init__(self, padding: int, enable_nhwc: bool = False):  # pragma: no cover
        """
        Parameters
        ----------
        padding : int
            The padding size
        enable_nhwc : bool, optional
            Whether to use channels-last memory format.
        """
        super().__init__()
        self.enable_nhwc = enable_nhwc
        self.unfold = HEALPixUnfoldFaces(num_faces=12, enable_nhwc=self.enable_nhwc)
        self.fold = HEALPixFoldFaces(enable_nhwc=self.enable_nhwc)
        self.padding = lambda x: healpix_pad(x, padding=padding)

    def forward(self, x):  # pragma: no cover
        """
        Pad each face consistently with its according neighbors in the HEALPix (see ordering and neighborhoods above).
        Assumes the Tensor is folded

        Parmaters
        ---------
        data: torch.Tensor
            The input tensor of shape [..., F, H, W] where each face is to be padded in its HPX context

        Returns
        -------
        torch.Tensor
            The padded tensor where each face's height and width are increased by 2*p
        """

        x = self.unfold(x)
        xp = self.padding(x)
        xp = self.fold(xp)

        if self.enable_nhwc:
            xp = xp.to(memory_format=th.channels_last)

        return xp


class HEALPixPadding(th.nn.Module):
    """
    Karlbauer et al. (2024) HEALPix padding.

    The requirements for using this layer are as follows:
    - The last three dimensions are (face=12, height, width)
    - The first four indices in the faces dimension [0, 1, 2, 3] are the faces on the northern hemisphere
    - The second four indices in the faces dimension [4, 5, 6, 7] are the faces on the equator
    - The last four indices in the faces dimension [8, 9, 10, 11] are the faces on the southern hemisphere

    Orientation and arrangement of the HEALPix faces are outlined in the module docstring.
    """

    def __init__(self, padding: int, enable_nhwc: bool = False):
        """
        Parameters
        ----------
        padding : int
            Symmetric pad width ``p >= 1`` on each face edge.
        enable_nhwc : bool, optional
            Match channels-last usage in the surrounding encoder/decoder.
        """
        super().__init__()
        self.p = padding
        self.d = [-2, -1]
        self.enable_nhwc = enable_nhwc
        if not isinstance(padding, int) or padding < 1:
            raise ValueError(
                f"invalid value for 'padding', expected int > 0 but got {padding}"
            )

        self.fold = HEALPixFoldFaces(enable_nhwc=self.enable_nhwc)
        self.unfold = HEALPixUnfoldFaces(num_faces=12, enable_nhwc=self.enable_nhwc)

    def forward(self, data: th.Tensor) -> th.Tensor:
        """
        Pad each face using values from neighboring faces (Karlbauer et al. scheme).

        Parameters
        ----------
        data : torch.Tensor
            Folded layout ``[N * 12, C, H, W]``. Internally reshaped to 12 faces per
            batch item before stitching.

        Returns
        -------
        torch.Tensor
            Folded layout ``[N * 12, C, H + 2*p, W + 2*p]`` where ``p`` is ``padding``.
        """
        # [N*12, C, H, W] -> [N, 12, C, H, W]
        data = self.unfold(data)

        # Extract the twelve faces (as views of the original tensors)
        f00, f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11 = [
            th.squeeze(x, dim=1)
            for x in th.split(tensor=data, split_size_or_sections=1, dim=1)
        ]

        # Assemble the four padded faces on the northern hemisphere
        p00 = self.pn(
            c=f00, t=f01, tl=f02, lft=f03, bl=f03, b=f04, br=f08, rgt=f05, tr=f01
        )
        p01 = self.pn(
            c=f01, t=f02, tl=f03, lft=f00, bl=f00, b=f05, br=f09, rgt=f06, tr=f02
        )
        p02 = self.pn(
            c=f02, t=f03, tl=f00, lft=f01, bl=f01, b=f06, br=f10, rgt=f07, tr=f03
        )
        p03 = self.pn(
            c=f03, t=f00, tl=f01, lft=f02, bl=f02, b=f07, br=f11, rgt=f04, tr=f00
        )

        # Assemble the four padded faces on the equator
        p04 = self.pe(
            c=f04,
            t=f00,
            tl=self.tl(f00, f03),
            lft=f03,
            bl=f07,
            b=f11,
            br=self.br(f11, f08),
            rgt=f08,
            tr=f05,
        )
        p05 = self.pe(
            c=f05,
            t=f01,
            tl=self.tl(f01, f00),
            lft=f00,
            bl=f04,
            b=f08,
            br=self.br(f08, f09),
            rgt=f09,
            tr=f06,
        )
        p06 = self.pe(
            c=f06,
            t=f02,
            tl=self.tl(f02, f01),
            lft=f01,
            bl=f05,
            b=f09,
            br=self.br(f09, f10),
            rgt=f10,
            tr=f07,
        )
        p07 = self.pe(
            c=f07,
            t=f03,
            tl=self.tl(f03, f02),
            lft=f02,
            bl=f06,
            b=f10,
            br=self.br(f10, f11),
            rgt=f11,
            tr=f04,
        )

        # Assemble the four padded faces on the southern hemisphere
        p08 = self.ps(
            c=f08, t=f05, tl=f00, lft=f04, bl=f11, b=f11, br=f10, rgt=f09, tr=f09
        )
        p09 = self.ps(
            c=f09, t=f06, tl=f01, lft=f05, bl=f08, b=f08, br=f11, rgt=f10, tr=f10
        )
        p10 = self.ps(
            c=f10, t=f07, tl=f02, lft=f06, bl=f09, b=f09, br=f08, rgt=f11, tr=f11
        )
        p11 = self.ps(
            c=f11, t=f04, tl=f03, lft=f07, bl=f10, b=f10, br=f09, rgt=f08, tr=f08
        )

        res = th.stack(
            (p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11), dim=1
        )

        # [N, 12, C, H', W'] -> [N*12, C, H', W']
        res = self.fold(res)

        if self.enable_nhwc:
            res = res.to(memory_format=th.channels_last)

        return res

    def pn(
        self,
        c: th.Tensor,
        t: th.Tensor,
        tl: th.Tensor,
        lft: th.Tensor,
        bl: th.Tensor,
        b: th.Tensor,
        br: th.Tensor,
        rgt: th.Tensor,
        tr: th.Tensor,
    ) -> th.Tensor:
        """
        Applies padding to a northern hemisphere face c under consideration of
        its given neighbors according to the strategy of Karlbauer et al. (2024)

        Parameters
        ----------
        c: torch.Tensor
            The central face and tensor that is subject for padding
        t: torch.Tensor
            The top neighboring face tensor
        tl: torch.Tensor
            The top left neighboring face tensor
        lft: torch.Tensor
            The left neighboring face tensor
        bl: torch.Tensor
            The bottom left neighboring face tensor
        b: torch.Tensor
            The bottom neighboring face tensor
        br: torch.Tensor
            The bottom right neighboring face tensor
        rgt: torch.Tensor
            The right neighboring face tensor
        tr: torch.Tensor
            The top right neighboring face  tensor

        Returns
        -------
        torch.Tensor
            Padded face tensor ``c`` with border strips from neighbors.
        """
        p = self.p
        d = self.d  # rotation plane: last two spatial dims of each face

        # Vertical extent: top row from ``t`` (rotated), bottom from ``b``
        c = th.cat((t.rot90(1, d)[..., -p:, :], c, b[..., :p, :]), dim=-2)

        # Horizontal strips: corners from ``tl`` / ``bl`` / ``tr`` / ``br``, edges from ``lft`` / ``rgt``
        left = th.cat(
            (
                tl.rot90(2, d)[..., -p:, -p:],
                lft.rot90(-1, d)[..., -p:],
                bl[..., :p, -p:],
            ),
            dim=-2,
        )
        right = th.cat((tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]), dim=-2)

        return th.cat((left, c, right), dim=-1)

    def pe(
        self,
        c: th.Tensor,
        t: th.Tensor,
        tl: th.Tensor,
        lft: th.Tensor,
        bl: th.Tensor,
        b: th.Tensor,
        br: th.Tensor,
        rgt: th.Tensor,
        tr: th.Tensor,
    ) -> th.Tensor:
        """
        Applies padding to an equatorial face c under consideration of its given neighbors.

        Parameters
        ----------
        c: torch.Tensor
            The central face and tensor that is subject for padding
        t: torch.Tensor
            The top neighboring face tensor
        tl: torch.Tensor
            The top left neighboring face tensor
        lft: torch.Tensor
            The left neighboring face tensor
        bl: torch.Tensor
            The bottom left neighboring face tensor
        b: torch.Tensor
            The bottom neighboring face tensor
        br: torch.Tensor
            The bottom right neighboring face tensor
        rgt: torch.Tensor
            The right neighboring face tensor
        tr: torch.Tensor
            The top right neighboring face  tensor

        Returns
        -------
        torch.Tensor
            Padded equatorial face ``c``.
        """
        p = self.p

        c = th.cat((t[..., -p:, :], c, b[..., :p, :]), dim=-2)
        left = th.cat((tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = th.cat((tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]), dim=-2)

        return th.cat((left, c, right), dim=-1)

    def ps(
        self,
        c: th.Tensor,
        t: th.Tensor,
        tl: th.Tensor,
        lft: th.Tensor,
        bl: th.Tensor,
        b: th.Tensor,
        br: th.Tensor,
        rgt: th.Tensor,
        tr: th.Tensor,
    ) -> th.Tensor:
        """
        Applies padding to a southern hemisphere face c under consideration of
        its given neighbors according to the strategy of Karlbauer et al. (2024).

        Parameters
        ----------
        c: torch.Tensor
            The central face and tensor that is subject for padding
        t: torch.Tensor
            The top neighboring face tensor
        tl: torch.Tensor
            The top left neighboring face tensor
        lft: torch.Tensor
            The left neighboring face tensor
        bl: torch.Tensor
            The bottom left neighboring face tensor
        b: torch.Tensor
            The bottom neighboring face tensor
        br: torch.Tensor
            The bottom right neighboring face tensor
        rgt: torch.Tensor
            The right neighboring face tensor
        tr: torch.Tensor
            The top right neighboring face  tensor

        Returns
        -------
        torch.Tensor
            Padded southern-hemisphere face ``c``.
        """
        p = self.p
        d = self.d

        c = th.cat((t[..., -p:, :], c, b.rot90(1, d)[..., :p, :]), dim=-2)
        left = th.cat((tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = th.cat(
            (tr[..., -p:, :p], rgt.rot90(-1, d)[..., :p], br.rot90(2, d)[..., :p, :p]),
            dim=-2,
        )

        return th.cat((left, c, right), dim=-1)

    def tl(self, top: th.Tensor, lft: th.Tensor) -> th.Tensor:
        """
        Assembles the top left corner of a center face according to the strategy
        of Karlbauer et al. (2024) in the cases where no according top left face
        is defined on the HPX grid.

        Parameters
        ----------
        top: torch.Tensor
            The face above the center face
        lft: torch.Tensor
            The face left of the center face

        Returns
        -------
        torch.Tensor
            ``p x p`` top-left corner block for the equatorial ``tl`` slot.
        """
        # Preallocated buffer for the synthetic corner (avoids per-call allocation).
        ret = th.zeros_like(top)[..., : self.p, : self.p]

        # Apex of the corner wedge (average of adjacent edge samples)
        ret[..., -1, -1] = 0.5 * top[..., -1, 0] + 0.5 * lft[..., 0, -1]

        for i in range(1, self.p):
            ret[..., -i - 1, -i:] = top[
                ..., -i - 1, :i
            ]  # Filling top right above main diagonal
            ret[..., -i:, -i - 1] = lft[
                ..., :i, -i - 1
            ]  # Filling bottom left below main diagonal
            ret[..., -i - 1, -i - 1] = (
                0.5 * top[..., -i - 1, 0] + 0.5 * lft[..., 0, -i - 1]
            )  # Diagonal

        return ret

    def br(self, b: th.Tensor, r: th.Tensor) -> th.Tensor:
        """
        Assembles the bottom right corner of a center face according to the
        strategy of Karlbauer et al. (2024) in the cases where no according
        bottom right face is defined on the HPX grid.

        Parameters
        ----------
        b: torch.Tensor
            The face below the center face
        r: torch.Tensor
            The face right of the center face

        Returns
        -------
        torch.Tensor
            ``p x p`` bottom-right corner block for the equatorial ``br`` slot.
        """
        ret = th.zeros_like(b)[..., : self.p, : self.p]

        ret[..., 0, 0] = 0.5 * b[..., 0, -1] + 0.5 * r[..., -1, 0]

        # Remaining points
        for i in range(1, self.p):
            ret[..., :i, i] = r[..., -i:, i]  # Filling top right above main diagonal
            ret[..., i, :i] = b[..., i, -i:]  # Filling bottom left below main diagonal
            ret[..., i, i] = 0.5 * b[..., i, -1] + 0.5 * r[..., -1, i]  # Diagonal

        return ret


def build_pole_correction_tables(nside: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Slots and weights of the isolatitude pole correction at ``padding == 1``.

    Returns ``(dst, src, w)``: ``dst`` [24] int32, the flat index of each halo slot in
    the padded ``[12, nside + 2, nside + 2]`` face stack; ``src`` [24, 12] int32, the
    flat index in the unpadded ``[12, nside, nside]`` stack of the 12 pixels it is
    built from (4 on ring 1, then the 8 on ring 2); ``w`` [24, 12] float32, the weights.

    Slots are ordered per pole (north faces 0-3, then south faces 8-11), per face: the
    two ring-0 slots (weights of ``P``), then the corner (weights of ``C = 2P - own``).

    Geometry used, for a polar face ``f`` of the north pole (pole pixel at array corner
    (0, 0), ring ``k = i + j + 1``): longitude of pixel (i, j) is
    ``45 + 90 f + 45 (j - i) / k`` degrees. The south pole is the same with pixels
    ``(n-1-i, n-1-j)`` and ``45 + 90 (f - 8) - 45 (j - i) / k``.
    """
    n = int(nside)
    hp = n + 2
    deg = math.pi / 180.0

    def ring_pixels(north: bool):
        """For each of the 4 polar faces: list of (flat index, longitude deg, ring)."""
        out = []
        for q in range(4):
            face = q if north else 8 + q
            pix = []
            for k in (1, 2):
                for i in range(k):
                    j = k - 1 - i
                    if north:
                        row, col, lon = i, j, 45 + 90 * q + 45.0 * (j - i) / k
                    else:
                        row, col, lon = n - 1 - i, n - 1 - j, 45 + 90 * q - 45.0 * (j - i) / k
                    pix.append((face * n * n + row * n + col, lon, k))
            out.append(pix)
        return out

    dst, src, wts = [], [], []
    for north in (True, False):
        faces = ring_pixels(north)
        pole_pixels = [[p for p in pix if p[2] == 1] for pix in faces]
        for q in range(4):
            face = q if north else 8 + q
            lam0 = (45 + 90 * q) * deg
            # pixel order: ring 1 (one per face), then ring 2 (two per face)
            ring1 = [pole_pixels[g][0] for g in range(4)]
            ring2 = [p for g in range(4) for p in faces[g] if p[2] == 2]
            order = ring1 + ring2
            wP = []
            for flat_idx, lon, k in order:
                c = math.cos(lon * deg - lam0)
                wP.append(0.5 + c if k == 1 else -0.125 - 0.25 * c)
            own = faces[q][0][0]
            wC = [2.0 * w for w in wP]
            for t, (flat_idx, _, _) in enumerate(order):
                if flat_idx == own:
                    wC[t] -= 1.0
            if north:
                slots = [(0, 1), (1, 0), (0, 0)]
            else:
                slots = [(hp - 2, hp - 1), (hp - 1, hp - 2), (hp - 1, hp - 1)]
            for t, (r, c_) in enumerate(slots):
                dst.append(face * hp * hp + r * hp + c_)
                src.append([o[0] for o in order])
                wts.append(wC if t == 2 else wP)
    return (
        th.tensor(dst, dtype=th.int32),
        th.tensor(src, dtype=th.int32),
        th.tensor(wts, dtype=th.float64).to(th.float32),
    )


class HEALPixPaddingIsolatitude(th.nn.Module):
    """
    Isolatitude HEALPix padding via **precomputed gather** indices.

    Indices are derived once from ``isolatitude_pad_folded``. On CUDA a fused
    Triton gather (forward) / scatter-add (backward) writes the padded tensor
    in-place in NCHW or NHWC so the training path does not permute, clone, then
    copy to channels-last. CPU (and ``PHYSICSNEMO_ISOLATITUDE_PAD=aten``) keep
    the original ATen gather implementation.

    **Pole correction** (``pole_correction=True``, ``padding == 1``). Plain isolatitude
    padding is exact at the poles except for three halo slots per polar face:
    the two ring-0 slots next to the pole pixel hold the neighbouring face's pole
    pixel (90 degrees of longitude away), and the corner holds the pixel across the
    pole, whose stored east/north frame is rotated by 180 degrees relative to the
    centre pixel (so a wind component arrives with the wrong sign). The corrected
    mode refills those 3 slots on each of the 4 polar faces of each pole (24 slots)
    with a fixed linear rule that is exact to first order for scalars and for wind
    components alike, in the centre pixel's own east/north frame.

    Rings around the pole are numbered ``k = 1, 2, ...`` outward (ring ``k`` has
    ``4k`` equally spaced pixels). For a polar face whose pole pixel ("own") sits at
    longitude ``lam0``, let ``m0_k`` be the mean of ring ``k`` and
    ``m1_k = (2 / N_k) * sum_j x_j * cos(lam_j - lam0)`` its *wave-1 value at lam0*:
    the amplitude, read along the face's own meridian, of the single-wave
    (one high and one low per circle) part of the ring. Near the pole, as functions
    of colatitude, a smooth scalar has ``m0`` flat and ``m1`` growing linearly from
    zero; a smooth wind component (in each pixel's local east/north frame) has
    ``m0`` growing linearly and ``m1`` constant. Extrapolating both harmonics to the
    pole with a straight line through rings 1 and 2 therefore gives the pole value
    of a scalar and the pole wind in the face's frame with the same weights::

        P = (2 m0_1 - m0_2) + (2 m1_1 - m1_2)    -> both ring-0 slots
        C = 2 P - own                            -> the across-pole corner

    ``C`` continues the face's own meridian straight through the pole, so the
    across pixel and its flipped frame are not used. Both are fixed weighted sums
    of the same 12 pixels (4 on ring 1, 8 on ring 2), with weights summing to 1:
    constants, zonal uniformity, the equatorial reflection and 90 degree rotation
    are preserved exactly. Every other halo pixel is unchanged.
    """

    def __init__(
        self,
        padding: int,
        nside: int,
        enable_nhwc: bool = False,
        pole_correction: bool = False,
    ):
        """
        Parameters
        ----------
        padding : int
            Pad width ``p >= 1``.
        nside : int
            Native face height/width ``H`` (square faces). Gather indices are built for
            this size at init; ``forward`` rejects other ``H``.
        enable_nhwc : bool, optional
            Channels-last output when True.
        pole_correction : bool, optional
            Refill the 24 polar halo slots as described in the class docstring.
            Requires ``padding == 1`` and ``nside >= 2``.
        """
        super().__init__()
        self.p = padding
        self.enable_nhwc = enable_nhwc
        self.pole_correction = bool(pole_correction)
        if not isinstance(nside, int) or nside < 1:
            raise ValueError(
                f"nside must be a positive int, got {nside!r}"
            )
        self._nside = nside
        if not isinstance(padding, int) or padding < 1:
            raise ValueError(
                f"invalid value for 'padding', expected int > 0 but got {padding}"
            )
        if self.pole_correction and (padding != 1 or nside < 2):
            raise ValueError(
                "the pole correction needs padding == 1 and nside >= 2, got "
                f"padding={padding}, nside={nside}"
            )
        idx, valid = build_isolatitude_gather_index(padding, nside)
        self.register_buffer("_index0", idx[0], persistent=False) # always 1
        # Dense second source; -1 means "no blend" (used by the fused CUDA kernel).
        self.register_buffer("_index1_or_neg1", idx[1].to(dtype=th.int32), persistent=False)
        self.register_buffer("_index1", idx[1].clamp_min(0), persistent=False)

        # Boolean mask for the second gather source
        v1_bool = valid[1]

        # Active positions in the 2nd gather source are extremely sparse, 
        # get the indices of the active positions
        pos_v1 = th.nonzero(v1_bool, as_tuple=False).squeeze(1).to(dtype=th.long)
        self.register_buffer("_pos_v1", pos_v1, persistent=False)
        self.register_buffer(
            "_index1_pos_v1",
            self._index1.index_select(0, pos_v1) if pos_v1.numel() > 0 else self._index1[:0],
            persistent=False,
        )

        if self.pole_correction:
            dst, src, w = build_pole_correction_tables(nside)
            # The 12 slots of a pole are combinations of the same 12 pixels (the fused kernel relies on it).
            src_p = src.view(2, 12, 12)
            if not bool((src_p == src_p[:, :1]).all()):
                raise RuntimeError("pole correction slots of one pole must share their source pixels")
            self.register_buffer("_pole_dst", dst, persistent=False)
            self.register_buffer("_pole_src", src, persistent=False)
            self.register_buffer("_pole_w", w, persistent=False)
            # The fused kernel skips outputs whose first source is -1; the pole kernel owns them.
            idx0_tri = self._index0.clone()
            idx0_tri[dst.long()] = -1
            self.register_buffer("_index0_tri", idx0_tri, persistent=False)

    def forward(self, data: th.Tensor) -> th.Tensor:
        """
        Gather-based isolatitude pad on folded input.

        Parameters
        ----------
        data : torch.Tensor
            ``[N * 12, C, H, W]`` with ``H == W``.

        Returns
        -------
        torch.Tensor
            ``[N * 12, C, H + 2*p, W + 2*p]``; numerically aligned with the reference
            module for the same ``padding`` and ``H``.
        """

        BF, C, H, W = data.shape
        F = 12
        B = BF // F

        if H != W:
            raise ValueError("HEALPix faces must be square (H == W)")
        if H != self._nside:
            raise ValueError(
                f"HEALPixPaddingIsolatitude expected face size H={self._nside} "
                f"(from init), but input has H={H}. Make sure that nside was set correctly "
                f"in the model config."
            )
        if BF % F != 0:
            raise ValueError(
                f"Folded batch {BF} is not divisible by {F} HEALPix faces"
            )

        # Fused gather/scatter keeps NHWC (or NCHW) without permute+clone+copy.
        # CPU, missing Triton, or PHYSICSNEMO_ISOLATITUDE_PAD=aten use ATen gather.
        use_fused = (
            data.is_cuda
            and isolatitude_pad_triton is not None
            and isolatitude_pad_cuda_available()
            and os.environ.get("PHYSICSNEMO_ISOLATITUDE_PAD", "triton") != "aten"
        )
        if use_fused:
            if self.pole_correction:
                return isolatitude_pad_triton(
                    data,
                    self._index0_tri,
                    self._index1_or_neg1,
                    self.p,
                    self.enable_nhwc,
                    self._pole_dst,
                    self._pole_src,
                    self._pole_w,
                )
            return isolatitude_pad_triton(
                data,
                self._index0,
                self._index1_or_neg1,
                self.p,
                self.enable_nhwc,
            )

        x = data.reshape(B, F, C, H, W)
        flat = x.permute(0, 2, 1, 3, 4).reshape(B, C, F * H * W)

        # Gather-based isolatitude pad.
        # valid[0] is always 1, so output is:
        #   out = g0                   (where valid1==0)
        #   out = 0.5 * (g0 + g1)      (where valid1==1)
        # and valid1==1 positions are extremely sparse.
        i0 = self._index0.view(1, 1, -1).expand(B, C, -1)
        g0 = flat.gather(dim=2, index=i0.to(device=data.device).long())

        pos_v1 = self._pos_v1
        if pos_v1.numel() > 0:
            Nv1 = int(pos_v1.numel())
            g0_sub = g0.index_select(dim=2, index=pos_v1.to(device=data.device).long())

            i1_sub = self._index1_pos_v1.view(1, 1, Nv1).expand(B, C, Nv1)
            g1_sub = flat.gather(dim=2, index=i1_sub.to(device=data.device).long())

            # out[pos_v1] = 0.5 * (g0_sub + g1_sub)
            #             = g0_sub + 0.5 * (g1_sub - g0_sub)
            delta = (g1_sub - g0_sub) * 0.5
            out_flat = g0.clone()
            out_flat.index_add_(dim=2, index=pos_v1.to(device=data.device), source=delta)
        else:
            out_flat = g0

        if self.pole_correction:
            # Fixed weighted sum of 12 pixels per slot; half precision accumulates in fp32
            # like the fused kernel, fp32 and fp64 stay in their own dtype.
            acc = th.float32 if data.dtype in (th.float16, th.bfloat16) else data.dtype
            src = flat.gather(
                dim=2, index=self._pole_src.reshape(1, 1, -1).expand(B, C, -1).long()
            ).reshape(B, C, *self._pole_src.shape)
            fills = (src.to(acc) * self._pole_w.to(acc)).sum(-1).to(out_flat.dtype)
            out_flat = out_flat.index_copy(2, self._pole_dst.long(), fills)

        Hp, Wp = H + 2 * self.p, W + 2 * self.p
        out = out_flat.reshape(B, C, F, Hp, Wp).permute(0, 2, 1, 3, 4).reshape(
            BF, C, Hp, Wp
        )
        if self.enable_nhwc:
            out = out.to(memory_format=th.channels_last)

        return out


# ------------------------------------------------------------
# Isolatitude padding helper functions 
# ------------------------------------------------------------

def kth_diag_indices(n: int, k: int) -> tuple[th.Tensor, th.Tensor]:
    """
    Row and column indices for the ``k``-th diagonal of an ``n x n`` matrix.

    Parameters
    ----------
    n : int
        Matrix side length.
    k : int
        Diagonal offset: ``0`` main, ``> 0`` above, ``< 0`` below.

    Returns
    -------
    tuple of torch.Tensor
        ``(rows, cols)`` such that ``matrix[rows, cols]`` selects that diagonal.
    """
    rows, cols = th.arange(n), th.arange(n)
    if k < 0:
        return rows[-k:], cols[:k]
    if k > 0:
        return rows[:-k], cols[k:]
    return rows, cols


def _pn_isolat(
    p: int,
    d: tuple[int, int],
    c: th.Tensor,
    t: th.Tensor,
    tl: th.Tensor,
    lft: th.Tensor,
    bl: th.Tensor,
    b: th.Tensor,
    br: th.Tensor,
    rgt: th.Tensor,
    tr: th.Tensor,
) -> th.Tensor:
    """
    Isolatitude padding assembly for a **northern** hemisphere face (shared helper).

    Applies along-edge rolls to ``t`` and ``lft`` after rotation so isolatitude lines
    align across face boundaries, then concatenates border strips like ``HEALPixPadding.pn``.
    """
    t = t.rot90(1, dims=d)[..., -p:, :]
    for i in range(p):
        t[..., -i - 1, :] = th.roll(t[..., -i - 1, :], 2 * i + 1, dims=-1)
        t[..., -i - 1, : 2 * i + 1] = t[..., -i - 1, 2 * i + 1].unsqueeze(-1)

    lft = lft.rot90(-1, dims=d)[..., -p:]
    for i in range(p):
        lft[..., -i - 1] = th.roll(lft[..., -i - 1], 2 * i + 1, dims=-1)
        lft[..., : 2 * i + 1, -i - 1] = lft[..., 2 * i + 1, -i - 1].unsqueeze(-1)

    c = th.cat((t, c, b[..., :p, :]), dim=-2)
    left = th.cat((tl.rot90(2, d)[..., -p:, -p:], lft, bl[..., :p, -p:]), dim=-2)
    right = th.cat((tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]), dim=-2)
    return th.cat((left, c, right), dim=-1)


def _pe_isolat(
    p: int,
    c: th.Tensor,
    t: th.Tensor,
    tl: th.Tensor,
    lft: th.Tensor,
    bl: th.Tensor,
    b: th.Tensor,
    br: th.Tensor,
    rgt: th.Tensor,
    tr: th.Tensor,
) -> th.Tensor:
    """Isolatitude padding for an **equatorial** face (no extra rolls on top/bottom strips)."""
    c = th.cat((t[..., -p:, :], c, b[..., :p, :]), dim=-2)
    left = th.cat((tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]), dim=-2)
    right = th.cat((tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]), dim=-2)
    return th.cat((left, c, right), dim=-1)


def _ps_isolat(
    p: int,
    d: tuple[int, int],
    c: th.Tensor,
    t: th.Tensor,
    tl: th.Tensor,
    lft: th.Tensor,
    bl: th.Tensor,
    b: th.Tensor,
    br: th.Tensor,
    rgt: th.Tensor,
    tr: th.Tensor,
) -> th.Tensor:
    """
    Isolatitude padding assembly for a **southern** hemisphere face.

    Rolls ``b`` and ``rgt`` after rotation so bottom and right strips match isolatitude
    connectivity, analogous to ``HEALPixPadding.ps``.
    """
    b = b.rot90(1, d)[..., :p, :]
    for i in range(p):
        b[..., i, :] = th.roll(b[..., i, :], -(2 * i + 1), dims=-1)
        b[..., i, -(2 * i + 1) :] = b[..., i, -(2 * i + 1) - 1].unsqueeze(-1)

    rgt = rgt.rot90(-1, d)[..., :p]
    for i in range(p):
        rgt[..., i] = th.roll(rgt[..., i], -(2 * i + 1), dims=-1)
        rgt[..., -(2 * i + 1) :, i] = rgt[..., -(2 * i + 1) - 1, i].unsqueeze(-1)

    c = th.cat((t[..., -p:, :], c, b), dim=-2)
    left = th.cat((tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]), dim=-2)
    right = th.cat((tr[..., -p:, :p], rgt, br.rot90(2, d)[..., :p, :p]), dim=-2)
    return th.cat((left, c, right), dim=-1)


def _tl_isolat(p: int, d: tuple[int, int], top: th.Tensor, lft: th.Tensor) -> th.Tensor:
    """
    Synthesize the missing **top-left** equatorial corner under isolatitude rules.

    Fills a ``p x p`` block by averaging samples along diagonals from ``top`` and ``lft``,
    then rotates to match the edge orientation expected by ``_pe_isolat``.
    """
    ret = th.zeros_like(top)[..., :p, :p]
    n = 2 * p - 1
    if n + 1 > top.shape[-1]:
        raise ValueError(
            f"Padding {p} must not exceed half the face height/width {top.shape[-1]}"
        )
    diag_nums = range(p - 1, -p, -1)
    for i in range(n):
        fill_val = 0.5 * (top[..., -i - 2, 0] + lft[..., 0, -i - 2])
        diag_indices = kth_diag_indices(p, diag_nums[i])
        for j in range(len(diag_indices[0])):
            ret[..., diag_indices[0][j], diag_indices[1][j]] = fill_val
    return th.rot90(ret, k=-1, dims=d)


def _br_isolat(p: int, d: tuple[int, int], b: th.Tensor, r: th.Tensor) -> th.Tensor:
    """
    Synthesize the missing **bottom-right** equatorial corner under isolatitude rules.

    Same diagonal-averaging pattern as ``_tl_isolat``, mirrored for the bottom/right faces.
    """
    ret = th.zeros_like(b)[..., :p, :p]
    n = 2 * p - 1
    if n + 1 > b.shape[-1]:
        raise ValueError(
            f"Padding {p} must not exceed half the face height/width {b.shape[-1]}"
        )
    diag_nums = range(p - 1, -p, -1)
    for i in range(n):
        fill_val = 0.5 * (b[..., i + 1, -1] + r[..., -1, i + 1])
        diag_indices = kth_diag_indices(p, diag_nums[i])
        for j in range(len(diag_indices[0])):
            ret[..., diag_indices[0][j], diag_indices[1][j]] = fill_val
    return th.rot90(ret, k=1, dims=d)


def isolatitude_pad_folded(data: th.Tensor, p: int, enable_nhwc: bool) -> th.Tensor:
    """
    Apply isolatitude HEALPix padding on **folded** face data.

    Shared by ``HEALPixPaddingIsolatitudeReference.forward`` and by
    ``build_isolatitude_gather_index_cpu`` (via a two-channel identity probe).

    Parameters
    ----------
    data : torch.Tensor
        Shape ``[N * 12, C, H, W]`` with square faces ``H == W``.
    p : int
        Pad width per edge; output spatial size is ``H + 2 * p``.
    enable_nhwc : bool
        If True, return channels-last memory format.

    Returns
    -------
    torch.Tensor
        Shape ``[N * 12, C, H + 2*p, W + 2*p]``.
    """
    d = (-2, -1)
    nf, c, h, w = data.shape
    data = data.reshape(-1, 12, c, h, w)

    f00, f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11 = [
        th.squeeze(x, dim=1)
        for x in th.split(tensor=data, split_size_or_sections=1, dim=1)
    ]

    p00 = _pn_isolat(p, d, f00, f01, f02, f03, f03, f04, f08, f05, f01)
    p01 = _pn_isolat(p, d, f01, f02, f03, f00, f00, f05, f09, f06, f02)
    p02 = _pn_isolat(p, d, f02, f03, f00, f01, f01, f06, f10, f07, f03)
    p03 = _pn_isolat(p, d, f03, f00, f01, f02, f02, f07, f11, f04, f00)

    p04 = _pe_isolat(
        p,
        f04,
        f00,
        _tl_isolat(p, d, f00, f03),
        f03,
        f07,
        f11,
        _br_isolat(p, d, f11, f08),
        f08,
        f05,
    )
    p05 = _pe_isolat(
        p,
        f05,
        f01,
        _tl_isolat(p, d, f01, f00),
        f00,
        f04,
        f08,
        _br_isolat(p, d, f08, f09),
        f09,
        f06,
    )
    p06 = _pe_isolat(
        p,
        f06,
        f02,
        _tl_isolat(p, d, f02, f01),
        f01,
        f05,
        f09,
        _br_isolat(p, d, f09, f10),
        f10,
        f07,
    )
    p07 = _pe_isolat(
        p,
        f07,
        f03,
        _tl_isolat(p, d, f03, f02),
        f02,
        f06,
        f10,
        _br_isolat(p, d, f10, f11),
        f11,
        f04,
    )

    p08 = _ps_isolat(p, d, f08, f05, f00, f04, f11, f11, f10, f09, f09)
    p09 = _ps_isolat(p, d, f09, f06, f01, f05, f08, f08, f11, f10, f10)
    p10 = _ps_isolat(p, d, f10, f07, f02, f06, f09, f09, f08, f11, f11)
    p11 = _ps_isolat(p, d, f11, f04, f03, f07, f10, f10, f09, f08, f08)

    res = th.stack((p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11), dim=1)
    n, _, _, hpad, wpad = res.shape
    res = res.reshape(n * 12, c, hpad, wpad)
    if enable_nhwc:
        res = res.to(memory_format=th.channels_last)
    return res

def decode_two_channel_identity_to_linear_indices(
    y0: th.Tensor, y1: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
    """
    Recover two nonnegative integer indices from a redundant two-channel encoding.

    For integers ``a, b``, the probe uses ``y0 = (a + b) / 2`` and
    ``y1 = (a**2 + b**2) / 2`` so that after padding, pairs ``(a, b)`` can be read
    back as source linear indices (used to build gather maps).

    Parameters
    ----------
    y0, y1 : torch.Tensor
        Same shape, floating-point recovered values from the padded probe.

    Returns
    -------
    tuple of torch.Tensor
        ``(a, b)`` as ``int64``, same shape as ``y0``.
    """
    s = 2.0 * y0
    ss = 2.0 * y1
    ab = (s * s - ss) / 2.0
    disc = (s * s - 4.0 * ab).clamp_min(0.0)
    root = disc.sqrt()
    a = ((s + root) / 2.0).round().to(th.int64)
    b = ((s - root) / 2.0).round().to(th.int64)
    return a, b


def build_isolatitude_gather_index(
    p: int, H: int
) -> tuple[th.Tensor, th.Tensor]:
    """
    Precompute gather indices that reproduce ``isolatitude_pad_folded`` on flat data.

    Runs the reference padding on a tensor of identity pairs per pixel, then decodes
    which input linear indices contribute to each output pixel.

    Parameters
    ----------
    p : int
        Padding width (must be ``>= 1`` and compatible with face size; see ``_tl_isolat``).
    H : int
        Native face height/width before padding.

    Returns
    -------
    index : torch.Tensor
        Shape ``[2, 12 * Hpad * Hpad]`` with ``Hpad = H + 2*p``. Row 0: first source
        index into flattened ``[12*H*H]`` input; row 1: second source or ``-1`` if
        only one source applies.
    valid : torch.Tensor
        Shape ``[2, N]``, boolean mask of which gather slots are active (second row
        False when both channels collapsed to the same index).
    """
    Hpad = H + 2 * p
    ids = th.arange(12 * H * H, dtype=th.float64).reshape(12, H, H)
    ch0 = ids
    ch1 = ids * ids
    x_folded = th.stack((ch0, ch1), dim=1).unsqueeze(0).reshape(12, 2, H, H)

    y_folded = isolatitude_pad_folded(x_folded, p, False)
    y = y_folded.reshape(1, 12, 2, Hpad, Hpad)
    y0 = y[:, :, 0].reshape(-1)
    y1 = y[:, :, 1].reshape(-1)
    a_flat, b_flat = decode_two_channel_identity_to_linear_indices(y0, y1)
    n_out = a_flat.numel()
    index = th.empty(2, n_out, dtype=th.int)
    index[0] = a_flat
    index[1] = b_flat
    valid = th.ones(2, n_out, dtype=th.bool)
    same = a_flat == b_flat
    index[1] = th.where(same, th.tensor(-1, dtype=th.int), index[1])
    valid[1] = ~same
    return index, valid
