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

"""Hard RecUNet spherical low-pass constraints on selected prognostics.

``ResidualSpectralLowPassConstraint`` is applied after residual add as
``y := x + LP_{ℓ < ℓ_cut}(y − x)``. Filtering the increment (not the full field)
keeps the initial high-ℓ state (orography in surface pressure) while stopping a
random walk of unresolved Δy.

``InputSkipSpectralLowPassConstraint`` is the other split:
``y := LP_{ℓ < ℓ_cut}(x) + (y − x)``. The decoder still sees the unfiltered
state. High-ℓ of the carried state is dropped each step, so unpredictable
small-scale noise does not accumulate; high-ℓ in the output comes only from
this step's residual.

Per-variable cutoffs are compile-friendly buffers; SHT matches the FACE→RING
path in ``healpix_loss``.

``taper_ends`` replaces the brick wall with a raised cosine. ``cutoffs[name]``
is then the last degree kept in full, and ``taper_ends[name]`` is the first
degree set to zero. Degrees in between use ``½(1 + cos(π(ℓ − start)/(end − start)))``.
"""

from __future__ import annotations

import math

import torch

import earth2grid
from cuhpx import SHTCUDA, iSHTCUDA
from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder


def raised_cosine_window(ell: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Weight 1 through ``start``, 0 from ``end`` up, cosine in between.

    At the midpoint the weight is 1/2. ``ell`` is an integer degree tensor.
    """
    if end <= start:
        raise ValueError(f"taper end {end} must be greater than start {start}")
    width = float(end - start)
    t = (ell.to(torch.float32) - float(start)) / width
    window = 0.5 * (1.0 + torch.cos(t * math.pi))
    window = torch.where(ell <= start, torch.ones_like(window), window)
    window = torch.where(ell >= end, torch.zeros_like(window), window)
    return window


class ResidualSpectralLowPassConstraint(torch.nn.Module):
    def __init__(
        self,
        cutoffs: dict[str, int],
        in_channels: list[str] | None = None,
        out_channels: list[str] | None = None,
        nside: int = 64,
        lmax: int | None = None,
        mmax: int | None = None,
        taper_ends: dict[str, int] | None = None,
    ):
        """
        Parameters
        ----------
        cutoffs: dict[str, int]
            Prognostic variable → exclusive spherical-harmonic cutoff. Modes
            with ``ℓ < cutoff`` are kept; ``ℓ >= cutoff`` of the residual is
            zeroed.             Example: ``{"PRESsfc": 128}``. With ``taper_ends``, this is the last
            degree whose weight is 1, not a brick wall.
        taper_ends: dict[str, int], optional
            Prognostic → first degree whose weight is 0. Omitted variables keep
            the brick wall ``ℓ < cutoff``. The cosine is centered halfway
            between the cutoff and this end.
        in_channels: list[str]
            Prognostic / RecUNet input channel names. Residuals are taken from
            these channels of ``input``.
        out_channels: list[str], optional
            Full output channel names (prognostics + diagnostics). Defaults to
            ``in_channels``. Cutoff names must appear here and in ``in_channels``
            (diagnostics have no residual against ``orig_input``).
        nside: int
            HEALPix nside of the tensors passed to ``forward`` (decoder output).
        lmax, mmax: int, optional
            SHT bandwidth. Default ``3 * nside - 1`` (same as ``healpix_loss``).
        """
        super().__init__()
        if not cutoffs:
            raise ValueError("cutoffs must list at least one prognostic variable")
        if in_channels is None and out_channels is None:
            raise ValueError("in_channels or out_channels is required")

        self.in_names = list(in_channels) if in_channels is not None else list(out_channels)
        self.out_names = list(out_channels) if out_channels is not None else list(in_channels)
        self.nside = int(nside)
        if self.nside < 1 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError(f"nside must be a positive power of 2, got {nside}")
        self.lmax = int(lmax) if lmax is not None else 3 * self.nside - 1
        self.mmax = int(mmax) if mmax is not None else self.lmax
        if self.lmax < 1 or self.mmax < 1:
            raise ValueError(f"lmax/mmax must be positive, got lmax={self.lmax} mmax={self.mmax}")

        pred_idx: list[int] = []
        orig_idx: list[int] = []
        cutoff_list: list[int] = []
        taper_ends = {} if taper_ends is None else dict(taper_ends)
        unknown_taper = set(taper_ends) - set(cutoffs)
        if unknown_taper:
            raise ValueError(
                f"taper_ends {sorted(unknown_taper)} are not in cutoffs {sorted(cutoffs)}"
            )
        taper_list: list[int | None] = []
        for name, raw_cut in cutoffs.items():
            if name not in self.out_names:
                raise ValueError(
                    f"cutoff variable {name!r} is not in out_channels {self.out_names}"
                )
            if name not in self.in_names:
                raise ValueError(
                    f"cutoff variable {name!r} is diagnostic or missing from in_channels "
                    f"{self.in_names}; residual low-pass requires a prognostic"
                )
            cut = int(raw_cut)
            if cut < 1:
                raise ValueError(f"cutoff for {name!r} must be >= 1, got {raw_cut}")
            if cut > self.lmax:
                raise ValueError(
                    f"cutoff for {name!r} is {cut} but SHT lmax is {self.lmax}; "
                    "keep ℓ < cutoff so cutoff must be <= lmax"
                )
            end = taper_ends.get(name)
            if end is not None:
                end = int(end)
                if end <= cut:
                    raise ValueError(
                        f"taper end for {name!r} is {end} but cutoff start is {cut}"
                    )
                if end > self.lmax:
                    raise ValueError(
                        f"taper end for {name!r} is {end} but SHT lmax is {self.lmax}"
                    )
            pred_idx.append(self.out_names.index(name))
            orig_idx.append(self.in_names.index(name))
            cutoff_list.append(cut)
            taper_list.append(end)

        # Python tuples so torch.compile unrolls constant integer slices.
        self._pred_idx = tuple(pred_idx)
        self._orig_idx = tuple(orig_idx)
        self._cutoffs = tuple(cutoff_list)

        ell = torch.arange(self.lmax)
        # [n_sel, lmax, 1] — broadcast over m.
        masks = []
        for cut, end in zip(self._cutoffs, taper_list):
            if end is None:
                masks.append((ell < cut).to(torch.float32))
            else:
                masks.append(raised_cosine_window(ell, cut, end))
        ell_mask = torch.stack(masks, dim=0)
        self.register_buffer("ell_mask", ell_mask.unsqueeze(-1), persistent=False)

        src_grid = earth2grid.healpix.Grid(
            level=int(math.log2(self.nside)), pixel_order=HEALPIX_PAD_XY
        )
        tar_grid = earth2grid.healpix.Grid(
            level=int(math.log2(self.nside)), pixel_order=PixelOrder.RING
        )
        self.reorder_to_ring = earth2grid.get_regridder(src_grid, tar_grid).to(torch.float32)
        self.reorder_from_ring = earth2grid.get_regridder(tar_grid, src_grid).to(torch.float32)
        self.sht = SHTCUDA(
            nside=self.nside, lmax=self.lmax, mmax=self.mmax, quad_weights="ring"
        )
        self.isht = iSHTCUDA(
            nside=self.nside, lmax=self.lmax, mmax=self.mmax, quad_weights="ring"
        )

    def _to_alm(self, faces: torch.Tensor) -> torch.Tensor:
        """SHT of ``[B, F, T, C, H, W]`` → complex ``[B, T, C, lmax, mmax]``."""
        x = torch.movedim(faces, 1, -3)
        if x.shape[-3:] != (12, self.nside, self.nside):
            raise ValueError(
                f"expected 12 faces of nside={self.nside}, got spatial {tuple(x.shape[-3:])}"
            )
        x = x.reshape(*x.shape[:-3], -1)
        x = self.reorder_to_ring(x.contiguous())
        return self.sht(x)

    def _from_alm(self, alm: torch.Tensor) -> torch.Tensor:
        """Inverse SHT ``[B, T, C, lmax, mmax]`` → ``[B, F, T, C, H, W]``."""
        x = self.isht(alm)
        x = self.reorder_from_ring(x)
        x = x.reshape(*x.shape[:-1], 12, self.nside, self.nside)
        return torch.movedim(x, -3, 1)

    def _lowpass_residual(self, residual: torch.Tensor) -> torch.Tensor:
        alm = self._to_alm(residual)
        mask = self.ell_mask.to(device=alm.device, dtype=alm.real.dtype)
        return self._from_alm(alm * mask)

    def _select(self, tensor: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        return torch.cat([tensor[:, :, :, i : i + 1] for i in indices], dim=3)

    def _replace(self, tensor: torch.Tensor, indices: tuple[int, ...], values: torch.Tensor) -> torch.Tensor:
        out = tensor
        for j, i in enumerate(indices):
            out = torch.cat(
                [out[:, :, :, :i], values[:, :, :, j : j + 1], out[:, :, :, i + 1 :]],
                dim=3,
            )
        return out

    def _combine(self, orig_sel: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """``y = x + LP(y − x)``. High-ℓ of the carried state is unchanged."""
        return orig_sel + self._lowpass_residual(residual)

    def forward(self, prediction: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        prediction, input:
            ``[B, F, T, C, H, W]``. ``input`` is RecUNet ``orig_input`` (prognostic
            prefix). If time lengths differ, the last input time is used.
        """
        orig_dtype = prediction.dtype
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            orig = input.float()
            if orig.shape[2] != prediction.shape[2]:
                orig = orig[:, :, -1:]
            orig_sel = self._select(orig, self._orig_idx)
            residual = self._select(prediction, self._pred_idx) - orig_sel
            filtered = self._combine(orig_sel, residual)
            out = self._replace(prediction, self._pred_idx, filtered)
        return out.to(dtype=orig_dtype)


class InputSkipSpectralLowPassConstraint(ResidualSpectralLowPassConstraint):
    """Low-pass the residual-add skip, not the increment.

    RecUNet has already run on the unfiltered state. This rewrites selected
    channels as ``y := LP(x) + (y − x)``, which is the same as filtering ``x``
    immediately before the residual is added.
    """

    def _combine(self, orig_sel: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self._lowpass_residual(orig_sel) + residual
