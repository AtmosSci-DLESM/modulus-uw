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

"""Hard RecUNet constraint: spherical low-pass of selected prognostic residuals.

Applied after residual add as ``y := x + LP_{ℓ < ℓ_cut}(y − x)``. Filtering the
increment (not the full field) keeps the initial high-ℓ state (orography in
surface pressure) while stopping a random walk of unresolved Δy. Per-variable
cutoffs are compile-friendly buffers; SHT matches the FACE→RING path in
``healpix_loss``.
"""

from __future__ import annotations

import math

import torch

import earth2grid
from cuhpx import SHTCUDA, iSHTCUDA
from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder


class ResidualSpectralLowPassConstraint(torch.nn.Module):
    def __init__(
        self,
        cutoffs: dict[str, int],
        in_channels: list[str] | None = None,
        out_channels: list[str] | None = None,
        nside: int = 64,
        lmax: int | None = None,
        mmax: int | None = None,
    ):
        """
        Parameters
        ----------
        cutoffs: dict[str, int]
            Prognostic variable → exclusive spherical-harmonic cutoff. Modes
            with ``ℓ < cutoff`` are kept; ``ℓ >= cutoff`` of the residual is
            zeroed. Example: ``{"PRESsfc": 128}``.
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
            pred_idx.append(self.out_names.index(name))
            orig_idx.append(self.in_names.index(name))
            cutoff_list.append(cut)

        # Python tuples so torch.compile unrolls constant integer slices.
        self._pred_idx = tuple(pred_idx)
        self._orig_idx = tuple(orig_idx)
        self._cutoffs = tuple(cutoff_list)

        ell = torch.arange(self.lmax)
        # [n_sel, lmax, 1] — broadcast over m. Keep ℓ < cutoff.
        ell_mask = torch.stack([(ell < cut).to(torch.float32) for cut in self._cutoffs], dim=0)
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
            residual = self._select(prediction, self._pred_idx) - self._select(
                orig, self._orig_idx
            )
            filtered = self._select(orig, self._orig_idx) + self._lowpass_residual(residual)
            out = self._replace(prediction, self._pred_idx, filtered)
        return out.to(dtype=orig_dtype)
