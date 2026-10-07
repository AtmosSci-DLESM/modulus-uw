# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Polar-cap corrections for the HEALPix down- and up-sampling layers.

Near a pole the rings around it (ring ``k`` has ``4k`` equally spaced pixels) carry
two things that a stride-2 blur or a face-wise bilinear lookup handles badly:

* the **ring mean** ``a0`` (wave 0), and
* the **wave-1 part** ``a1 cos(lam) + b1 sin(lam)``: the single-wave (one high, one
  low per circle) component of the ring.

As functions of colatitude, a smooth scalar has ``a0`` flat and wave 1 growing
linearly from zero; a smooth wind component (stored in each pixel's local east/north
frame) has ``a0`` growing linearly from zero and wave 1 constant (cross-polar flow).
Longitude-direction filters and lookups on rings only a few pixels wide cannot
follow either, so winds near the pole lose or invent their cross-polar part
(about 25 percent of it after a stride-2 blur, and 47 percent after the PCHIP upsample, on
the innermost ring). Treating waves 0 and 1 *linearly in colatitude* instead is exact
to first order for both kinds of field, so one rule serves scalars and winds:

* :class:`PolarCapDownCorrection`: after a stride-2 downsample, replace waves 0 and 1
  of coarse rings ``k <= K`` by ``1/4, 1/2, 1/4`` times those of fine rings
  ``2k-1, 2k, 2k+1``.
* :class:`PolarCapUpCorrection`: after a ring-mean PCHIP upsample, set waves 0 and 1
  of the innermost fine ring (poleward of every coarse ring) to
  ``1.5 * (coarse ring 1) - 0.5 * (coarse ring 2)``, and the wave-1 part of fine rings
  ``2..2K`` to the linear interpolation in colatitude between the coarse rings that
  bracket them. Higher waves and the wave-0 part of rings ``>= 2`` stay as the
  resampler made them.

Each correction is a fixed linear map on a few hundred pixels per pole that commutes
with the equatorial reflection, 90 degree rotation, and zonal uniformity, and leaves
constants unchanged. Applying it is one gather, one small matmul and one in-place write.
"""

from __future__ import annotations

import math

import numpy as np
import torch as th

POLE_CORRECTION_MODE = "isolatitude_pole_correction"
_MAX_DEPTH = 4


def cap_depth(coarse_nside: int) -> int:
    """Number of coarse rings ``K`` corrected at each pole: ``min(4, coarse_nside - 1)``."""
    return min(_MAX_DEPTH, int(coarse_nside) - 1)


def _ring(nside: int, north: bool, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat index ``face * n * n + i * n + j`` and longitude (rad) of the ``4k`` pixels of ring ``k``.

    North pole: faces 0-3 with the pole at array corner (0, 0), ring ``k = i + j + 1`` and
    longitude ``45 + 90 f + 45 (j - i) / k`` degrees. South pole: faces 8-11 with the pole at
    (n-1, n-1), pixel ``(n-1-i, n-1-j)`` and longitude ``45 + 90 (f - 8) - 45 (j - i) / k``.
    """
    n = int(nside)
    idx, lon = [], []
    for q in range(4):
        face = q if north else 8 + q
        for i in range(k):
            j = k - 1 - i
            if north:
                row, col, lam = i, j, 45.0 + 90.0 * q + 45.0 * (j - i) / k
            else:
                row, col, lam = n - 1 - i, n - 1 - j, 45.0 + 90.0 * q - 45.0 * (j - i) / k
            idx.append(face * n * n + row * n + col)
            lon.append(math.radians(lam))
    return np.asarray(idx, dtype=np.int64), np.asarray(lon, dtype=np.float64)


def _colatitude(nside: int, k: int) -> float:
    """Colatitude of polar-cap ring ``k``: ``cos(theta) = 1 - k^2 / (3 nside^2)``."""
    return math.acos(1.0 - k * k / (3.0 * nside * nside))


def _harmonic_basis(lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``B [N, 3]`` of (1, cos, sin) and its pseudo-inverse ``P [3, N]`` (the wave-0/1 fit)."""
    basis = np.stack([np.ones_like(lon), np.cos(lon), np.sin(lon)], axis=1)
    return basis, np.linalg.pinv(basis)


def build_down_correction(fine_nside: int, depth: int | None = None):
    """Matrices of the downsample correction.

    Returns ``(fine_idx, coarse_idx, Mx, My)``: the correction reads the fine pixels
    ``fine_idx`` (rings ``1..2K+1`` of both poles) and the stride-2 output pixels
    ``coarse_idx`` (rings ``1..K``), and replaces the latter by ``My @ y + Mx @ x``.
    """
    nf = int(fine_nside)
    nc = nf // 2
    depth = cap_depth(nc) if depth is None else int(depth)
    fine_idx, coarse_idx = [], []
    f_off = c_off = 0
    blocks = []
    for north in (True, False):
        f_slices, c_slices = {}, {}
        for r in range(1, 2 * depth + 2):
            idx, lon = _ring(nf, north, r)
            f_slices[r] = (slice(f_off, f_off + idx.size), lon)
            fine_idx.append(idx)
            f_off += idx.size
        for k in range(1, depth + 1):
            idx, lon = _ring(nc, north, k)
            c_slices[k] = (slice(c_off, c_off + idx.size), lon)
            coarse_idx.append(idx)
            c_off += idx.size
        blocks.append((f_slices, c_slices))
    Mx = np.zeros((c_off, f_off))
    My = np.zeros((c_off, c_off))
    for f_slices, c_slices in blocks:
        for k, (cs, clon) in c_slices.items():
            basis, pinv = _harmonic_basis(clon)
            My[cs, cs] = np.eye(clon.size) - basis @ pinv  # keep waves >= 2 of the stride-2 output
            for r, wt in ((2 * k - 1, 0.25), (2 * k, 0.5), (2 * k + 1, 0.25)):
                fs, flon = f_slices[r]
                _, fp = _harmonic_basis(flon)
                # coefficients (a0, a1, b1) of the fine ring, evaluated on the coarse ring's longitudes
                Mx[cs, fs] += wt * basis @ fp
    return np.concatenate(fine_idx), np.concatenate(coarse_idx), Mx, My


def build_up_correction(coarse_nside: int, depth: int | None = None):
    """Matrices of the upsample correction.

    Returns ``(coarse_idx, fine_idx, Mc, Mf)``: the correction reads the coarse pixels
    ``coarse_idx`` (rings ``1..K+1``) and the PCHIP output pixels ``fine_idx`` (rings
    ``1..2K``), and replaces the latter by ``Mf @ f + Mc @ c``.
    """
    nc = int(coarse_nside)
    nf = 2 * nc
    depth = cap_depth(nc) if depth is None else int(depth)
    coarse_idx, fine_idx = [], []
    c_off = f_off = 0
    blocks = []
    for north in (True, False):
        c_slices, f_slices = {}, {}
        for k in range(1, depth + 2):
            idx, lon = _ring(nc, north, k)
            c_slices[k] = (slice(c_off, c_off + idx.size), lon)
            coarse_idx.append(idx)
            c_off += idx.size
        for r in range(1, 2 * depth + 1):
            idx, lon = _ring(nf, north, r)
            f_slices[r] = (slice(f_off, f_off + idx.size), lon)
            fine_idx.append(idx)
            f_off += idx.size
        blocks.append((c_slices, f_slices))
    Mc = np.zeros((f_off, c_off))
    Mf = np.zeros((f_off, f_off))
    theta_c = {k: _colatitude(nc, k) for k in range(1, depth + 2)}
    for c_slices, f_slices in blocks:
        for r, (fs, flon) in f_slices.items():
            basis, pinv = _harmonic_basis(flon)
            if r == 1:
                replaced = [0, 1, 2]  # waves 0 and 1: extrapolate from coarse rings 1 and 2
            else:
                replaced = [1, 2]  # wave 1 only: interpolate in colatitude
            keep = np.eye(flon.size) - basis[:, replaced] @ pinv[replaced]
            Mf[fs, fs] = keep
            if r == 1:
                weights = {1: 1.5, 2: -0.5}
            else:
                theta = _colatitude(nf, r)
                k = max(kk for kk in range(1, depth + 1) if theta_c[kk] <= theta + 1e-12)
                t = (theta - theta_c[k]) / (theta_c[k + 1] - theta_c[k])
                weights = {k: 1.0 - t, k + 1: t}
            for k, w in weights.items():
                cs, clon = c_slices[k]
                _, cpinv = _harmonic_basis(clon)
                # coefficients of coarse ring k, mapped onto the fine ring's longitudes
                Mc[fs, cs] += w * basis[:, replaced] @ cpinv[replaced]
    return np.concatenate(coarse_idx), np.concatenate(fine_idx), Mc, Mf


def _flat_to_fhw(idx: np.ndarray, nside: int):
    n2 = nside * nside
    face = idx // n2
    rem = idx - face * n2
    return face, rem // nside, rem % nside


class PolarCapDownCorrection(th.nn.Module):
    """Correct a stride-2 downsample near both poles (see the module docstring).

    ``forward(x_fine, y_coarse)`` takes the stage input ``[B*12, C, 2n, 2n]`` and its
    stride-2 output ``[B*12, C, n, n]`` and overwrites the coarse polar rings of the
    output in place (NCHW or channels-last). The map is linear in both arguments.
    """

    def __init__(self, fine_nside: int, depth: int | None = None):
        super().__init__()
        self.fine_nside = int(fine_nside)
        if self.fine_nside < 4 or self.fine_nside % 2:
            raise ValueError(f"fine_nside must be even and >= 4, got {fine_nside}")
        self.coarse_nside = self.fine_nside // 2
        fine_idx, coarse_idx, Mx, My = build_down_correction(self.fine_nside, depth)
        for name, (idx, ns) in {"x": (fine_idx, self.fine_nside), "y": (coarse_idx, self.coarse_nside)}.items():
            f, h, w = _flat_to_fhw(idx, ns)
            self.register_buffer(f"_{name}_face", th.from_numpy(f), persistent=False)
            self.register_buffer(f"_{name}_h", th.from_numpy(h), persistent=False)
            self.register_buffer(f"_{name}_w", th.from_numpy(w), persistent=False)
        self.register_buffer("_Mx", th.from_numpy(Mx).float(), persistent=False)
        self.register_buffer("_My", th.from_numpy(My).float(), persistent=False)

    def _gather(self, t: th.Tensor, name: str) -> th.Tensor:
        b = t.shape[0] // 12
        n = th.arange(b, device=t.device)[:, None] * 12 + getattr(self, f"_{name}_face")[None, :]
        return t[n, :, getattr(self, f"_{name}_h")[None, :], getattr(self, f"_{name}_w")[None, :]]  # [B, N, C]

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        if x.shape[-1] != self.fine_nside or y.shape[-1] != self.coarse_nside:
            raise ValueError(
                f"expected fine/coarse face sizes {self.fine_nside}/{self.coarse_nside}, "
                f"got {x.shape[-1]}/{y.shape[-1]}"
            )
        px = self._gather(x, "x").float()
        py = self._gather(y, "y").float()
        new = th.einsum("on,bnc->boc", self._Mx, px) + th.einsum("op,bpc->boc", self._My, py)
        b = y.shape[0] // 12
        n = th.arange(b, device=y.device)[:, None] * 12 + self._y_face[None, :]
        y[n, :, self._y_h[None, :], self._y_w[None, :]] = new.to(y.dtype)
        return y


class PolarCapUpCorrection(th.nn.Module):
    """Correct a ring-mean PCHIP upsample near both poles (see the module docstring).

    ``forward(fine, coarse)`` takes the resampled field ``[B, C, 12*(2n)^2]`` and the
    coarse input ``[B, C, 12*n^2]`` (face-major flat pixels) and overwrites the fine
    polar rings in place.
    """

    def __init__(self, coarse_nside: int, depth: int | None = None):
        super().__init__()
        self.coarse_nside = int(coarse_nside)
        if self.coarse_nside < 2:
            raise ValueError(f"coarse_nside must be >= 2, got {coarse_nside}")
        coarse_idx, fine_idx, Mc, Mf = build_up_correction(self.coarse_nside, depth)
        self.register_buffer("_coarse_idx", th.from_numpy(coarse_idx), persistent=False)
        self.register_buffer("_fine_idx", th.from_numpy(fine_idx), persistent=False)
        self.register_buffer("_Mc", th.from_numpy(Mc).float(), persistent=False)
        self.register_buffer("_Mf", th.from_numpy(Mf).float(), persistent=False)

    def forward(self, fine: th.Tensor, coarse: th.Tensor) -> th.Tensor:
        pc = coarse[..., self._coarse_idx].float()
        pf = fine[..., self._fine_idx].float()
        new = pf @ self._Mf.t() + pc @ self._Mc.t()
        fine[..., self._fine_idx] = new.to(fine.dtype)
        return fine
