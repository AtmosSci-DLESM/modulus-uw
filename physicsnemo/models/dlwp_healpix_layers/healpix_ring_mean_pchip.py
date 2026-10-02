"""Ring-mean PCHIP upsample followed by a reflection-steerable convolution.

The resample reads the twelve unpadded faces. Ring means are a function of
latitude, so they are prolonged with a monotone cubic in latitude. Departures
from those means use HEALPix bilinear weights, which already look up neighbors
on adjacent faces. The learned 3×3 that follows is padded, because that kernel
still runs on each face.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

import earth2grid
from earth2grid.healpix import HEALPIX_PAD_XY

from physicsnemo.models.layers.activations import Tanh

from .reflection_ops import bank_sizes
from .healpix_paddings import warn_deprecated_enable_healpixpad
from .reflection_steerable_blocks import _PaddedReflectionSteerableConv
from .reflection_steerable_conv import (
    ParitySplitActivation,
    require_reflection_steerable_tanh_activation,
)


def _geographic_latitude(grid) -> np.ndarray:
    """Geographic latitude in radians for an earth2grid HEALPix grid."""
    lat = np.asarray(grid.lat, dtype=np.float64)
    peak = float(np.nanmax(np.abs(lat)))
    if peak > math.pi + 0.1:
        lat = np.deg2rad(lat)
    if float(np.nanmin(lat)) >= -0.1 and peak > math.pi / 2 + 0.1:
        lat = math.pi / 2 - lat
    return lat


def _ring_index(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel → ring id, and the latitude of each ring, south to north."""
    key = np.round(lat, decimals=8)
    uniq = np.unique(key)
    order = np.argsort(uniq)
    rank = np.empty(uniq.size, dtype=np.int64)
    rank[order] = np.arange(uniq.size)
    inv = rank[np.searchsorted(uniq, key)]
    return inv.astype(np.int64), uniq[order]


class RingMeanPCHIPUpsample(nn.Module):
    """Double HEALPix nside. Zonal input stays zonal; a front does not overshoot.

    ``scale_factor`` must be 2. Queries poleward of the coarse rings take the
    end ring's value.
    """

    def __init__(self, nside: int, scale_factor: int = 2):
        super().__init__()
        if scale_factor != 2:
            raise ValueError(f"ring-mean PCHIP scale_factor must be 2, got {scale_factor}")
        nside = int(nside)
        if nside < 1 or (nside & (nside - 1)) != 0:
            raise ValueError(f"nside must be a positive power of 2, got {nside}")
        self.nside = nside
        self.fine_nside = nside * 2

        src = earth2grid.healpix.Grid(level=int(math.log2(nside)), pixel_order=HEALPIX_PAD_XY)
        dst = earth2grid.healpix.Grid(
            level=int(math.log2(self.fine_nside)), pixel_order=HEALPIX_PAD_XY
        )
        src_lat = _geographic_latitude(src)
        dst_lat = _geographic_latitude(dst)
        src_ring, c_lat = _ring_index(src_lat)
        dst_ring, f_lat = _ring_index(dst_lat)
        if c_lat.size < 3:
            raise ValueError(f"need at least 3 coarse rings, got {c_lat.size}")

        # Interval containing each fine ring. Ends clamp to the coarse endpoint.
        k = np.searchsorted(c_lat, f_lat, side="right") - 1
        clamp_left = f_lat <= c_lat[0]
        clamp_right = f_lat >= c_lat[-1]
        k = np.clip(k, 0, c_lat.size - 2)
        h_all = np.diff(c_lat)
        t = (f_lat - c_lat[k]) / h_all[k]
        t = np.where(clamp_left | clamp_right, 0.0, t)

        self.register_buffer("src_ring", torch.from_numpy(src_ring), persistent=False)
        self.register_buffer("dst_ring", torch.from_numpy(dst_ring), persistent=False)
        counts = np.bincount(src_ring, minlength=c_lat.size).astype(np.float64)
        fine_counts = np.bincount(dst_ring, minlength=f_lat.size).astype(np.float64)
        self.register_buffer("src_counts", torch.from_numpy(counts), persistent=False)
        self.register_buffer("dst_counts", torch.from_numpy(fine_counts), persistent=False)
        self.register_buffer("h", torch.from_numpy(h_all.copy()), persistent=False)
        self.register_buffer("interval", torch.from_numpy(k.astype(np.int64)), persistent=False)
        self.register_buffer("interval_t", torch.from_numpy(t.astype(np.float64)), persistent=False)
        self.register_buffer(
            "clamp_left", torch.from_numpy(clamp_left.astype(np.bool_)), persistent=False
        )
        self.register_buffer(
            "clamp_right", torch.from_numpy(clamp_right.astype(np.bool_)), persistent=False
        )
        self.regrid = earth2grid.get_regridder(src, dst).float()

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        """``[B, C, N]`` coarse pixels → ``[B, C, N_fine]`` in the same pixel order."""
        means = _ring_means(faces, self.src_ring, self.src_counts)
        anomaly = faces - means.index_select(-1, self.src_ring)
        # Same weights as earth2grid's regridder. A short gather stays in the
        # CUDA graph; embedding_bag's per-sample-weight backward is slow and
        # its runtime varies across ranks, which stalls the DDP step.
        fine_anom = _bilinear_regrid(anomaly, self.regrid.index, self.regrid.weight)
        fine_mean = _ring_means(fine_anom, self.dst_ring, self.dst_counts)
        zonal = _pchip_to_fine(
            means,
            self.h,
            self.interval,
            self.interval_t,
            self.clamp_left,
            self.clamp_right,
        )
        return fine_anom - fine_mean.index_select(-1, self.dst_ring) + zonal.index_select(
            -1, self.dst_ring
        )


def _bilinear_regrid(
    values: torch.Tensor, index: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Weighted sum of ``P`` source pixels. ``values`` is ``[B, C, N_in]``.

    ``index`` and ``weight`` are the earth2grid regridder buffers, shape
    ``(*N_out, P)``. ``P`` is 4 for HEALPix bilinear, so the loop is a handful
    of gathers rather than one ``embedding_bag``.
    """
    *out_shape, n_neighbors = index.shape
    flat = values.reshape(-1, values.shape[-1])
    src_index = index.reshape(-1, n_neighbors)
    src_weight = weight.reshape(-1, n_neighbors).to(dtype=flat.dtype)
    acc = flat.new_zeros(flat.shape[0], src_index.shape[0])
    for neighbor in range(n_neighbors):
        acc += flat[:, src_index[:, neighbor]] * src_weight[:, neighbor]
    return acc.view(*values.shape[:-1], *out_shape)


def _ring_means(values: torch.Tensor, ring: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Mean of ``[B, C, N]`` on each ring. ``ring`` indexes the last axis."""
    batch, channels, _ = values.shape
    flat = values.reshape(batch * channels, -1)
    sums = values.new_zeros(batch * channels, counts.shape[0])
    sums.scatter_add_(1, ring.view(1, -1).expand_as(flat), flat)
    means = sums / counts.to(dtype=values.dtype)
    return means.view(batch, channels, -1)


def _edge_slope(h0, h1, m0, m1):
    """One-sided PCHIP slope, limited so it does not introduce a new extremum."""
    slope = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    slope = torch.where(torch.sign(slope) != torch.sign(m0), torch.zeros_like(slope), slope)
    capped = (torch.sign(m0) != torch.sign(m1)) & (slope.abs() > 3 * m0.abs())
    return torch.where(capped, 3 * m0, slope)


def _pchip_derivatives(means: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Fritsch–Carlson slopes at the coarse rings. ``means`` is ``[B, C, R]``."""
    delta = (means[..., 1:] - means[..., :-1]) / h.to(dtype=means.dtype)
    left = delta[..., :-1]
    right = delta[..., 1:]
    same = (torch.sign(left) * torch.sign(right)) > 0
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    w1 = w1.to(dtype=means.dtype)
    w2 = w2.to(dtype=means.dtype)
    left_safe = torch.where(same, left, torch.ones_like(left))
    right_safe = torch.where(same, right, torch.ones_like(right))
    interior = torch.where(
        same,
        (w1 + w2) / (w1 / left_safe + w2 / right_safe),
        torch.zeros_like(left),
    )
    hd = h.to(dtype=means.dtype)
    deriv = torch.zeros_like(means)
    deriv[..., 1:-1] = interior
    deriv[..., 0] = _edge_slope(hd[0], hd[1], delta[..., 0], delta[..., 1])
    deriv[..., -1] = _edge_slope(hd[-1], hd[-2], delta[..., -1], delta[..., -2])
    return deriv


def _pchip_to_fine(means, h, interval, interval_t, clamp_left, clamp_right):
    """Evaluate the monotone cubic at each fine ring."""
    deriv = _pchip_derivatives(means, h)
    k = interval
    y0 = means.index_select(-1, k)
    y1 = means.index_select(-1, k + 1)
    d0 = deriv.index_select(-1, k)
    d1 = deriv.index_select(-1, k + 1)
    hk = h.to(dtype=means.dtype).index_select(0, k)
    t = interval_t.to(dtype=means.dtype)
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)
    values = h00 * y0 + h10 * hk * d0 + h01 * y1 + h11 * hk * d1
    values = torch.where(clamp_left, means[..., :1], values)
    values = torch.where(clamp_right, means[..., -1:], values)
    return values


class ReflectionSteerableRingMeanPCHIPConv(nn.Module):
    """Ring-mean PCHIP upsample, then the same padded steerable conv as nearest upsample.

    Constructor matches ``ReflectionSteerableSmoothedInterpolateConv`` so a
    decoder config can swap the target. ``self.conv`` and ``self.act`` keep
    the learned-parameter names of that block.
    """

    reflection_steerable = True

    def __init__(
        self,
        geometry_layer=None,
        in_channels: int = 3,
        out_channels: int = 3,
        kernel_size: int = 3,
        dilation: int = 1,
        scale_factor: int = 2,
        mode: str = "nearest",
        activation: nn.Module | None = None,
        odd_fraction: float = 0.25,
        enable_nhwc: bool = False,
        hpx_padding_mode: str | None = "isolatitude",
        compile_padding: bool = False,
        nside: int = 64,
        enable_healpixpad: bool | None = None,
        **kwargs,
    ):
        super().__init__()
        del geometry_layer, mode, kwargs
        hpx_padding_mode = warn_deprecated_enable_healpixpad(enable_healpixpad, hpx_padding_mode)
        if dilation > 1:
            raise ValueError(
                f"dilation > 1 is not supported for parity hpx resize convolutions, got {dilation}"
            )
        self.odd_fraction = float(odd_fraction)
        self.enable_nhwc = bool(enable_nhwc)
        self.resample = RingMeanPCHIPUpsample(nside=nside, scale_factor=scale_factor)
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
                activation, where="ReflectionSteerableRingMeanPCHIPConv", allow_none=False
            )
            self.act = ParitySplitActivation(
                out_channels, self.odd_fraction, Tanh(), Tanh(), unified=True
            )

    def forward(self, x, sin_lat_gate=None):
        del sin_lat_gate
        channels_last = x.is_contiguous(memory_format=torch.channels_last) or self.enable_nhwc
        x = x.contiguous()
        batch_faces, channels, height, width = x.shape
        if height != self.resample.nside or width != self.resample.nside:
            raise ValueError(
                f"expected face size {self.resample.nside}, got {(height, width)}"
            )
        if batch_faces % 12 != 0:
            raise ValueError(f"batch*faces must be divisible by 12, got {batch_faces}")
        batch = batch_faces // 12
        pixels = x.view(batch, 12, channels, height, width).permute(0, 2, 1, 3, 4).reshape(
            batch, channels, -1
        )
        orig_dtype = pixels.dtype
        with torch.amp.autocast("cuda", enabled=False):
            fine = self.resample(pixels.float())
        fine_n = self.resample.fine_nside
        fine = fine.to(dtype=orig_dtype).view(batch, channels, 12, fine_n, fine_n)
        fine = fine.permute(0, 2, 1, 3, 4).contiguous().view(batch_faces, channels, fine_n, fine_n)
        if channels_last:
            fine = fine.to(memory_format=torch.channels_last)
        fine = self.conv(fine)
        if self.act is not None:
            fine = self.act(fine)
        return fine
