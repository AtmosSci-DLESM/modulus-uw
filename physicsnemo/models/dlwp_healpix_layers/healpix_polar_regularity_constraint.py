"""Polar regularity of the zonal-mean meridional wind.

A smooth vector field has no net meridional flow through a vanishing circle: the ring mean of the northward component
satisfies  v_bar(theta) * 2*pi*a*sin(theta) = -integral over the cap of the divergence,  so  v_bar -> 0  like the colatitude
theta at each pole. The network predicts the (east, north) components on a grid whose polar rings hold only a handful of
pixels, and nothing in the loss ties a ring-mean ``v`` near the pole to this rule. A cap-wide monopole of ``v`` is therefore a
free direction of the map, and it is the direction along which free rollouts run away.

``PolarZonalMeanRegularityConstraint`` replaces, at each pole, the ring means of the chosen channels poleward of ``|lat| = lat_c``
by the straight line in colatitude from the ring mean at the innermost ring equatorward of ``lat_c`` to zero at the pole. It
leaves every departure from the ring mean unchanged and everything equatorward of ``lat_c`` unchanged.

Properties (all exact, no parameters):
* linear in the prediction, so gradients pass through unchanged where the rule does not act;
* acts on ring means only, so a zonally uniform field stays zonally uniform;
* the rule depends on ``|lat|`` only and treats both poles alike, so it commutes with the (lat, lon) -> (-lat, lon) reflection:
  an odd channel stays odd;
* a field that already satisfies the rule (ring mean linear in colatitude inside the cap) is a fixed point.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch


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
    """Pixel -> ring id, and the latitude of each ring, south to north.

    Same helpers as in ``healpix_ring_mean_pchip``; kept here so this constraint does not depend on the ring-mean
    upsampling module.
    """
    key = np.round(lat, decimals=8)
    uniq = np.unique(key)
    order = np.argsort(uniq)
    rank = np.empty(uniq.size, dtype=np.int64)
    rank[order] = np.arange(uniq.size)
    inv = rank[np.searchsorted(uniq, key)]
    return inv.astype(np.int64), uniq[order]


class PolarZonalMeanRegularityConstraint(torch.nn.Module):
    """Make the zonal-mean of the selected channels regular (proportional to colatitude) at both poles.

    Parameters
    ----------
    variables:
        Channels to regularise, normally the northward-wind-like channels (``northward_wind_*``, ``VGRD10m``,
        ``northward_surface_stress``). Names that are not model outputs are ignored.
    in_channels, out_channels:
        Model channel lists. Output channels define the layout of the prediction.
    nside:
        HEALPix nside of the model grid (``[B, F, T, C, nside, nside]`` faces in earth2grid ``HEALPIX_PAD_XY`` order).
    lat_c:
        Latitude in degrees poleward of which the rule applies. 75 degrees covers the polar cap in which the instability
        lives (about 1.7 % of the sphere).
    """

    def __init__(
        self,
        variables: Sequence[str],
        in_channels: Sequence[str],
        out_channels: Sequence[str] | None = None,
        nside: int = 64,
        lat_c: float = 75.0,
    ):
        super().__init__()
        import earth2grid
        from earth2grid.healpix import HEALPIX_PAD_XY

        channels = list(out_channels if out_channels is not None else in_channels)
        self.variables = [v for v in variables if v in channels]
        self.register_buffer(
            "channel_index", torch.tensor([channels.index(v) for v in self.variables], dtype=torch.long), persistent=False
        )
        self.lat_c = float(lat_c)
        self.nside = int(nside)

        grid = earth2grid.healpix.Grid(level=int(round(math.log2(nside))), pixel_order=HEALPIX_PAD_XY)
        ring, ring_lat_rad = _ring_index(_geographic_latitude(grid))
        ring_lat = np.rad2deg(ring_lat_rad)
        sel_rings, src, wgt = [], [], []
        for sign in (1.0, -1.0):
            zone = np.where(sign * ring_lat >= self.lat_c)[0]
            cand = np.where(sign * ring_lat < self.lat_c)[0]
            if zone.size == 0 or cand.size == 0:
                raise ValueError(f"lat_c={lat_c} leaves no polar zone or no boundary ring for nside={nside}")
            boundary = cand[np.argmax(sign * ring_lat[cand])]
            theta_b = 90.0 - sign * ring_lat[boundary]
            sel_rings += [boundary, *zone.tolist()]
            src += [boundary] * (1 + zone.size)
            wgt += [0.0] + [(90.0 - sign * ring_lat[r]) / theta_b for r in zone]
        # position of each selected ring inside ``sel_rings``; the boundary ring is its own source and is not changed
        pos = {r: i for i, r in enumerate(sel_rings)}
        n_sel = len(sel_rings)
        # L @ ring_means = delta of the ring means: zone ring r gets (w_r * mean_boundary - mean_r)
        L = np.zeros((n_sel, n_sel), dtype=np.float64)
        for i, (r, b, w) in enumerate(zip(sel_rings, src, wgt)):
            if r != b:
                L[i, pos[b]] = w
                L[i, i] = -1.0
        member = np.isin(ring, sel_rings)
        pix = np.where(member)[0]
        onehot = np.zeros((pix.size, n_sel), dtype=np.float64)
        for j, p in enumerate(pix):
            onehot[j, pos[ring[p]]] = 1.0
        counts = onehot.sum(0)
        self.register_buffer("pixel_index", torch.tensor(pix, dtype=torch.long), persistent=False)
        self.register_buffer("to_ring_mean", torch.tensor(onehot / counts, dtype=torch.float32), persistent=False)
        self.register_buffer("ring_delta", torch.tensor(L, dtype=torch.float32), persistent=False)
        self.register_buffer("to_pixels", torch.tensor(onehot.T, dtype=torch.float32), persistent=False)

    def forward(self, prediction: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """``prediction`` is ``[B, F, T, C, H, W]``; ``input`` is unused (kept for the constraint interface)."""
        if self.channel_index.numel() == 0:
            return prediction
        b, f, t, c, h, w = prediction.shape
        orig_dtype = prediction.dtype
        with torch.amp.autocast("cuda", enabled=False):
            x = prediction.float()[:, :, :, self.channel_index]                      # [B, F, T, k, H, W]
            k = x.shape[3]
            flat = x.permute(0, 2, 3, 1, 4, 5).reshape(b, t, k, f * h * w)           # pixel order of the face grid
            sel = flat[..., self.pixel_index]                                         # [B, T, k, n_pix]
            delta_pix = ((sel @ self.to_ring_mean) @ self.ring_delta.T) @ self.to_pixels
            flat = flat.index_add(-1, self.pixel_index, delta_pix)
            x = flat.reshape(b, t, k, f, h, w).permute(0, 3, 1, 2, 4, 5)
            out = prediction.float().clone()
            out[:, :, :, self.channel_index] = x
        return out.to(orig_dtype)
