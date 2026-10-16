"""Training-time noise built from the error of the previous step of the same rollout window.

``SpectralInputNoise`` (online modes) tracks the model's error across many training windows (an EMA of per-degree variance,
a cap, a random level). These modules need no state across windows. At rollout step ``k >= 1`` of a training window, the
fed-back state ``x_k`` is compared with the truth of the previous step, ``e = x_k - truth_{k-1}`` (normalized units, detached),
and the noise is a re-arranged copy of that very error:

* ``WindowSpectralNoise`` (``alpha`` is the only parameter): analyse ``e`` into spherical-harmonic coefficients ``e_lm`` for
  every channel, multiply every coefficient by one random unit phase ``exp(i theta_lm)`` that is the same for all channels
  of a sample (for ``m = 0`` a random sign, so the field stays real), drop degree 0, and synthesise ``alpha`` times the result.
  Per channel the power at every ``(l, m)`` equals ``alpha**2`` times the error's, so the zonal-mean (polar-cap) content is
  kept; one common rotation per ``(l, m)`` also keeps the cross-spectra between channels (relative phases) exactly. The
  pattern is decorrelated from the sample's own error. ``alpha = 1`` is the natural choice: the noise has the energy of
  the error the model really made, doubling the input error variance.
* ``WindowErrorSwap`` (``alpha``): the previous-step error of another sample of the batch, global mean removed.

Both add to the state and clamp channels with a physical lower bound (specific humidity), return the noisy state, run
under ``no_grad`` in fp32, and are active in training mode only (the model decides). They hold no state, so a
CUDA-graph capture sees fresh noise at every replay.
"""

from __future__ import annotations

import math
import os
from typing import Mapping, Sequence

import torch

from physicsnemo.models.dlwp_healpix_layers.healpix_input_noise import trim_sht_tables


def _lower_table(in_channels, scaling, lower_bounds):
    if not lower_bounds:
        return None
    lower = torch.full((len(in_channels),), float("-inf"))
    for name, bound in lower_bounds.items():
        i = list(in_channels).index(name)
        lower[i] = (float(bound) - float(scaling[name]["mean"])) / float(scaling[name]["std"])
    return lower.reshape(1, 1, 1, -1, 1, 1)


class _WindowNoiseBase(torch.nn.Module):
    def __init__(self, in_channels, scaling, alpha, lower_bounds):
        super().__init__()
        self.in_channels = [str(c) for c in in_channels]
        self.n_channels = len(self.in_channels)
        self.alpha = float(alpha)
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        unknown = sorted(set(lower_bounds or {}) - set(self.in_channels))
        if unknown:
            raise ValueError(f"lower_bounds names not in in_channels: {unknown}")
        self.register_buffer("_lower", _lower_table(self.in_channels, scaling, lower_bounds), persistent=False)

    def _noise(self, err: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, state: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        """``state`` and ``err`` are ``[B, F, T, C, H, W]``; returns the noisy state."""
        if os.environ.get("WIN_NOISE_DISABLE") == "1":
            return self._forward_eager(state, err)
        return self._forward_impl(state, err)

    @torch.compiler.disable
    def _forward_eager(self, state, err):
        return self._forward_impl(state, err)

    def _forward_impl(self, state: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.amp.autocast(device_type=state.device.type, enabled=False):
            noise = self._noise(err.detach().float())
        out = state + noise.to(state.dtype)
        if self._lower is not None:
            out = torch.maximum(out, self._lower.to(device=out.device, dtype=out.dtype))
        return out


class WindowErrorSwap(_WindowNoiseBase):
    """Noise = ``alpha`` times the previous-step error of another sample of the batch (global mean removed)."""

    def __init__(self, in_channels, scaling, alpha: float = 1.0, lower_bounds: Mapping[str, float] | None = None, **_):
        super().__init__(in_channels, scaling, alpha, lower_bounds)

    def _noise(self, err):
        other = torch.roll(err, shifts=1, dims=0)
        other = other - other.mean(dim=(1, 4, 5), keepdim=True)  # HEALPix cells have equal area: pixel mean = degree 0
        return self.alpha * other


class WindowSpectralNoise(_WindowNoiseBase):
    """Noise = ``alpha`` times the previous-step error with one common random phase rotation per ``(l, m)`` (see module doc)."""

    def __init__(
        self,
        in_channels: Sequence[str],
        scaling: Mapping[str, Mapping[str, float]],
        alpha: float = 1.0,
        nside: int = 64,
        lmax: int | None = None,
        lower_bounds: Mapping[str, float] | None = None,
        **_,
    ):
        super().__init__(in_channels, scaling, alpha, lower_bounds)
        self.nside = int(nside)
        self.lmax = int(lmax) if lmax is not None else 3 * self.nside - 1
        L = self.lmax
        m = torch.arange(L)
        self.register_buffer("_m0", (m == 0)[None, :].expand(L, L).clone(), persistent=False)
        object.__setattr__(self, "_sht", None)
        object.__setattr__(self, "_isht", None)
        object.__setattr__(self, "_to_ring", None)
        object.__setattr__(self, "_from_ring", None)
        self._build_transforms(torch.device("cuda", torch.cuda.current_device()))

    def _build_transforms(self, device: torch.device) -> None:
        import earth2grid
        from cuhpx import SHTCUDA, iSHTCUDA
        from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder

        level = int(round(math.log2(self.nside)))
        ring = earth2grid.healpix.Grid(level=level, pixel_order=PixelOrder.RING)
        faces = earth2grid.healpix.Grid(level=level, pixel_order=HEALPIX_PAD_XY)
        sht = SHTCUDA(nside=self.nside, lmax=self.lmax, mmax=self.lmax, quad_weights="ring")
        isht = iSHTCUDA(nside=self.nside, lmax=self.lmax, mmax=self.lmax, quad_weights="ring")
        trim_sht_tables(sht, keep="pct_weights")
        trim_sht_tables(isht, keep="pct")
        object.__setattr__(self, "_sht", sht)
        object.__setattr__(self, "_isht", isht)
        object.__setattr__(self, "_to_ring", earth2grid.get_regridder(faces, ring).to(torch.float32).to(device))
        object.__setattr__(self, "_from_ring", earth2grid.get_regridder(ring, faces).to(torch.float32).to(device))

    def __deepcopy__(self, memo):
        import copy

        new = type(self).__new__(type(self))
        memo[id(self)] = new
        shared = ("_sht", "_isht", "_to_ring", "_from_ring")
        for key, value in self.__dict__.items():
            new.__dict__[key] = value if key in shared else copy.deepcopy(value, memo)
        return new

    def _rotation(self, device) -> torch.Tensor:
        """One random unit complex number per (l, m); a random sign for m = 0 (real coefficients)."""
        L = self.lmax
        theta = torch.rand(L, L, device=device) * (2.0 * math.pi)
        sign = (torch.rand(L, 1, device=device) < 0.5).float() * 2.0 - 1.0
        re = torch.where(self._m0, sign.expand(L, L), theta.cos())
        im = torch.where(self._m0, torch.zeros_like(theta), theta.sin())
        return torch.complex(re, im)

    def _noise(self, err):
        B, F, T, C, H, W = err.shape
        e = err.reshape(B, F, C, H, W).permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        variant = os.environ.get("WIN_NOISE_VARIANT", "sample")
        if variant == "all":  # all fields at once: one transform call, larger temporaries
            ring = self._to_ring(e.reshape(B * C, -1).contiguous()).contiguous()
            alm = self._sht(ring).reshape(B, C, self.lmax, self.lmax)
            rot = torch.stack([self._rotation(err.device) for _ in range(B)])  # one rotation per sample, shared by its channels
            alm = alm * (self.alpha * rot)[:, None]
            alm[:, :, 0, :] = 0
            field = self._from_ring(self._isht(alm.reshape(B * C, self.lmax, self.lmax)).float().contiguous())
            noise = field.reshape(B, C, F, H, W)
        else:
            nchunk = 2 if variant == "half" else 1
            out = []
            for b in range(B):  # one sample at a time keeps the transform temporaries small (the graph holds their peak)
                rot = self._rotation(err.device)
                parts = []
                for c0 in range(0, C, -(-C // nchunk)):  # channel chunks (the rotation is shared across them)
                    ec = e[b, c0:c0 + -(-C // nchunk)]
                    ring = self._to_ring(ec.reshape(ec.shape[0], -1).contiguous()).contiguous()
                    alm = self._sht(ring)  # [c, L, L] complex
                    alm = alm * (self.alpha * rot)[None]
                    alm[:, 0, :] = 0  # degree 0 (the global mean) stays out, as in SpectralInputNoise
                    parts.append(self._from_ring(self._isht(alm).float().contiguous()).reshape(-1, F, H, W))
                out.append(torch.cat(parts) if len(parts) > 1 else parts[0])
            noise = torch.stack(out)  # [B, C, F, H, W]
        return noise.permute(0, 2, 1, 3, 4).reshape(B, F, T, C, H, W)
