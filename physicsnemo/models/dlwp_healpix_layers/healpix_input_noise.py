"""Training-time spectral noise for the prognostic inputs of a HEALPix RecUNet.

Some datasets carry much less small-scale variability than the fields the model has
to be stable on (for example thick vertically averaged layers regridded through a
coarse grid). A residual network trained only on such inputs sees almost no
small-scale perturbation, so the loss never asks it to remove one: the residual
add passes small scales through with gain near one and anything the network itself
adds at the grid scale accumulates over a rollout.

``SpectralInputNoise`` adds a random isotropic Gaussian field to the prognostic state
before each model step. The targets stay clean, so the MSE-optimal response to the
noise is to remove it: at each degree the network learns to keep roughly
``S / (S + N)`` of the input, with ``S`` the signal variance and ``N`` the noise
variance. The noise therefore sets how strongly each scale is damped.

The noise spectrum is data-driven. ``spectrum_path`` holds, per channel, the
per-degree pixel variance the training data is missing relative to a reference
dataset (see ``save_noise_spectrum``). Channels whose data already has enough small-scale
variance carry zero in that file and receive no noise. Around that fixed spectrum the
draw is random in three ways:

* Pattern: independent complex Gaussian spherical-harmonic coefficients per batch
  member, channel and call, so the expected spectrum is the stored one while every
  realization differs.
* Window: noise is zero for degrees at or below ``ell_start`` and ramps to full
  strength at ``ell_full`` with a raised cosine. Large scales carry predictable
  signal that the noise would only devalue, and the field has no degree-0 term, so
  global means (and mass or energy constraints that pin them) are unchanged.
* Level: each batch member draws a variance multiplier ``s ~ U(lo, hi)`` from
  ``variance_scale_range``. ``s = 1`` is the stored (reference) level; ``(1, 1)`` is a
  fixed level, ``(0, 2)`` averages to the reference level and also shows the model
  clean inputs, ``(0, 5)`` reaches five times the reference variance.

Noise is in the normalized units of the model inputs. It is applied by the model in
training mode only; evaluation, rollout validation and inference are unaffected.

The cuhpx transform is exact up to degree about ``2 * nside`` and loses power
toward the band edge (``lmax = 3 * nside - 1``): analysing a synthesized unit
coefficient at degree 190 returns 0.4 to 1.0 of its power depending on the order. The
stored spectrum is measured with the same analysis transform, so the noise matches it
closely below degree ``2 * nside`` and is mildly under-represented above.
"""

from __future__ import annotations

import logging
import math
from typing import Mapping, Sequence

import torch

logger = logging.getLogger(__name__)

SPECTRUM_VARIABLE = "variance"
_FOUR_PI = 4.0 * math.pi


def degree_variance(alm: torch.Tensor) -> torch.Tensor:
    """Pixel variance contributed by each degree, from coefficients stored at ``m >= 0``.

    The cuhpx transform is orthonormal, so the area mean of ``f**2`` equals
    ``sum_l degree_variance[l]`` with ``degree_variance[l] = (|a_l0|^2 + 2 sum_{m>0}
    |a_lm|^2) / (4 pi)``. ``alm`` is ``[..., lmax, mmax]`` complex; the result is
    ``[..., lmax]``.
    """
    amp = alm.real.square() + alm.imag.square()
    weight = torch.full((amp.shape[-1],), 2.0, device=amp.device, dtype=amp.dtype)
    weight[0] = 1.0
    return (amp * weight).sum(dim=-1) / _FOUR_PI


def spectral_ramp(ell: torch.Tensor, start: int, full: int) -> torch.Tensor:
    """Amplitude window: 0 through ``start``, 1 from ``full`` up, raised cosine between."""
    if full <= start:
        raise ValueError(f"ell_full ({full}) must be greater than ell_start ({start})")
    t = ((ell.to(torch.float64) - float(start)) / float(full - start)).clamp(0.0, 1.0)
    return 0.5 * (1.0 - torch.cos(math.pi * t))


def save_noise_spectrum(
    path: str,
    channels: Sequence[str],
    variance: torch.Tensor,
    attrs: Mapping[str, object] | None = None,
) -> None:
    """Write a per-channel, per-degree pixel-variance spectrum in physical units.

    ``variance`` is ``[n_channels, lmax]``: the variance each degree contributes to the
    pixel variance of the channel (so the row sum is the missing pixel variance).
    """
    import numpy as np
    import xarray as xr

    values = torch.as_tensor(variance).detach().cpu().double().numpy()
    if values.ndim != 2 or values.shape[0] != len(channels):
        raise ValueError(
            f"variance must be [n_channels={len(channels)}, lmax], got {values.shape}"
        )
    ds = xr.Dataset(
        {SPECTRUM_VARIABLE: (("channel", "ell"), values)},
        coords={"channel": list(channels), "ell": np.arange(values.shape[1])},
        attrs={k: str(v) for k, v in (attrs or {}).items()},
    )
    ds.to_netcdf(path)


def load_noise_spectrum(path: str) -> tuple[list[str], torch.Tensor]:
    """Read a spectrum written by ``save_noise_spectrum``: channel names and ``[C, lmax]``."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        channels = [str(c) for c in ds["channel"].values]
        values = torch.as_tensor(ds[SPECTRUM_VARIABLE].values, dtype=torch.float64)
    return channels, values


class SpectralInputNoise(torch.nn.Module):
    """Random band-limited Gaussian noise for ``[B, F, T, C, H, W]`` prognostic states.

    Parameters
    ----------
    spectrum_path:
        Netcdf file from ``save_noise_spectrum``. If it does not exist the module is
        built in an unavailable state (a warning is logged) so that checkpoints which
        record this layer still load where the file is absent, such as inference; a
        training-mode call then raises.
    in_channels:
        Names of the state channels, in tensor order.
    scaling:
        Mapping channel name to ``{"mean", "std"}`` used to normalize the model inputs.
        The stored spectrum is divided by ``std**2``.
    nside, lmax:
        HEALPix resolution of the state and spherical-harmonic bandwidth
        (default ``3 * nside - 1``, matching the loss and constraint code).
    ell_start, ell_full:
        Amplitude window: no noise at degrees ``<= ell_start``, full strength from
        ``ell_full``.
    variance_scale_range:
        ``(lo, hi)`` of the per-sample variance multiplier ``s ~ U(lo, hi)``.
        ``(1, 1)`` gives a fixed level without a random draw.
    amplitude:
        Global multiplier on the noise standard deviation.
    lower_bounds:
        Mapping channel name to a physical lower bound. Those channels are clamped at
        the (normalized) bound after the noise is added, so noisy inputs stay physical.
    """

    def __init__(
        self,
        spectrum_path: str,
        in_channels: Sequence[str],
        scaling: Mapping[str, Mapping[str, float]],
        nside: int = 64,
        lmax: int | None = None,
        ell_start: int = 32,
        ell_full: int = 64,
        variance_scale_range: Sequence[float] = (0.0, 2.0),
        amplitude: float = 1.0,
        lower_bounds: Mapping[str, float] | None = None,
    ):
        super().__init__()
        self.in_channels = [str(c) for c in in_channels]
        self.n_channels = len(self.in_channels)
        self.nside = int(nside)
        if self.nside < 1 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError(f"nside must be a positive power of 2, got {nside}")
        self.lmax = int(lmax) if lmax is not None else 3 * self.nside - 1
        self.ell_start = int(ell_start)
        self.ell_full = int(ell_full)
        if not (0 <= self.ell_start < self.ell_full <= self.lmax):
            raise ValueError(
                "need 0 <= ell_start < ell_full <= lmax, got "
                f"ell_start={ell_start}, ell_full={ell_full}, lmax={self.lmax}"
            )
        scale_range = [float(v) for v in variance_scale_range]
        if len(scale_range) != 2:
            raise ValueError(
                f"variance_scale_range must be [lo, hi], got {list(variance_scale_range)}"
            )
        self.scale_lo, self.scale_hi = scale_range
        if not (0.0 <= self.scale_lo <= self.scale_hi):
            raise ValueError(
                "variance_scale_range must satisfy 0 <= lo <= hi, got "
                f"{scale_range}"
            )
        self.amplitude = float(amplitude)
        if self.amplitude < 0.0:
            raise ValueError(f"amplitude must be >= 0, got {amplitude}")
        lower_bounds = dict(lower_bounds or {})
        unknown = sorted(set(lower_bounds) - set(self.in_channels))
        if unknown:
            raise ValueError(f"lower_bounds names not in in_channels: {unknown}")

        self._unavailable_reason: str | None = None
        self.expected_rms: dict[str, float] = {}
        self.register_buffer("_re_scale", None, persistent=False)
        self.register_buffer("_im_scale", None, persistent=False)
        self.register_buffer("_lower", None, persistent=False)
        # Transform objects stay out of the module registry: the regridder carries
        # persistent buffers that would otherwise enter the model state dict.
        object.__setattr__(self, "_isht", None)
        object.__setattr__(self, "_from_ring", None)
        object.__setattr__(self, "_transform_device", None)

        try:
            channels, spectrum = load_noise_spectrum(spectrum_path)
        except FileNotFoundError:
            self._unavailable_reason = f"spectrum file not found: {spectrum_path}"
            logger.warning(
                "SpectralInputNoise is unavailable (%s); it can be built for inference "
                "but not used for training.",
                self._unavailable_reason,
            )
            return
        self._build_tables(channels, spectrum, scaling, lower_bounds)
        try:
            self._build_transforms(
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        except Exception as exc:  # cuhpx needs a CUDA device and the earth2grid weights
            self._unavailable_reason = f"spherical-harmonic transform unavailable: {exc}"
            logger.warning("SpectralInputNoise is unavailable (%s).", self._unavailable_reason)
            return
        self._log_summary()

    @property
    def available(self) -> bool:
        return self._unavailable_reason is None

    def _build_tables(self, channels, spectrum, scaling, lower_bounds) -> None:
        index = {name: i for i, name in enumerate(channels)}
        missing = [c for c in self.in_channels if c not in index]
        if missing:
            raise ValueError(f"spectrum file has no entry for channels: {missing}")
        if spectrum.shape[1] != self.lmax:
            raise ValueError(
                f"spectrum file has {spectrum.shape[1]} degrees but lmax is {self.lmax}"
            )
        if (spectrum < 0).any():
            raise ValueError("spectrum file contains negative variances")
        spectrum = spectrum[[index[c] for c in self.in_channels]]
        std = torch.tensor(
            [float(scaling[c]["std"]) for c in self.in_channels], dtype=torch.float64
        )

        ell = torch.arange(self.lmax)
        window = spectral_ramp(ell, self.ell_start, self.ell_full)
        window[0] = 0.0
        # Normalized pixel variance each degree contributes at s = 1.
        variance = spectrum / std[:, None].square() * window.square()[None, :]
        self.expected_rms = {
            c: float(v) for c, v in zip(self.in_channels, variance.sum(dim=1).sqrt())
        }
        # E|a_lm|^2 per (l, m >= 0) giving that variance: the field's pixel variance at
        # degree l is (2l + 1) E|a_lm|^2 / (4 pi) for a real isotropic field.
        sigma = (_FOUR_PI * variance / (2 * ell + 1).double()[None, :]).sqrt()
        m = torch.arange(self.lmax)
        valid = (m[None, :] <= ell[:, None]).double()
        # m = 0 is real with variance sigma^2; m > 0 splits sigma^2 over re and im.
        half = 1.0 / math.sqrt(2.0)
        re_m = torch.where(m == 0, torch.ones(()), torch.full((), half)).double()
        im_m = torch.where(m == 0, torch.zeros(()), torch.full((), half)).double()
        self._re_scale = (sigma[:, :, None] * valid[None] * re_m).float()
        self._im_scale = (sigma[:, :, None] * valid[None] * im_m).float()

        if lower_bounds:
            lower = torch.full((self.n_channels,), float("-inf"))
            for name, bound in lower_bounds.items():
                i = self.in_channels.index(name)
                lower[i] = (float(bound) - float(scaling[name]["mean"])) / float(
                    scaling[name]["std"]
                )
            self._lower = lower.reshape(1, 1, 1, -1, 1, 1)

    def _build_transforms(self, device: torch.device) -> None:
        import earth2grid
        from cuhpx import iSHTCUDA
        from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder

        level = int(round(math.log2(self.nside)))
        ring = earth2grid.healpix.Grid(level=level, pixel_order=PixelOrder.RING)
        faces = earth2grid.healpix.Grid(level=level, pixel_order=HEALPIX_PAD_XY)
        from_ring = earth2grid.get_regridder(ring, faces).to(torch.float32).to(device)
        isht = iSHTCUDA(
            nside=self.nside, lmax=self.lmax, mmax=self.lmax, quad_weights="ring"
        )
        object.__setattr__(self, "_from_ring", from_ring)
        object.__setattr__(self, "_isht", isht)
        object.__setattr__(self, "_transform_device", device)

    def _log_summary(self) -> None:
        mean_scale = 0.5 * (self.scale_lo + self.scale_hi)
        rms = ", ".join(f"{c}={v:.3g}" for c, v in self.expected_rms.items())
        logger.info(
            "SpectralInputNoise: degrees > %d ramp to full strength at %d (lmax %d); "
            "variance_scale_range=(%g, %g) (mean %g); amplitude=%g. Expected noise RMS "
            "per channel at scale 1 in normalized units (multiply by sqrt(scale) * "
            "amplitude): %s",
            self.ell_start,
            self.ell_full,
            self.lmax,
            self.scale_lo,
            self.scale_hi,
            mean_scale,
            self.amplitude,
            rms,
        )

    def _sample(self, batch: int, device: torch.device) -> torch.Tensor:
        """Unit-level noise ``[B, C, 12, H, W]`` in normalized units."""
        # The tables follow ``model.to(device)``; the regridder is outside the module
        # registry, so it and a module used on its own are moved on first use.
        if self._re_scale.device != device:
            self.to(device)
        if self._transform_device != device:
            object.__setattr__(self, "_from_ring", self._from_ring.to(device))
            object.__setattr__(self, "_transform_device", device)
        L = self.lmax
        shape = (batch, self.n_channels, L, L)
        re = torch.randn(shape, device=device) * self._re_scale
        im = torch.randn(shape, device=device) * self._im_scale
        alm = torch.complex(re, im).reshape(-1, L, L)
        ring = self._isht(alm).float().contiguous()
        field = self._from_ring(ring)
        return field.reshape(batch, self.n_channels, 12, self.nside, self.nside)

    def _level(self, batch: int, device: torch.device):
        """Per-sample std multiplier ``amplitude * sqrt(s)``: a float, or ``[B,1,1,1,1]``."""
        if self.scale_lo == self.scale_hi:
            return self.amplitude * math.sqrt(self.scale_lo)
        u = torch.rand((batch, 1, 1, 1, 1), device=device)
        s = self.scale_lo + (self.scale_hi - self.scale_lo) * u
        return self.amplitude * s.sqrt()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return ``state`` plus noise; ``state`` is ``[B, F, T, C, H, W]``.

        One field per batch member and channel is broadcast over the time axis.
        """
        if self._unavailable_reason is not None:
            raise RuntimeError(
                f"SpectralInputNoise cannot be applied: {self._unavailable_reason}"
            )
        batch, faces, _, channels, height, width = state.shape
        if (
            faces != 12
            or channels != self.n_channels
            or height != self.nside
            or width != self.nside
        ):
            raise ValueError(
                f"expected state [B, 12, T, {self.n_channels}, {self.nside}, "
                f"{self.nside}], got {tuple(state.shape)}"
            )
        with torch.no_grad():
            noise = self._sample(batch, state.device)
            level = self._level(batch, state.device)
            if not (isinstance(level, float) and level == 1.0):
                noise = noise * level
            noise = noise.permute(0, 2, 1, 3, 4).unsqueeze(2)
        out = state + noise.to(state.dtype)
        if self._lower is not None:
            out = torch.maximum(out, self._lower.to(device=out.device, dtype=out.dtype))
        return out
