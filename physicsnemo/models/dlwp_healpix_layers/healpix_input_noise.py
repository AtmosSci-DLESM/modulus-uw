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

Online mode (``online=...``, ``spectrum_path`` null) needs no file and no trained model: the
module measures what it needs from the training batches themselves, through ``observe``.

* ``model_error``: running mean of the per-degree variance of the model's own 24 h training error.
* ``persistence``: same for the 24 h persistence error (target minus input), times ``factor``.
* ``replay``: a bank of recent real 24 h error fields (spherical-harmonic coefficients, stored
  after the per-channel cap), redrawn with a random zonal phase and the usual level draw, so
  non-Gaussian and cross-channel structure of the model's own error is kept.

All three cap the error rms per channel at ``cap_sigma`` normalized units and apply the same
degree window. Tables are updated in place, so a captured CUDA graph sees the new noise.

Noise is in the normalized units of the model inputs. It is applied by the model in
training mode only; evaluation, rollout validation and inference are unaffected.

The cuhpx transform is exact up to degree about ``2 * nside`` and loses power
toward the band edge (``lmax = 3 * nside - 1``): analysing a synthesized unit
coefficient at degree 190 returns 0.4 to 1.0 of its power depending on the order. The
stored spectrum is measured with the same analysis transform, so the noise matches it
closely below degree ``2 * nside`` and is mildly under-represented above.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Mapping, Sequence

import torch

logger = logging.getLogger(__name__)

# Transform objects that are read-only after construction. ``copy.deepcopy`` of a model (the rollout validator copies the
# model for its private fp32 graph) shares them instead of duplicating their CUDA tables.
_SHARED_TRANSFORMS = ("_isht", "_sht", "_from_ring", "_to_ring")

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


def trim_sht_tables(transform, keep: str) -> None:
    """Replace the unused fp64 / duplicate Legendre tables of a cuhpx transform by empty CUDA tensors.

    ``SHTCUDA`` and ``iSHTCUDA`` each build ``pct``, ``pct_f64``, ``pct_weights`` and ``pct_weights_f64`` (about 213 MiB at
    nside 64, lmax 191). The fp32 forward reads only ``pct_weights`` (``SHTCUDA``) or ``pct`` (``iSHTCUDA``); the others
    serve fp64 input and the backward pass, neither of which this module uses (no gradient, fp32 only).
    ``keep`` is the name of the table to keep. Empty tensors stay on the CUDA device: ``SHTFunction`` tests ``pct.is_cuda``.
    """
    for name in ("pct", "pct_f64", "pct_weights", "pct_weights_f64"):
        if name != keep:
            setattr(transform, name, torch.empty(0, device=getattr(transform, keep).device))


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
        spectrum_path: str | None,
        in_channels: Sequence[str],
        scaling: Mapping[str, Mapping[str, float]],
        nside: int = 64,
        lmax: int | None = None,
        ell_start: int = 32,
        ell_full: int = 64,
        variance_scale_range: Sequence[float] = (0.0, 2.0),
        amplitude: float = 1.0,
        lower_bounds: Mapping[str, float] | None = None,
        online: str | None = None,
        ema_obs: int = 40,
        update_every: int = 5,
        cap_sigma: float = 0.3,
        factor: float = 1.0,
        bank_size: int = 32,
        clean_s: float = 0.3,
        ell_stop: int | None = None,
    ):
        super().__init__()
        if online not in (None, "model_error", "persistence", "replay"):
            raise ValueError(f"online must be null, model_error, persistence or replay, got {online}")
        if online is None and spectrum_path is None:
            raise ValueError("spectrum_path is required unless online is set")
        self.online = online
        self.ema_obs, self.update_every = int(ema_obs), int(update_every)
        self.cap_sigma, self.factor, self.bank_size = float(cap_sigma), float(factor), int(bank_size)
        self.clean_s = float(clean_s)
        self._n_obs = 0
        self._bank_ptr = 0
        self._bank_filled = 0
        self.in_channels = [str(c) for c in in_channels]
        self.n_channels = len(self.in_channels)
        self.nside = int(nside)
        if self.nside < 1 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError(f"nside must be a positive power of 2, got {nside}")
        self.lmax = int(lmax) if lmax is not None else 3 * self.nside - 1
        self.ell_start = int(ell_start)
        self.ell_full = int(ell_full)
        self.ell_stop = None if ell_stop is None else int(ell_stop)  # no noise at degrees >= ell_stop (band-limit ablation)
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
        object.__setattr__(self, "_sht", None)
        object.__setattr__(self, "_to_ring", None)

        if self.online is not None:
            self._init_online(scaling, lower_bounds)
            return
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

    def __deepcopy__(self, memo):
        """Deep copy that shares the transform objects (read-only, with CUDA tables) instead of duplicating them."""
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = value if key in _SHARED_TRANSFORMS else copy.deepcopy(value, memo)
        return new

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
        if self.ell_stop is not None:
            window = window * (ell < self.ell_stop).to(window.dtype)
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

        self._build_lower(scaling, lower_bounds)

    def _build_lower(self, scaling, lower_bounds) -> None:
        if lower_bounds:
            lower = torch.full((self.n_channels,), float("-inf"))
            for name, bound in lower_bounds.items():
                i = self.in_channels.index(name)
                lower[i] = (float(bound) - float(scaling[name]["mean"])) / float(
                    scaling[name]["std"]
                )
            self._lower = lower.reshape(1, 1, 1, -1, 1, 1)

    def _init_online(self, scaling, lower_bounds) -> None:
        """Zero tables and the transforms; ``observe`` fills the tables from training batches."""
        C, L = self.n_channels, self.lmax
        ell = torch.arange(L)
        m = torch.arange(L)
        self._window = spectral_ramp(ell, self.ell_start, self.ell_full).float()
        if self.ell_stop is not None:
            self._window = self._window * (ell < self.ell_stop).float()
        self._window[0] = 0.0
        valid = (m[None, :] <= ell[:, None]).float()
        half = 1.0 / math.sqrt(2.0)
        # E|a_lm|^2 split: m = 0 real, m > 0 half in re and half in im
        self._valid_re = valid * torch.where(m == 0, 1.0, half)[None, :]
        self._valid_im = valid * torch.where(m == 0, 0.0, half)[None, :]
        self._two_l1 = (2 * ell + 1).float()
        self.register_buffer("_re_scale", torch.zeros(C, L, L), persistent=False)
        self.register_buffer("_im_scale", torch.zeros(C, L, L), persistent=False)
        self.register_buffer("_last_s", torch.zeros(256), persistent=False)
        self.register_buffer("_var_ema", torch.zeros(C, L, dtype=torch.float64), persistent=False)
        if self.online == "replay":
            self.register_buffer("_bank", torch.zeros(self.bank_size, 2 * C * L, L), persistent=False)  # rank 3: the model casts rank-4/5 buffers to channels_last
            self.register_buffer("_bank_n", torch.zeros(1), persistent=False)
        self._build_lower(scaling, dict(lower_bounds))
        try:
            self._build_transforms(
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        except Exception as exc:
            self._unavailable_reason = f"spherical-harmonic transform unavailable: {exc}"
            logger.warning("SpectralInputNoise is unavailable (%s).", self._unavailable_reason)
            return
        logger.info(
            "SpectralInputNoise online=%s: cap %g sigma, window ell %d..%d, ema_obs %d, update_every %d, factor %g",
            self.online, self.cap_sigma, self.ell_start, self.ell_full, self.ema_obs, self.update_every, self.factor,
        )

    @torch.no_grad()
    def _analyse(self, err: torch.Tensor):
        """``err`` [B, C, 12, H, W] normalized -> alm [B, C, L, L] complex (fp32)."""
        device = err.device
        if self._transform_device != device:
            object.__setattr__(self, "_from_ring", self._from_ring.to(device))
            object.__setattr__(self, "_to_ring", self._to_ring.to(device))
            object.__setattr__(self, "_transform_device", device)
        B, C = err.shape[:2]
        flat = err.float().reshape(B * C, -1).contiguous()
        alm = self._sht(self._to_ring(flat).contiguous())
        return alm.reshape(B, C, self.lmax, self.lmax)

    @torch.no_grad()
    def observe(self, error: torch.Tensor) -> None:
        """Update the online noise from a 24 h error field.

        ``error`` is ``[B, 12, C, H, W]`` in normalized units: model output minus target for
        ``model_error`` and ``replay``, target minus input for ``persistence``.
        """
        if self.online is None or self._unavailable_reason is not None:
            return
        if self._re_scale.device != error.device:
            self.to(error.device)
        with torch.amp.autocast(device_type=error.device.type, enabled=False):
            B = error.shape[0]
            # The error of a noisy forward contains the injected noise the network has not learned to remove yet, which
            # would feed back into the estimate. Only low-level samples (this forward's variance multiplier s < clean_s)
            # measure the model's own error; the persistence error is of the data and needs no selection.
            if self.online == "persistence":
                w = torch.ones(B, device=error.device)
            else:
                w = (1.0 - self._last_s[:B].to(error.device) / self.clean_s).clamp(min=0.0)
            alm = self._analyse(error.permute(0, 2, 1, 3, 4))
            dv = degree_variance(alm)  # [B, C, L]
            if self.online == "replay":
                use = (w > 0).nonzero().flatten()
                if use.numel() == 0:
                    return
                alm, dv = alm[use], dv[use]
                keep = (self.cap_sigma / dv.sum(-1).sqrt().clamp_min(1e-12)).clamp(max=1.0)
                alm = alm * keep[..., None, None] * self._window.to(alm.device)[:, None]
                n = alm.shape[0]
                idx = (self._bank_ptr + torch.arange(n, device=alm.device)) % self.bank_size
                self._bank.index_copy_(0, idx, torch.stack([alm.real, alm.imag], dim=1).reshape(n, -1, self.lmax))
                self._bank_ptr = (self._bank_ptr + n) % self.bank_size
                self._bank_filled = min(self.bank_size, self._bank_filled + n)
                self._bank_n.fill_(float(self._bank_filled))
                self._n_obs += 1
                return
            num = torch.cat([(w[:, None, None] * dv).sum(0).double().flatten(), w.sum().double().reshape(1)])
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(num, op=torch.distributed.ReduceOp.SUM)
            if num[-1] < 0.2:  # no clean-enough sample anywhere this time
                return
            mean = (num[:-1] / num[-1]).reshape(self.n_channels, self.lmax)
            self._n_obs += 1
            a = 1.0 / min(self._n_obs, self.ema_obs)
            self._var_ema.mul_(1 - a).add_(mean * a)
            var = self._var_ema * self.factor
            keep = (self.cap_sigma / var.sum(-1).sqrt().clamp_min(1e-12)).clamp(max=1.0)
            var = (var * keep.square()[:, None]).float() * self._window.to(var.device).square()[None, :]
            sigma = (_FOUR_PI * var / self._two_l1.to(var.device)[None, :]).sqrt()
            self._re_scale.copy_(sigma[:, :, None] * self._valid_re.to(var.device)[None])
            self._im_scale.copy_(sigma[:, :, None] * self._valid_im.to(var.device)[None])

    @property
    def rms_now(self) -> torch.Tensor:
        """Noise rms per channel at level 1 (normalized units), for logging."""
        L = self.lmax
        w = torch.full((L,), 2.0, device=self._re_scale.device)
        w[0] = 1.0
        if self.online == "replay":
            n = int(self._bank_n.item())
            if n == 0:
                return torch.zeros(self.n_channels)
            a = self._bank[:n].reshape(n, 2, self.n_channels, L, L)
            return ((a[:, 0].square() + a[:, 1].square()) * w).sum((-1, -2)).div(_FOUR_PI).mean(0).sqrt().cpu()
        return ((self._re_scale.square() + self._im_scale.square()) * w).sum((-1, -2)).div(_FOUR_PI).sqrt().cpu()

    def _build_transforms(self, device: torch.device) -> None:
        import earth2grid
        from cuhpx import SHTCUDA, iSHTCUDA
        from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder

        level = int(round(math.log2(self.nside)))
        ring = earth2grid.healpix.Grid(level=level, pixel_order=PixelOrder.RING)
        faces = earth2grid.healpix.Grid(level=level, pixel_order=HEALPIX_PAD_XY)
        from_ring = earth2grid.get_regridder(ring, faces).to(torch.float32).to(device)
        isht = iSHTCUDA(
            nside=self.nside, lmax=self.lmax, mmax=self.lmax, quad_weights="ring"
        )
        trim_sht_tables(isht, keep="pct")
        object.__setattr__(self, "_from_ring", from_ring)
        object.__setattr__(self, "_isht", isht)
        if self.online is not None:
            sht = SHTCUDA(nside=self.nside, lmax=self.lmax, mmax=self.lmax, quad_weights="ring")
            trim_sht_tables(sht, keep="pct_weights")
            object.__setattr__(self, "_to_ring", earth2grid.get_regridder(faces, ring).to(torch.float32).to(device))
            object.__setattr__(self, "_sht", sht)
        object.__setattr__(self, "_transform_device", device)

    def _log_summary(self) -> None:
        # One line per job, not per rank.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
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
        if self.online == "replay":
            # a stored real error, rotated in longitude by one random angle per sample (all channels
            # together, so cross-channel structure survives); an empty bank gives zeros.
            idx = (torch.rand(batch, device=device) * self._bank_n.clamp(min=1)).long().clamp(max=self.bank_size - 1)
            a = self._bank[idx].reshape(batch, 2, self.n_channels, L, L)
            ang = torch.rand(batch, 1, 1, 1, device=device) * (2.0 * math.pi) * torch.arange(L, device=device)
            c, s_ = ang.cos(), ang.sin()
            re = a[:, 0] * c - a[:, 1] * s_
            im = a[:, 0] * s_ + a[:, 1] * c
        else:
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

    def draw_level(self, batch: int, device: torch.device):
        """Per-sample std multiplier ``amplitude * sqrt(s)`` for a whole model forward (all rollout steps).

        Drawn once per forward so the online modes know which samples were nearly noise-free; they record ``s``.
        """
        u = torch.rand((batch, 1, 1, 1, 1), device=device)
        s = self.scale_lo + (self.scale_hi - self.scale_lo) * u
        if self.online is not None:
            if self._last_s.device != device:
                self.to(device)
            self._last_s[:batch].copy_(s.reshape(batch))
        return self.amplitude * s.sqrt()

    def forward(self, state: torch.Tensor, level=None) -> torch.Tensor:
        """Return ``state`` plus noise; ``state`` is ``[B, F, T, C, H, W]``.

        ``level`` is a multiplier from ``draw_level`` shared by several calls; ``None`` draws a fresh one.

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
        # The transform runs in fp32: under autocast its internal matmuls would drop to
        # bf16, which complex views reject, and the noise needs no gradient or bf16 speed.
        with torch.no_grad(), torch.amp.autocast(device_type=state.device.type, enabled=False):
            noise = self._sample(batch, state.device)
            level = self._level(batch, state.device) if level is None else level
            if not (isinstance(level, float) and level == 1.0):
                noise = noise * level
            noise = noise.permute(0, 2, 1, 3, 4).unsqueeze(2)
        out = state + noise.to(state.dtype)
        if self._lower is not None:
            out = torch.maximum(out, self._lower.to(device=out.device, dtype=out.dtype))
        return out
