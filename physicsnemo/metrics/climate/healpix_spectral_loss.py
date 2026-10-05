"""Per-variable spherical-harmonic power loss on a HEALPix rollout.

The pixel loss already matches each grid point. This term matches the
global power spectrum of each output channel over the same rollout window.
An absolute penalty on ``P(l)`` would ignore the tail: atmospheric power
falls by orders of magnitude from planetary degrees to the grid scale.
The loss uses the squared relative error at each degree and averages those
errors with weight ``1/l``, so each octave of scales contributes the same
total. ``l = 0`` is omitted; the pixel loss already sees the spatial mean.
"""

from __future__ import annotations

import math
from typing import Sequence

import earth2grid
import torch
from cuhpx import SHTCUDA
from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder

from physicsnemo.metrics.climate.healpix_soft_constraints import (
    SoftConstraint,
    _constraint_name,
)

# Fraction of the mean target power at l >= 1. A degree whose target power
# sits under this floor does not explode the relative error.
POWER_FLOOR_FRACTION = 1e-4


def degree_power(alm: torch.Tensor) -> torch.Tensor:
    """``P(l) = sum_m |a_lm|^2 / (2l+1)`` from coefficients stored at ``m >= 0``.

    The last axis of ``alm`` is azimuthal order. The same sum is applied to
    the prediction and the target, so a factor shared by both spectra cancels
    in the relative error.
    """
    amp = alm.real.square() + alm.imag.square()
    ell = torch.arange(amp.shape[-2], device=amp.device, dtype=amp.dtype)
    return amp.sum(dim=-1) / (2 * ell + 1)


class _DegreePowerFromRing(torch.autograd.Function):
    """Degree power of a RING map, without keeping the SHT graph.

    The forward transform runs under ``no_grad``. Backward recomputes it.
    Saving the rfft and Legendre products through the 24h rollout exhausts
    the 40 GB train-graph pool and the eval capture that reuses that pool
    cannot allocate.
    """

    sht = None

    @staticmethod
    def forward(ctx, ring, two_ell_plus_one):
        ctx.save_for_backward(ring, two_ell_plus_one)
        sht = _DegreePowerFromRing.sht
        with torch.no_grad():
            alm = sht(ring)
            amp = alm.real.square() + alm.imag.square()
            power = amp.sum(dim=-1) / two_ell_plus_one.to(dtype=amp.dtype)
        return power

    @staticmethod
    def backward(ctx, grad_power):
        ring, two_ell_plus_one = ctx.saved_tensors
        sht = _DegreePowerFromRing.sht
        # Backward runs with autograd disabled. Re-enable it so the
        # recomputed transform has a graph to differentiate.
        with torch.enable_grad():
            ring_in = ring.detach().requires_grad_(True)
            alm = sht(ring_in)
            amp = alm.real.square() + alm.imag.square()
            power = amp.sum(dim=-1) / two_ell_plus_one.to(dtype=amp.dtype)
            (grad_ring,) = torch.autograd.grad(power, ring_in, grad_power)
        return grad_ring, None


def log_ell_relative_power_loss(
    pred_power: torch.Tensor,
    target_power: torch.Tensor,
    inv_ell: torch.Tensor,
    inv_ell_sum: torch.Tensor,
    floor_fraction: float = POWER_FLOOR_FRACTION,
) -> torch.Tensor:
    """Squared relative degree error, reduced with weights ``1/l``.

    The last axis of each power tensor is degree and includes ``l = 0``.
    That degree is dropped. ``inv_ell`` is ``1/l`` for ``l = 1 … lmax`` and
    lines up with ``power[..., 1:]``. Target power is detached: the
    normalizer is a constant, and the gradient flows through the prediction.

    The floor is ``floor_fraction`` times the mean target power over
    ``l >= 1`` on that spectrum (one map, one channel).
    """
    if pred_power.shape != target_power.shape:
        raise ValueError(
            "pred and target power shapes must match, "
            f"got {tuple(pred_power.shape)} and {tuple(target_power.shape)}"
        )
    if pred_power.shape[-1] < 2:
        raise ValueError(
            "power spectra must include l=0 and at least one higher degree, "
            f"got length {pred_power.shape[-1]}"
        )
    if inv_ell.shape[-1] != pred_power.shape[-1] - 1:
        raise ValueError(
            "inv_ell must cover l >= 1, "
            f"got {inv_ell.shape[-1]} weights for "
            f"{pred_power.shape[-1]} degrees"
        )

    pred = pred_power[..., 1:]
    target = target_power[..., 1:].detach()
    mean_target = target.mean(dim=-1, keepdim=True)
    floor = float(floor_fraction) * mean_target
    denom = torch.maximum(target, floor)
    # A spectrum that is zero at every l >= 1 has a zero floor. Clamp only
    # so that 0/0 from an identically empty tail stays zero instead of NaN.
    denom = denom.clamp_min(torch.finfo(pred.dtype).tiny)
    relative = (pred - target) / denom
    weight = inv_ell.to(dtype=relative.dtype)
    weighted = relative.square() * weight
    return weighted.sum(dim=-1) / inv_ell_sum.to(dtype=relative.dtype)


class SpectralPowerSoftConstraint(SoftConstraint):
    """Match each output channel's spherical-harmonic power spectrum.

    One loss term per output variable. A float ``relative_loss_scale`` is
    applied to every variable (adaptive loss weights require a mapping once
    a constraint has more than one term).
    """

    needs_input = False
    needs_input_diagnostics = False

    def __init__(
        self,
        name: str = "spectral",
        relative_loss_scale: float = 0.1,
        nside: int = 64,
        lmax: int | None = None,
        floor_fraction: float = POWER_FLOOR_FRACTION,
        channels: Sequence[str] | None = None,
    ):
        super().__init__()
        self.name = _constraint_name(name)
        if isinstance(relative_loss_scale, bool) or not isinstance(
            relative_loss_scale, (int, float)
        ):
            raise TypeError(
                "relative_loss_scale must be a float applied to every "
                f"variable, got {type(relative_loss_scale).__name__}"
            )
        if float(relative_loss_scale) < 0:
            raise ValueError(
                f"relative_loss_scale must be >= 0, got {relative_loss_scale}"
            )
        self.relative_loss_scale = float(relative_loss_scale)
        self.nside = int(nside)
        if self.nside < 1 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError(f"nside must be a power of two, got {nside}")
        self.lmax = int(lmax) if lmax is not None else 3 * self.nside - 1
        self.mmax = self.lmax
        if self.lmax < 2:
            raise ValueError(f"lmax must be >= 2 so l >= 1 exists, got {self.lmax}")
        if floor_fraction < 0:
            raise ValueError(f"floor_fraction must be >= 0, got {floor_fraction}")
        self.floor_fraction = float(floor_fraction)
        self.channel_chunk = 8
        self._variable_names = None if channels is None else [str(c) for c in channels]

        # Degrees 0 .. lmax-1. The loss uses l >= 1, weighted by 1/l.
        ell = torch.arange(self.lmax, dtype=torch.float32)
        inv_ell = 1.0 / ell[1:]
        self.register_buffer("two_ell_plus_one", 2 * ell + 1, persistent=False)
        self.register_buffer("inv_ell", inv_ell, persistent=False)
        self.register_buffer("inv_ell_sum", inv_ell.sum(), persistent=False)

        self.sht = None
        self.reorder_to_ring = None
        _DegreePowerFromRing.sht = None

    def term_names(self) -> list[str]:
        if not self._variable_names:
            raise RuntimeError(
                "SpectralPowerSoftConstraint.setup(trainer) must run before "
                "term names are known"
            )
        return list(self._variable_names)

    def constraint_spec(self) -> tuple[str, list[str], dict]:
        names = [str(term) for term in self.term_names()]
        scale = {name: self.relative_loss_scale for name in names}
        return (_constraint_name(self.name), names, scale)

    def setup(self, trainer) -> None:
        names = list(getattr(trainer, "output_variables", None) or [])
        if self._variable_names is None:
            if not names:
                raise ValueError(
                    "SpectralPowerSoftConstraint needs trainer.output_variables"
                )
            self._variable_names = [str(name) for name in names]
        elif names and names != self._variable_names:
            raise ValueError(
                "channels do not match trainer.output_variables: "
                f"{self._variable_names} vs {names}"
            )
        device = trainer.device
        self.to(device=device)
        if device.type != "cuda":
            return
        if self.sht is not None:
            return
        level = int(math.log2(self.nside))
        src = earth2grid.healpix.Grid(level=level, pixel_order=HEALPIX_PAD_XY)
        dst = earth2grid.healpix.Grid(level=level, pixel_order=PixelOrder.RING)
        self.reorder_to_ring = earth2grid.get_regridder(src, dst).to(
            dtype=torch.float32, device=device
        )
        self.sht = SHTCUDA(
            nside=self.nside,
            lmax=self.lmax,
            mmax=self.mmax,
            quad_weights="ring",
        ).to(device)
        # One process, one constraint. The autograd Function cannot take the
        # module as a tensor argument.
        _DegreePowerFromRing.sht = self.sht

    def _degree_power(self, alm: torch.Tensor) -> torch.Tensor:
        amp = alm.real.square() + alm.imag.square()
        if amp.shape[-2] != self.lmax:
            raise ValueError(
                f"SHT degree axis is {amp.shape[-2]}, expected lmax={self.lmax}"
            )
        denom = self.two_ell_plus_one.to(dtype=amp.dtype)
        return amp.sum(dim=-1) / denom

    def _faces_to_ring(self, faces: torch.Tensor) -> torch.Tensor:
        """``[B, F, T, C, H, W]`` PAD_XY faces → RING ``[B, T, C, npix]``."""
        x = torch.movedim(faces, 1, -3)
        if x.shape[-3:] != (12, self.nside, self.nside):
            raise ValueError(
                "expected 12 faces of nside="
                f"{self.nside}, got spatial {tuple(x.shape[-3:])}"
            )
        x = x.reshape(*x.shape[:-3], -1)
        return self.reorder_to_ring(x.contiguous())

    def _to_alm(self, faces: torch.Tensor) -> torch.Tensor:
        """SHT of ``[B, F, T, C, H, W]`` → complex ``[B, T, C, l, m]``."""
        return self.sht(self._faces_to_ring(faces))

    def constraint_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        average_channels: bool = True,
    ) -> torch.Tensor:
        if self.sht is None or self.reorder_to_ring is None:
            raise RuntimeError(
                "SpectralPowerSoftConstraint.setup(trainer) must run on a "
                "CUDA device before the loss"
            )
        if prediction.shape != target.shape or prediction.ndim != 6:
            raise ValueError(
                "prediction and target must both be [B, F, T, C, H, W], "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        n_channels = prediction.shape[3]
        if n_channels != len(self.term_names()):
            raise ValueError(
                f"prediction has {n_channels} channels but term names "
                f"cover {len(self.term_names())}"
            )
        # Channel chunks keep the recomputed SHT off the full model graph.
        # A full-channel adjoint on the 24h rollout exhausts the 40 GB card.
        # The target spectrum is a constant, so its chunk carries no gradient.
        chunk = self.channel_chunk
        with torch.amp.autocast("cuda", enabled=False):
            target_parts = []
            pred_parts = []
            for c0 in range(0, n_channels, chunk):
                pred_sl = prediction[:, :, :, c0 : c0 + chunk].float()
                tgt_sl = target[:, :, :, c0 : c0 + chunk].detach().float()
                with torch.no_grad():
                    target_parts.append(
                        self._degree_power(self._to_alm(tgt_sl))
                    )
                pred_ring = self._faces_to_ring(pred_sl)
                pred_parts.append(
                    _DegreePowerFromRing.apply(pred_ring, self.two_ell_plus_one)
                )
            target_power = torch.cat(target_parts, dim=2)
            pred_power = torch.cat(pred_parts, dim=2)
            per_map = log_ell_relative_power_loss(
                pred_power,
                target_power,
                self.inv_ell,
                self.inv_ell_sum,
                floor_fraction=self.floor_fraction,
            )
            per_channel = per_map.mean(dim=(0, 1))
        if average_channels:
            return per_channel.mean()
        return per_channel
