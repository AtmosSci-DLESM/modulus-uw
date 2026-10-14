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


"""Deterministic two-lead ocean blocks and shared normalized state policy.

Layouts are [batch, time, channel, face, height, width]. Native RecUNet residuals
add to corresponding histories, each representing a 192-hour block advance.
Masks represent ocean validity, not land fraction; target validity may be dynamic.
"""

from collections.abc import Mapping

import torch
from torch import nn

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module


def _names(names):
    names = tuple(str(name) for name in names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Channel names must be nonempty and unique")
    return names


class _LinearGapStencil(nn.Module):
    """Cache one exact linear interpolation pass for a fixed missing domain."""

    def __init__(self, valid, axis, roll=0):
        super().__init__()
        import numpy as np

        self.axis, self.roll = axis, roll
        valid = np.roll(valid, roll, axis=axis)
        lines = np.moveaxis(valid, axis, -1)
        length = lines.shape[-1]
        positions = np.broadcast_to(np.arange(length), lines.shape)
        left = np.maximum.accumulate(np.where(lines, positions, -1), axis=-1)
        right = np.minimum.accumulate(
            np.where(lines, positions, length)[..., ::-1], axis=-1
        )[..., ::-1]
        fill = ~lines & (left >= 0) & (right < length)
        grid = np.arange(valid.size).reshape(valid.shape)
        index_lines = np.moveaxis(grid, axis, -1)
        target = index_lines[fill]
        lower = np.take_along_axis(index_lines, left.clip(0, length - 1), -1)[fill]
        upper = np.take_along_axis(index_lines, right.clip(0, length - 1), -1)[fill]
        alpha = (positions[fill] - left[fill]) / (right[fill] - left[fill])
        self.register_buffer("target", torch.as_tensor(target, dtype=torch.long))
        self.register_buffer("lower", torch.as_tensor(lower, dtype=torch.long))
        self.register_buffer("upper", torch.as_tensor(upper, dtype=torch.long))
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float64))
        completed = np.moveaxis(lines | fill, -1, axis)
        self.completed = np.roll(completed, -roll, axis=axis)

    def forward(self, field):
        axis = field.ndim - 2 + self.axis
        rolled = torch.roll(field, self.roll, dims=axis)
        flat = rolled.reshape(*field.shape[:-2], -1)
        output = flat.clone()
        alpha = self.alpha.to(dtype=field.dtype)
        output[..., self.target] = (
            flat[..., self.lower] * (1 - alpha) + flat[..., self.upper] * alpha
        )
        return torch.roll(output.reshape_as(field), -self.roll, dims=axis)


class TripleInterpSST(nn.Module):
    """Differentiable ETL SST interpolation on a fixed selected ocean domain.

    Reuse the exact earth2grid HPX_PAD_XY->equiangular latlon->HPX operators.
    The three cached linear-gap passes reproduce add_derived_channels.py's
    longitude, half-rolled longitude and half-rolled latitude interpolation
    (use_coordinate=False, no extrapolation). Defaults match ETL 721x1440.
    Regridding and stencil buffers persist in checkpoints. Only current ocean
    values are used; supplied land never affects interpolated values.

    This is exact for the selected mask, which may differ from the ETL's lost
    native NaN mask. Stored land equality must be measured separately.
    """

    def __init__(self, mask, nlat=721, nlon=1440):
        super().__init__()
        import earth2grid
        import numpy as np

        mask = torch.as_tensor(mask, dtype=torch.bool).detach().cpu()
        if mask.ndim != 3 or mask.shape[0] != 12 or mask.shape[1] != mask.shape[2]:
            raise ValueError("SST mask must have shape [12,nside,nside]")
        width = mask.shape[-1]
        if width < 2 or width & (width - 1):
            raise ValueError("SST nside must be a power of two")
        if nlat < 2 or nlon < 2 or not mask.any():
            raise ValueError(
                "Interpolation needs ocean cells and valid latlon dimensions"
            )
        self.width, self.nlat, self.nlon = width, nlat, nlon
        hpx = earth2grid.healpix.Grid(
            level=width.bit_length() - 1,
            pixel_order=earth2grid.healpix.HEALPIX_PAD_XY,
        )
        latlon = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
        self.regrid = earth2grid.get_regridder(hpx, latlon)
        self.regrid_back = earth2grid.get_regridder(latlon, hpx)
        self.register_buffer("mask", mask)
        with torch.no_grad():
            probe = torch.where(mask, 1.0, float("nan")).double().reshape(-1)
            valid = torch.isfinite(self.regrid(probe)).numpy()
        self.register_buffer("latlon_valid", torch.as_tensor(valid))
        passes = []
        for axis, roll in ((1, 0), (1, nlon // 2), (0, nlat // 2)):
            stencil = _LinearGapStencil(valid, axis, roll)
            valid = stencil.completed
            del stencil.completed
            passes.append(stencil)
        self.passes = nn.Sequential(*passes)
        if not np.all(valid):
            raise ValueError(
                "TripleInterp cannot fill selected ocean domain completely"
            )

    def get_extra_state(self):
        """Persist the ETL grid, stencil and missing-domain contract."""
        return {
            "width": self.width,
            "nlat": self.nlat,
            "nlon": self.nlon,
            "algorithm": "etl-triple-linear-selected-domain-v1",
        }

    def set_extra_state(self, state):
        """Reject inconsistent interpolation configuration on resume."""
        if state != self.get_extra_state():
            raise ValueError("Checkpoint SST interpolation configuration differs")

    def forward(self, ocean):
        if ocean.shape[-3:] != self.mask.shape:
            raise ValueError("SST field and interpolation mask must match")
        if (self.mask & ~torch.isfinite(ocean)).any():
            raise FloatingPointError("Nonfinite valid ocean SST before interpolation")
        dtype = ocean.dtype
        # Match ETL double arithmetic; prohibit autocast altering gap blends.
        with torch.autocast(device_type=ocean.device.type, enabled=False):
            clean = torch.where(self.mask, ocean, 0.0).double()
            latlon = self.regrid(clean.reshape(*ocean.shape[:-3], -1))
            latlon = torch.where(self.latlon_valid, latlon, 0.0)
            filled = self.regrid_back(self.passes(latlon)).reshape_as(clean)
            result = torch.where(self.mask, clean, filled)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite SST after TripleInterp")
        return result.to(dtype)


def _safe_normalized_boundary(mean, std, boundary, lower):
    """Round a normalized bound inward for both float32 and float64 decoding."""
    mean64, std64 = mean.detach().double(), std.detach().double()
    value = ((boundary - mean64) / std64).float()
    direction = torch.full_like(value, float("inf") if lower else -float("inf"))
    # A float32 multiply may round outward even when exact decoding is safe.
    # Validate both decoders, rather than assuming one rounded quotient suffices.
    for _ in range(8):
        decoded32 = value * std.float() + mean.float()
        decoded64 = value.double() * std64 + mean64
        outside = (
            ((decoded32 < boundary) | (decoded64 < boundary))
            if lower
            else ((decoded32 > boundary) | (decoded64 > boundary))
        )
        if not outside.any():
            return value
        value = torch.where(outside, torch.nextafter(value, direction), value)
    raise FloatingPointError("Cannot represent safe normalized physical boundary")


class OceanStatePolicy(nn.Module):
    """Process observations, reconstructed states and feedback identically.

    Invalid non-SST cells become normalized zero before any arithmetic. SST land
    is recomputed by the shared TripleInterpSST from current valid ocean values. Valid sic is
    clamped in physical units to [0,1], and valid mlotst to [0,infinity).
    Scaling, per-channel domains and channel order persist in state_dict.
    Bounds are inward-rounded for both float32 and float64 physical decode.
    strict=True rejects nonfinite valid predictions before processing.
    Nonfinite observed values are replaced with normalized zero; loss validity
    must independently exclude missing targets before their sanitization.
    """

    def __init__(self, channels, mean, std, mask, sst_nlat=721, sst_nlon=1440):
        super().__init__()
        self.channels = _names(channels)
        count = len(self.channels)
        mean = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
        std = torch.as_tensor(std, dtype=torch.float32).detach().clone()
        mask = torch.as_tensor(mask).detach().clone()
        if mean.shape != (count,) or std.shape != (count,):
            raise ValueError("mean/std must have shape [C]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Scaling must be finite")
        if (std <= 0).any():
            raise ValueError("Standard deviations must be positive")
        if mask.ndim == 3:
            mask = mask.unsqueeze(0).expand(count, -1, -1, -1).clone()
        if mask.ndim != 4 or mask.shape[:2] != (count, 12):
            raise ValueError("mask must be [C,12,H,W] or [12,H,W]")
        if mask.shape[-1] != mask.shape[-2] or mask.shape[-1] < 2:
            raise ValueError("HEALPix faces must be square with nside>=2")
        if not torch.isfinite(mask).all() or ((mask != 0) & (mask != 1)).any():
            raise ValueError("State mask must be boolean ocean validity")
        self.register_buffer("mean", mean.reshape(1, 1, count, 1, 1, 1))
        self.register_buffer("std", std.reshape(1, 1, count, 1, 1, 1))
        self.register_buffer("mask", mask.bool().reshape(1, 1, *mask.shape))
        self.sst_nlat, self.sst_nlon = sst_nlat, sst_nlon
        self.sst_interpolator = None
        if "sst" in self.channels:
            sst_mask = mask[self.channels.index("sst")].bool()
            if not sst_mask.all():
                self.sst_interpolator = TripleInterpSST(sst_mask, sst_nlat, sst_nlon)

    def get_extra_state(self):
        """Persist channel order and the shared policy contract."""
        return {
            "channels": self.channels,
            "policy": "sst-triple-interp-other-neutral-safe-bounds-v3",
            "sst_nlat": self.sst_nlat,
            "sst_nlon": self.sst_nlon,
        }

    def set_extra_state(self, state):
        """Reject checkpoints with incompatible channel order or policy."""
        if state != self.get_extra_state():
            raise ValueError("Checkpoint state policy/channel ordering differs")

    def forward(self, state, strict=False):
        if state.ndim != 6 or state.shape[2:] != self.mask.shape[2:]:
            raise ValueError("state must be [B,T,C,12,H,W] matching policy mask")
        if strict and (self.mask & ~torch.isfinite(state)).any():
            raise FloatingPointError(
                "Nonfinite valid ocean prediction before state policy"
            )
        # Autocast/low-precision observations must not quantize safe boundaries.
        if state.dtype in (torch.float16, torch.bfloat16):
            state = state.float()
        clean = torch.where(self.mask & torch.isfinite(state), state, 0.0)
        # Linear scaling makes normalized bounds equivalent to physical clamps.
        result = []
        for index, name in enumerate(self.channels):
            value = clean[:, :, index : index + 1]
            if name in ("sic", "mlotst"):
                mean = self.mean[:, :, index : index + 1]
                std = self.std[:, :, index : index + 1]
                lower = _safe_normalized_boundary(mean, std, 0.0, lower=True)
                value = torch.maximum(value, lower)
                if name == "sic":
                    upper = _safe_normalized_boundary(mean, std, 1.0, lower=False)
                    if (lower > upper).any():
                        raise FloatingPointError(
                            "SIC interval has no representable normalized value"
                        )
                    value = torch.minimum(value, upper)
            if name == "sst" and self.sst_interpolator is not None:
                mean = self.mean[:, :, index : index + 1]
                std = self.std[:, :, index : index + 1]
                filled = self.sst_interpolator((value * std + mean).squeeze(2))
                filled = (filled.unsqueeze(2) - mean) / std
                value = torch.where(self.mask[:, :, index : index + 1], value, filled)
            elif name != "sst":
                value = torch.where(self.mask[:, :, index : index + 1], value, 0.0)
            result.append(value)
        return torch.cat(result, dim=2)

    def to_physical(self, state):
        """Denormalize without changing validity; use mask for physical metrics."""
        return state * self.std + self.mean


class GlorysOceanModel(Module):
    """Compose the existing recurrent HEALPix UNet with ocean state processing.

    The native model node configures encoder, decoder and recurrent blocks.
    Native residual leads reference corresponding histories. RecUNet resets
    hidden state on every call; presteps=0 uses exactly the supplied histories.
    """

    def __init__(
        self,
        channels,
        forcing_channels,
        mean,
        std,
        mask,
        backbone_config,
        sst_nlat=721,
        sst_nlon=1440,
    ):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from physicsnemo.models.dlwp_healpix.HEALPixRecUNet import HEALPixRecUNet

        super().__init__(meta=ModelMetaData(name="GlorysOceanModel"))
        self.register_buffer("device_buffer", self.device_buffer, persistent=False)
        self.channels = _names(channels)
        self.forcing_channels = _names(forcing_channels)
        if OmegaConf.is_config(backbone_config):
            backbone_config = OmegaConf.to_container(backbone_config, resolve=True)
        # An archive must carry the full plain model node, not a live module.
        import json

        self.backbone_config = json.loads(json.dumps(backbone_config, allow_nan=False))
        self.state_policy = OceanStatePolicy(
            self.channels, mean, std, mask, sst_nlat, sst_nlon
        )
        self.register_buffer(
            "land_constant",
            (~self.state_policy.mask.any(dim=2))
            .float()
            .reshape(1, 12, 1, *self.state_policy.mask.shape[-2:]),
        )
        self.backbone = instantiate(self.backbone_config, _recursive_=False)
        if type(self.backbone) is not HEALPixRecUNet:
            raise ValueError("Ocean backbone must be the existing HEALPixRecUNet")
        self.backbone.register_buffer(
            "device_buffer", self.backbone.device_buffer, persistent=False
        )
        self._validate_ocean_contract()
        self.checkpoint_args()

    def _validate_ocean_contract(self):
        """Check only the interface required by ocean sampling and state policy.

        Native RecUNet and its configured modules own architecture, execution
        options and their validation. Presets select the verified configuration.
        """
        required = {
            "input_channels": len(self.channels),
            "output_channels": len(self.channels),
            "n_constants": 1,
            "decoder_input_channels": len(self.forcing_channels),
            "input_time_dim": 2,
            "output_time_dim": 2,
            "delta_t": 96,
            "presteps": 0,
            "residual_prediction": True,
            "couplings": [],
        }
        for name, value in required.items():
            if getattr(self.backbone, name) != value:
                raise ValueError(f"Ocean backbone requires {name}={value!r}")
        if self.state_policy.mask.shape[-2:] != (self.backbone.nside[0],) * 2:
            raise ValueError(
                "Ocean policy mask and native full-resolution nside differ"
            )

    def checkpoint_args(self):
        """Refresh JSON-ready full native constructor and ocean policy arguments."""
        import copy

        self._args = {
            "__name__": type(self).__name__,
            "__module__": type(self).__module__,
            "__args__": {
                "channels": list(self.channels),
                "forcing_channels": list(self.forcing_channels),
                "mean": self.state_policy.mean.detach().cpu().flatten().tolist(),
                "std": self.state_policy.std.detach().cpu().flatten().tolist(),
                "mask": self.state_policy.mask[0, 0].detach().cpu().tolist(),
                "backbone_config": copy.deepcopy(self.backbone_config),
                "sst_nlat": int(self.state_policy.sst_nlat),
                "sst_nlon": int(self.state_policy.sst_nlon),
            },
        }
        return self._args

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Require every native backbone tensor and scientific extra-state key."""
        expected, supplied = set(self.state_dict()), set(state_dict)
        if expected != supplied:
            raise RuntimeError(
                f"Ocean checkpoint state keys differ: missing={sorted(expected-supplied)}, "
                f"unexpected={sorted(supplied-expected)}"
            )
        result = super().load_state_dict(state_dict, strict=True, assign=assign)
        self.checkpoint_args()
        return result

    def save(self, file_name=None, verbose=False):
        """Save a canonical Module archive with full native architecture metadata."""
        self.checkpoint_args()
        return super().save(file_name=file_name, verbose=verbose)

    def get_extra_state(self):
        """Persist native model config and native corresponding-history residual semantics."""
        import copy

        return {
            "architecture": "existing-healpix-recunet-ocean-v1",
            "channels": self.channels,
            "forcing_channels": self.forcing_channels,
            "backbone_config": copy.deepcopy(self.backbone_config),
            "reference": "native-corresponding-history",
        }

    def set_extra_state(self, state):
        """Reject architecture or temporal drift, including old CNN checkpoints."""
        if state != self.get_extra_state():
            raise ValueError("Checkpoint model configuration differs")

    def forward(self, history, forcing):
        """Process observations, execute native RecUNet and constrain both leads."""
        history = self.state_policy(history)
        batch, steps, count, faces, height, width = history.shape
        expected = (batch, 2, len(self.forcing_channels), faces, height, width)
        if steps != 2 or count != len(self.channels) or forcing.shape != expected:
            raise ValueError("Expected two history states and two forcing windows")
        forcing = torch.where(torch.isfinite(forcing), forcing, 0.0)
        native = self.backbone(
            [
                history.permute(0, 3, 1, 2, 4, 5),
                forcing.permute(0, 3, 1, 2, 4, 5),
                self.land_constant[0],
            ]
        )
        states = native.permute(0, 2, 3, 1, 4, 5)
        if states.shape != history.shape:
            raise ValueError(
                "Native RecUNet returned an incompatible two-lead state shape"
            )
        return self.state_policy(states, strict=True)


def masked_channel_sums(prediction, target, mask):
    """Return normalized/physical squared-error numerators and valid denominators.

    Accept static [C,F,H,W] or dynamic [B,T,C,F,H,W] masks (broadcastable).
    Boolean validity or nonnegative fractional weights are supported. Nonfinite
    targets are excluded before subtraction, including their gradients. Valid
    nonfinite predictions remain visible as a failed metric. Reduce over batch,
    time, faces and pixels; sum these across batches/ranks BEFORE division.
    Accumulate at least float32 to avoid half-precision reduction overflow.
    """
    if prediction.shape != target.shape or target.ndim != 6:
        raise ValueError("prediction and target must match [B,T,C,F,H,W]")
    weights = torch.as_tensor(mask, device=target.device)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Mask weights must be finite and nonnegative")
    weights = torch.broadcast_to(weights, target.shape)
    valid = (weights > 0) & torch.isfinite(target)
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    safe_prediction = torch.where(valid, prediction, 0.0).to(dtype)
    safe_target = torch.where(valid, target, 0.0).to(dtype)
    weights = torch.where(valid, weights, 0).to(dtype)
    dimensions = (0, 1, 3, 4, 5)
    numerator = ((safe_prediction - safe_target).square() * weights).sum(dimensions)
    return numerator, weights.sum(dimensions)


def masked_channel_mse(prediction, target, mask):
    """Return per-channel MSE; absent channels are zero with zero denominator."""
    numerator, denominator = masked_channel_sums(prediction, target, mask)
    return numerator / torch.where(denominator > 0, denominator, 1)


class FixedInverseValidationWeights(nn.Module):
    """Equal weights, then frozen inverse unweighted normalized validation MSE.

    freeze(epoch, mse) uses a ONE-BASED completed epoch (default transition 10).
    Formula: inverse=1/max(mse,floor), balance=inverse/mean(inverse), then
    cap balance by max_weight and apply named manual multipliers. Manual factors
    apply only after balancing. All reference channels must have finite positive
    denominators externally and finite nonnegative MSE; do not calibrate missing
    channels. Freeze once; subsequent calls are no-ops. Persist buffers and exact
    configuration through state_dict; incompatible resume configuration fails.
    """

    def __init__(
        self,
        channels,
        equal_epochs=10,
        floor=1e-8,
        max_weight=100.0,
        manual_multipliers: Mapping[str, float] | None = None,
    ):
        super().__init__()
        self.channels = _names(channels)
        if equal_epochs < 1 or int(equal_epochs) != equal_epochs:
            raise ValueError("equal_epochs must be a positive integer")
        if not 0 < floor < float("inf") or not 0 < max_weight < float("inf"):
            raise ValueError("floor and max_weight must be finite and positive")
        manual = dict(manual_multipliers or {})
        if set(manual) - set(self.channels):
            raise ValueError("Unknown manual multiplier channels")
        multipliers = torch.tensor([manual.get(name, 1.0) for name in self.channels])
        if not torch.isfinite(multipliers).all() or (multipliers <= 0).any():
            raise ValueError("Manual multipliers must be finite and positive")
        self.equal_epochs, self.floor, self.max_weight = (
            int(equal_epochs),
            float(floor),
            float(max_weight),
        )
        self.manual_config = tuple(
            float(manual.get(name, 1.0)) for name in self.channels
        )
        self.register_buffer("manual", multipliers)
        self.register_buffer("weights", torch.ones(len(self.channels)))
        self.register_buffer("reference_mse", torch.zeros(len(self.channels)))
        self.register_buffer("frozen", torch.tensor(False))
        self.register_buffer("calibration_epoch", torch.tensor(-1))

    @property
    def phase(self):
        """Current persisted weighting phase."""
        return "fixed" if bool(self.frozen) else "equal"

    def get_extra_state(self):
        """Record the channel ordering, formula and schedule configuration."""
        return {
            "channels": self.channels,
            "equal_epochs": self.equal_epochs,
            "floor": self.floor,
            "max_weight": self.max_weight,
            "manual": self.manual_config,
            "formula": "mean-normalized-inverse-cap-then-manual-v1",
        }

    def set_extra_state(self, state):
        """Reject altered resume configuration before silent recalibration."""
        if state != self.get_extra_state():
            raise ValueError("Checkpoint balancing configuration differs")

    @torch.no_grad()
    def freeze(self, epoch, validation_mse):
        """Freeze once after the configured completed equal-weight epoch."""
        if bool(self.frozen) or epoch < self.equal_epochs:
            return False
        reference = torch.as_tensor(
            validation_mse, device=self.weights.device, dtype=self.weights.dtype
        )
        if (
            reference.shape != self.weights.shape
            or not torch.isfinite(reference).all()
            or (reference < 0).any()
        ):
            raise ValueError(
                "Calibration needs finite nonnegative MSE for every channel"
            )
        safe = reference.double().clamp_min(self.floor)
        relative_inverse = safe.min() / safe
        balanced = (relative_inverse / relative_inverse.mean()).clamp_max(
            self.max_weight
        )
        effective = (balanced * self.manual.double()).to(self.weights.dtype)
        if not torch.isfinite(effective).all() or (effective <= 0).any():
            raise ValueError("Effective balancing weights exceed numeric range")
        self.weights.copy_(effective)
        self.reference_mse.copy_(reference)
        self.calibration_epoch.fill_(epoch)
        self.frozen.fill_(True)
        return True

    def forward(self, channel_mse):
        return (channel_mse * self.weights).mean()
