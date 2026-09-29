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

"""Composable soft physical constraints for HEALPix DLWP training losses."""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Sequence

import earth2grid
import numpy as np
import torch
from cuhpx import SHTCUDA, iSHTCUDA
from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import RankZeroLoggingWrapper
from physicsnemo.metrics.climate.hydrostasy import (
    DifferentialHydrostaticBalanceConstraint,
    _load_topography,
)

logger = logging.getLogger(__name__)
if DistributedManager.is_initialized():
    logger = RankZeroLoggingWrapper(logger, DistributedManager())


def _error_tolerant(error: torch.Tensor) -> torch.Tensor:
    """Error-tolerant map used by hydrostasy soft losses: e / (1 + exp(1 - e))."""
    return error / (1.0 + torch.exp(1.0 - error))


def reduce_per_term_loss(
    terms: torch.Tensor, n_data_variables: int
) -> torch.Tensor:
    """Training scalar: ``sum(per-term losses) / n_data_variables``.

    Dividing by the data-channel count ``C`` (not the full term count) keeps the
    data contribution equal to a uniform-weight mean while each constraint term
    contributes ``1/C`` of its value. This replaces the older sum-of-block-means
    reduction (``mean(data) + mean(hydro) + dry_air + …``).
    """
    if n_data_variables < 1:
        raise ValueError(
            f"n_data_variables must be >= 1, got {n_data_variables}"
        )
    return terms.sum() / float(n_data_variables)


class SoftConstraint(torch.nn.Module):
    """Base class for soft constraints composed by :class:`LossWithSoftConstraints`.

    Subclasses set ``needs_input`` / ``needs_input_diagnostics`` and must match
    those flags in ``constraint_loss``:
    if ``needs_input`` is False, do not accept an ``input`` argument; if True,
    require ``input`` as a keyword-only argument. Same for
    ``needs_input_diagnostics`` / ``input_diagnostics``.

    ``physical_rmse`` returns the physical residual RMSE used by validation
    metrics (no alpha, no tolerant map, no loss weight).
    """

    needs_input: bool = False
    needs_input_diagnostics: bool = False

    def setup(self, trainer) -> None:
        """Move buffers to trainer device. Override as needed."""
        pass

    def constraint_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        average_channels: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError

    def physical_rmse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Physical residual RMSE (no alpha / tolerant map / loss weight)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement physical_rmse"
        )


class HydrostasySoftConstraint(SoftConstraint):
    """
    Soft differential hydrostatic-balance constraint.

    Logic mirrors :class:`~physicsnemo.metrics.climate.hydrostasy.LossWithHydrostasy`
    (Tv-only weights), so it can wrap an arbitrary data loss.
    """

    needs_input = False
    needs_input_diagnostics = False

    def __init__(
        self,
        hPa_levels: Sequence[float],
        channels: Sequence[str],
        weights: Sequence[float],
        alpha: Sequence[float],
        scaling: Dict[str, Dict[str, float]],
        dataset_path: str,
        surface_geopotential_name: str,
        surface_geopotential_mean: float = -597.7115478515625,
        surface_geopotential_std: float = 55658.21484375,
        convert_topography_to_meters: bool = True,
        R: float = 287,
        g0: float = 9.81,
        topography_masking: bool = True,
    ):
        super().__init__()
        self.g0 = g0
        self.convert_topography_to_meters = convert_topography_to_meters
        self.topography_masking = topography_masking
        self.loss_weights = torch.tensor(weights, dtype=torch.float32)

        self.pressure_levels = sorted(hPa_levels)
        self.z_pressure_levels = {
            channels.index(f"z{int(pl)}"): pl for pl in self.pressure_levels
        }
        self.T_pressure_levels = {
            channels.index(f"t{int(pl)}"): pl for pl in self.pressure_levels
        }
        self.q_pressure_levels = {
            channels.index(f"q{int(pl)}"): pl
            for pl in self.pressure_levels
            if f"q{int(pl)}" in channels
        }
        for i, pl in enumerate(self.pressure_levels):
            if f"q{int(pl)}" in channels:
                self.q_index_offset = i
                break
        else:
            raise ValueError("No humidity (q) channels found for hydrostasy soft constraint")

        self.z_constraint_pressure_levels = {
            i: pl for i, pl in enumerate(self.pressure_levels)
        }
        self.Tv_constraint_pressure_levels = {
            len(self.z_constraint_pressure_levels) + i: pl
            for i, pl in enumerate(self.pressure_levels)
        }

        self.z_mean = torch.Tensor(
            [scaling[f"z{int(pl)}"]["mean"] for pl in self.pressure_levels]
        ).reshape((1, 1, 1, -1, 1, 1))
        self.z_std = torch.Tensor(
            [scaling[f"z{int(pl)}"]["std"] for pl in self.pressure_levels]
        ).reshape((1, 1, 1, -1, 1, 1))
        self.T_mean = torch.Tensor(
            [scaling[f"t{int(pl)}"]["mean"] for pl in self.pressure_levels]
        ).reshape((1, 1, 1, -1, 1, 1))
        self.T_std = torch.Tensor(
            [scaling[f"t{int(pl)}"]["std"] for pl in self.pressure_levels]
        ).reshape((1, 1, 1, -1, 1, 1))
        self.q_mean = torch.Tensor(
            [
                scaling[f"q{int(pl)}"]["mean"]
                for pl in self.q_pressure_levels.values()
            ]
        ).reshape((1, 1, 1, -1, 1, 1))
        self.q_std = torch.Tensor(
            [
                scaling[f"q{int(pl)}"]["std"]
                for pl in self.q_pressure_levels.values()
            ]
        ).reshape((1, 1, 1, -1, 1, 1))

        if len(alpha) != len(hPa_levels) - 1:
            raise AssertionError(
                f"Incorrect number of alpha values. Expected len(hPa_levels)-1 "
                f"[{len(hPa_levels) - 1}], got {len(alpha)}"
            )
        if len(weights) != len(hPa_levels) - 1:
            raise AssertionError(
                f"Incorrect number of hydrostasy weights. Expected len(hPa_levels)-1 "
                f"[{len(hPa_levels) - 1}], got {len(weights)}"
            )
        self.alpha = torch.Tensor(alpha).reshape((1, 1, -1, 1, 1))
        self.Mw_ratio = 28.97 / 18.016 - 1.0

        self.constraint = DifferentialHydrostaticBalanceConstraint(
            self.z_constraint_pressure_levels,
            self.Tv_constraint_pressure_levels,
            0,
            len(self.z_pressure_levels),
            R,
            self.g0,
        )
        self.num_z_levels = len(self.z_constraint_pressure_levels)
        self.num_Tv_levels = len(self.Tv_constraint_pressure_levels)
        self.z_level_mapping = torch.tensor(list(self.z_pressure_levels.keys()))
        self.T_level_mapping = torch.tensor(list(self.T_pressure_levels.keys()))
        self.q_level_mapping = torch.tensor(list(self.q_pressure_levels.keys()))

        self.topography = _load_topography(
            dataset_path,
            surface_geopotential_name,
            surface_geopotential_mean,
            surface_geopotential_std,
            self.g0,
            self.convert_topography_to_meters,
        )
        if self.topography.min() < -1000.0 or self.topography.max() > 10000.0:
            raise ValueError("Topography values fall outside realistic range!")

    def setup(self, trainer) -> None:
        if len(self.z_pressure_levels) - 1 != len(self.loss_weights):
            raise ValueError(
                "Length of loss_weights is not one less than number of pressure levels!"
            )
        device = trainer.device
        self.loss_weights = self.loss_weights.to(device=device)
        self.z_mean = self.z_mean.to(device=device)
        self.z_std = self.z_std.to(device=device)
        self.T_mean = self.T_mean.to(device=device)
        self.T_std = self.T_std.to(device=device)
        self.q_mean = self.q_mean.to(device=device)
        self.q_std = self.q_std.to(device=device)
        self.alpha = self.alpha.to(device=device)
        self.z_level_mapping = self.z_level_mapping.to(device=device)
        self.T_level_mapping = self.T_level_mapping.to(device=device)
        self.q_level_mapping = self.q_level_mapping.to(device=device)
        self.topography = self.topography.to(device=device)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Scale to physical units and virtual temperature. Shape [N, F, B, C, H, W]."""
        N, F, B, C, H, W = x.shape
        C_scaled = self.num_z_levels + self.num_Tv_levels
        x_scaled = torch.zeros(
            (N, F, B, C_scaled, H, W),
            device=x.device,
            dtype=torch.float,
        )
        x_scaled[:, :, :, : self.num_z_levels, :, :] = (
            x[:, :, :, self.z_level_mapping, :, :] * self.z_std + self.z_mean
        ) / self.g0
        x_scaled[:, :, :, self.num_z_levels :, :, :] = (
            x[:, :, :, self.T_level_mapping, :, :] * self.T_std + self.T_mean
        )
        x_scaled[
            :,
            :,
            :,
            (self.num_z_levels + self.q_index_offset) :,
            :,
            :,
        ] *= 1.0 + self.Mw_ratio * (
            x[:, :, :, self.q_level_mapping, :, :] * self.q_std + self.q_mean
        )
        x_scaled = x_scaled.transpose(1, 2)
        return x_scaled.reshape((-1, F, C_scaled, H, W))

    def constraint_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        average_channels: bool = True,
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            if prediction.ndim != 6:
                raise AssertionError("Expected predictions to have 6 dimensions")

            x = self.scale(prediction)
            Tv_avg, Tv_model_avg = self.constraint(x)
            Tv_error = ((Tv_avg - Tv_model_avg) / self.alpha) ** 2
            if self.topography_masking:
                Tv_error[x[:, :, 1 : self.num_z_levels, :, :] < self.topography] = 0.0

            Tv_loss = self.loss_weights * _error_tolerant(Tv_error).mean(
                dim=(0, 1, 3, 4)
            )
            if average_channels:
                return torch.mean(Tv_loss)
            return Tv_loss

    def physical_rmse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Per-layer RMSE of virtual-temperature imbalance in K.

        Underground columns are masked when ``topography_masking`` is True.
        No alpha, tolerant map, or loss weight is applied.
        """
        del target, kwargs
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            if prediction.ndim != 6:
                raise AssertionError("Expected predictions to have 6 dimensions")
            x = self.scale(prediction)
            Tv_avg, Tv_model_avg = self.constraint(x)
            Tv_error = Tv_avg - Tv_model_avg
            n_layers = Tv_error.shape[2]
            if self.topography_masking:
                valid = x[:, :, 1 : self.num_z_levels, :, :] >= self.topography
            else:
                valid = torch.ones_like(Tv_error, dtype=torch.bool)
            rmse = []
            for layer in range(n_layers):
                err = Tv_error[:, :, layer, :, :]
                mask = valid[:, :, layer, :, :]
                if mask.any():
                    rmse.append(torch.sqrt((err[mask] ** 2).mean()))
                else:
                    rmse.append(
                        torch.zeros((), device=prediction.device, dtype=prediction.dtype)
                    )
            return torch.stack(rmse)


class DryAirMassSoftConstraint(SoftConstraint):
    """
    Soft dry-air-mass conservation constraint.

    Penalizes step-to-step changes in the global-mean dry surface pressure
    ``sp_dry = sp - g0 * tcwv``. The IC (last input time) anchors the first
    predicted timestep; later terms compare consecutive predicted timesteps.

    ``channels`` indexes ``prediction`` (typically ``output_variables``).
    ``input_channels`` indexes prognostic ``input`` (typically
    ``input_variables``). Fields required at the IC that are absent from
    ``input_channels`` are read from ``input_diagnostics`` (output-only
    channels at input times from the datapipe).
    """

    needs_input = True
    _REQUIRED_IC_FIELDS = ("sp", "tcwv")

    def __init__(
        self,
        channels: Sequence[str],
        scaling: Dict[str, Dict[str, float]],
        weight: float = 0.001,
        alpha: float = 0.383431,
        g0: float = 9.81,
        input_channels: Optional[Sequence[str]] = None,
        diagnostic_channels: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.channels = list(channels)
        self.input_channels = (
            list(input_channels) if input_channels is not None else list(channels)
        )
        self.sp_channel_index = self.channels.index("sp")
        self.tcwv_channel_index = self.channels.index("tcwv")
        self.weight = float(weight)
        self.g0 = float(g0)

        # IC field -> index in prognostic input, or None if diagnostic-only.
        self._ic_input_index: Dict[str, Optional[int]] = {}
        missing_from_input: list[str] = []
        for name in self._REQUIRED_IC_FIELDS:
            if name in self.input_channels:
                self._ic_input_index[name] = self.input_channels.index(name)
            else:
                self._ic_input_index[name] = None
                missing_from_input.append(name)

        self.needs_input_diagnostics = len(missing_from_input) > 0
        # diagnostic_channels: order of the IC-diagnostics tensor from the
        # datapipe (output_variables \\ input_variables). Resolved in setup()
        # from the trainer when not passed explicitly.
        if diagnostic_channels is not None:
            self.diagnostic_channels = list(diagnostic_channels)
        elif self.needs_input_diagnostics:
            # Fallback: output-only names that appear in the prediction list.
            input_set = set(self.input_channels)
            self.diagnostic_channels = [
                c for c in self.channels if c not in input_set
            ]
        else:
            self.diagnostic_channels = []

        self._ic_diag_index: Dict[str, int] = {}
        for name in missing_from_input:
            if name not in self.diagnostic_channels:
                raise ValueError(
                    f"DryAirMassSoftConstraint requires IC field '{name}' in "
                    f"input_channels or diagnostic_channels; got "
                    f"input_channels={self.input_channels!r}, "
                    f"diagnostic_channels={self.diagnostic_channels!r}."
                )
            self._ic_diag_index[name] = self.diagnostic_channels.index(name)

        self.register_buffer(
            "ps_mean",
            torch.tensor(scaling["sp"]["mean"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ps_std",
            torch.tensor(scaling["sp"]["std"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tcwv_mean",
            torch.tensor(scaling["tcwv"]["mean"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tcwv_std",
            torch.tensor(scaling["tcwv"]["std"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
            persistent=False,
        )

    def setup(self, trainer) -> None:
        device = trainer.device
        self.ps_mean = self.ps_mean.to(device=device)
        self.ps_std = self.ps_std.to(device=device)
        self.tcwv_mean = self.tcwv_mean.to(device=device)
        self.tcwv_std = self.tcwv_std.to(device=device)
        self.alpha = self.alpha.to(device=device)
        # Prefer datamodule channel order when the trainer exposes it.
        if self.needs_input_diagnostics:
            diag_vars = getattr(trainer, "ic_diagnostic_variables", None)
            if diag_vars is None:
                dm = getattr(trainer, "data_module", None)
                diag_vars = getattr(dm, "ic_diagnostic_variables", None) if dm else None
            if diag_vars is not None:
                self.diagnostic_channels = list(diag_vars)
                for name in self._REQUIRED_IC_FIELDS:
                    if self._ic_input_index[name] is None:
                        if name not in self.diagnostic_channels:
                            raise ValueError(
                                f"IC diagnostic field '{name}' not in "
                                f"trainer.ic_diagnostic_variables="
                                f"{self.diagnostic_channels!r}."
                            )
                        self._ic_diag_index[name] = self.diagnostic_channels.index(
                            name
                        )

    def _global_mean_sp_dry_from_channels(
        self,
        sp: torch.Tensor,
        tcwv: torch.Tensor,
    ) -> torch.Tensor:
        """Global-mean dry surface pressure from normalized ``sp`` / ``tcwv`` slices."""
        sp = sp * self.ps_std + self.ps_mean
        tcwv = tcwv * self.tcwv_std + self.tcwv_mean
        sp_dry = sp - self.g0 * tcwv
        return sp_dry.mean(dim=(1, 4, 5), keepdim=True)

    def _global_mean_sp_dry(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Global-mean dry surface pressure from a prediction-layout tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Shape ``[B, F, T, C, H, W]`` (normalized), channel order ``channels``.
        """
        sp = tensor[
            :, :, :, self.sp_channel_index : self.sp_channel_index + 1, :, :
        ]
        tcwv = tensor[
            :, :, :, self.tcwv_channel_index : self.tcwv_channel_index + 1, :, :
        ]
        return self._global_mean_sp_dry_from_channels(sp, tcwv)

    def _ic_field(
        self,
        name: str,
        input: torch.Tensor,
        input_diagnostics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Last-time slice of an IC field from prognostic input or diagnostics."""
        in_idx = self._ic_input_index[name]
        if in_idx is not None:
            return input[:, :, -1:, in_idx : in_idx + 1, :, :]
        if input_diagnostics is None:
            raise ValueError(
                f"DryAirMassSoftConstraint requires input_diagnostics for IC "
                f"field '{name}' (not present in prognostic input)."
            )
        diag_idx = self._ic_diag_index[name]
        return input_diagnostics[:, :, -1:, diag_idx : diag_idx + 1, :, :]

    def _global_mean_sp_dry_ic(
        self,
        input: torch.Tensor,
        input_diagnostics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        sp = self._ic_field("sp", input, input_diagnostics)
        tcwv = self._ic_field("tcwv", input, input_diagnostics)
        return self._global_mean_sp_dry_from_channels(sp, tcwv)

    def constraint_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        input: torch.Tensor,
        input_diagnostics: Optional[torch.Tensor] = None,
        average_channels: bool = True,
    ) -> torch.Tensor:
        if input is None:
            raise ValueError(
                "DryAirMassSoftConstraint requires prognostic input "
                "(pass input=inputs[0] from the trainer)."
            )
        if self.needs_input_diagnostics and input_diagnostics is None:
            raise ValueError(
                "DryAirMassSoftConstraint requires input_diagnostics because "
                "sp and/or tcwv are not in prognostic input_channels "
                "(enable data.return_ic_diagnostics and pass the third batch "
                "element from the trainer)."
            )
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            input = input.float()
            if input_diagnostics is not None:
                input_diagnostics = input_diagnostics.float()
            if prediction.ndim != 6:
                raise AssertionError("Expected predictions to have 6 dimensions")

            sp_dry_pred = self._global_mean_sp_dry(prediction)
            sp_dry_input = self._global_mean_sp_dry_ic(input, input_diagnostics)

            # [B, F, T, 1, 1, 1] transitions: t=0 vs IC, then consecutive preds
            T = sp_dry_pred.shape[2]
            prev = sp_dry_input
            losses = []
            for t in range(T):
                curr = sp_dry_pred[:, :, t : t + 1]
                violation = curr - prev
                error = (violation / self.alpha) ** 2
                losses.append(_error_tolerant(error))
                prev = curr

            stacked = torch.cat(losses, dim=2)
            loss = self.weight * stacked.mean()
            if average_channels:
                return loss
            return loss.reshape(())

    def physical_rmse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor = None,
        *,
        input: torch.Tensor,
        input_diagnostics: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """RMSE of global-mean dry-surface-pressure change in Pa.

        Same IC / consecutive-prediction transitions as ``constraint_loss``,
        without alpha, tolerant map, or loss weight.
        """
        del target, kwargs
        if input is None:
            raise ValueError(
                "DryAirMassSoftConstraint.physical_rmse requires prognostic input."
            )
        if self.needs_input_diagnostics and input_diagnostics is None:
            raise ValueError(
                "DryAirMassSoftConstraint.physical_rmse requires input_diagnostics."
            )
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            input = input.float()
            if input_diagnostics is not None:
                input_diagnostics = input_diagnostics.float()
            if prediction.ndim != 6:
                raise AssertionError("Expected predictions to have 6 dimensions")

            sp_dry_pred = self._global_mean_sp_dry(prediction)
            sp_dry_input = self._global_mean_sp_dry_ic(input, input_diagnostics)
            T = sp_dry_pred.shape[2]
            prev = sp_dry_input
            residuals = []
            for t in range(T):
                curr = sp_dry_pred[:, :, t : t + 1]
                residuals.append(curr - prev)
                prev = curr
            stacked = torch.cat(residuals, dim=2)
            return torch.sqrt((stacked ** 2).mean())


# ---------------------------------------------------------------------------
# Axial angular momentum (AAM)
# ---------------------------------------------------------------------------

# Earth radius [m], sidereal rotation rate [rad/s].
_EARTH_RADIUS_M = 6.371e6
_EARTH_OMEGA = 7.292115e-5


def layer_delta_p_pa(
    p_levels_hpa: torch.Tensor,
    sp_pa: torch.Tensor,
) -> torch.Tensor:
    """Layer pressure thickness between consecutive levels, clamped at ``p_s``.

    Parameters
    ----------
    p_levels_hpa:
        Ascending pressure levels in hPa, shape ``[L]``.
    sp_pa:
        Surface pressure in Pa, broadcastable to
        ``[…, 1, H, W]`` (channel dim before spatial).

    Returns
    -------
    torch.Tensor
        Thickness in Pa with shape ``[…, L-1, H, W]``. Levels with
        ``p > p_s`` contribute nothing; a layer that straddles ``p_s`` keeps
        only the above-ground thickness.
    """
    if p_levels_hpa.dim() != 1:
        raise ValueError(f"p_levels_hpa must be 1-D, got {tuple(p_levels_hpa.shape)}")
    if p_levels_hpa.numel() < 2:
        raise ValueError("Need at least two pressure levels for layer Δp")
    sp_hpa = sp_pa / 100.0
    # Insert a length-(L-1) axis just before the last two spatial dims.
    # sp_pa is [..., 1, H, W] → compare against p_top/p_bot [L-1] broadcast.
    p_top = p_levels_hpa[:-1].to(dtype=sp_pa.dtype, device=sp_pa.device)
    p_bot = p_levels_hpa[1:].to(dtype=sp_pa.dtype, device=sp_pa.device)
    # Reshape to [1, 1, ..., L-1, 1, 1] matching sp's ndim with channel→layer.
    view_shape = [1] * (sp_hpa.dim() - 3) + [-1, 1, 1]
    p_top = p_top.view(*view_shape)
    p_bot = p_bot.view(*view_shape)
    p_top_c = torch.minimum(p_top, sp_hpa)
    p_bot_c = torch.minimum(p_bot, sp_hpa)
    return (p_bot_c - p_top_c).clamp(min=0.0) * 100.0


def relative_angular_momentum(
    u_layer: torch.Tensor,
    delta_p_pa: torch.Tensor,
    cos_phi: torch.Tensor,
    d_omega: float,
    a: float = _EARTH_RADIUS_M,
    g: float = 9.81,
) -> torch.Tensor:
    """Global relative AAM ``M_r = (a³/g) Σ u cosφ Δp ΔΩ``.

    ``u_layer`` / ``delta_p_pa`` share shape ``[B, F, T, L, H, W]``;
    ``cos_phi`` broadcasts over batch/time/layer. Returns ``[B, T]``.
    """
    return (a ** 3 / g) * (u_layer * cos_phi * delta_p_pa * d_omega).sum(
        dim=(1, 3, 4, 5)
    )


def earth_angular_momentum(
    sp_pa: torch.Tensor,
    cos_phi: torch.Tensor,
    d_omega: float,
    a: float = _EARTH_RADIUS_M,
    omega: float = _EARTH_OMEGA,
    g: float = 9.81,
) -> torch.Tensor:
    """Global Earth (mass) AAM ``M_Ω = (Ω a⁴/g) Σ p_s cos²φ ΔΩ``.

    ``sp_pa`` shape ``[B, F, T, 1, H, W]``; returns ``[B, T]``.
    """
    return (omega * a ** 4 / g) * (sp_pa * (cos_phi ** 2) * d_omega).sum(
        dim=(1, 3, 4, 5)
    )


class AxialAngularMomentumSoftConstraint(SoftConstraint):
    """
    Soft axial angular-momentum budget constraint.

    Residual per native 6 h interval::

        (M(t+Δt) - M(t)) / Δt - (T_f + T_m)

    with ``M = M_r + M_Ω``, friction torque from eastward turbulent and
    gravity-wave surface stresses, and mountain torque from ``sp`` and surface
    geopotential (``∂h/∂λ`` via cuHPX SHT). Underground pressure layers are
    excluded via :func:`layer_delta_p_pa`.
    """

    needs_input = True
    _STRESS_NAMES = ("avg_iews-6h", "avg_iegwss-6h")
    _REQUIRED_IC_FIELDS_BASE = ("sp",)

    def __init__(
        self,
        hPa_levels: Sequence[float],
        channels: Sequence[str],
        scaling: Dict[str, Dict[str, float]],
        dataset_path: str,
        surface_geopotential_name: str,
        weight: float = 1.0,
        alpha: float = 1.0,
        g0: float = 9.81,
        a: float = _EARTH_RADIUS_M,
        omega: float = _EARTH_OMEGA,
        delta_t_seconds: float = 6.0 * 3600.0,
        nside: Optional[int] = None,
        surface_geopotential_mean: float = -597.7115478515625,
        surface_geopotential_std: float = 55658.21484375,
        convert_topography_to_meters: bool = True,
        input_channels: Optional[Sequence[str]] = None,
        diagnostic_channels: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.channels = list(channels)
        self.input_channels = (
            list(input_channels) if input_channels is not None else list(channels)
        )
        self.weight = float(weight)
        self.g0 = float(g0)
        self.a = float(a)
        self.omega = float(omega)
        self.delta_t_seconds = float(delta_t_seconds)

        self.pressure_levels = sorted(float(p) for p in hPa_levels)
        if len(self.pressure_levels) < 2:
            raise ValueError("Axial AAM requires at least two pressure levels")
        # LongTensor buffer so advanced indexing is CUDA-graph safe (no host
        # torch.tensor(..., device=) during capture/replay).
        u_idxs = []
        for pl in self.pressure_levels:
            name = f"u{int(pl)}"
            if name not in self.channels:
                raise ValueError(
                    f"Axial AAM requires channel {name!r} in channels={self.channels!r}"
                )
            u_idxs.append(self.channels.index(name))
        self.register_buffer(
            "u_channel_indices",
            torch.tensor(u_idxs, dtype=torch.long),
            persistent=False,
        )
        for stress in self._STRESS_NAMES:
            if stress not in self.channels:
                raise ValueError(
                    f"Axial AAM requires diagnostic {stress!r} in channels"
                )
        if "sp" not in self.channels:
            raise ValueError("Axial AAM requires 'sp' in prediction channels")

        self.sp_channel_index = self.channels.index("sp")
        self.iews_channel_index = self.channels.index("avg_iews-6h")
        self.iegwss_channel_index = self.channels.index("avg_iegwss-6h")

        # IC: sp and each u level from prognostic input or diagnostics.
        self._ic_field_names = ["sp"] + [f"u{int(pl)}" for pl in self.pressure_levels]
        self._ic_input_index: Dict[str, Optional[int]] = {}
        missing_from_input: list[str] = []
        for name in self._ic_field_names:
            if name in self.input_channels:
                self._ic_input_index[name] = self.input_channels.index(name)
            else:
                self._ic_input_index[name] = None
                missing_from_input.append(name)
        self.needs_input_diagnostics = len(missing_from_input) > 0
        if diagnostic_channels is not None:
            self.diagnostic_channels = list(diagnostic_channels)
        elif self.needs_input_diagnostics:
            input_set = set(self.input_channels)
            self.diagnostic_channels = [c for c in self.channels if c not in input_set]
        else:
            self.diagnostic_channels = []
        self._ic_diag_index: Dict[str, int] = {}
        for name in missing_from_input:
            if name not in self.diagnostic_channels:
                raise ValueError(
                    f"AxialAngularMomentumSoftConstraint requires IC field "
                    f"{name!r} in input_channels or diagnostic_channels."
                )
            self._ic_diag_index[name] = self.diagnostic_channels.index(name)

        def _stat(name: str, key: str) -> float:
            if name not in scaling:
                raise KeyError(f"scaling missing entry for {name!r}")
            return float(scaling[name][key])

        self.register_buffer(
            "p_levels_hpa",
            torch.tensor(self.pressure_levels, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "u_mean",
            torch.tensor(
                [_stat(f"u{int(pl)}", "mean") for pl in self.pressure_levels],
                dtype=torch.float32,
            ).view(1, 1, 1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "u_std",
            torch.tensor(
                [_stat(f"u{int(pl)}", "std") for pl in self.pressure_levels],
                dtype=torch.float32,
            ).view(1, 1, 1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "ps_mean",
            torch.tensor(_stat("sp", "mean"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ps_std",
            torch.tensor(_stat("sp", "std"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "iews_mean",
            torch.tensor(_stat("avg_iews-6h", "mean"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "iews_std",
            torch.tensor(_stat("avg_iews-6h", "std"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "iegwss_mean",
            torch.tensor(_stat("avg_iegwss-6h", "mean"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "iegwss_std",
            torch.tensor(_stat("avg_iegwss-6h", "std"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
            persistent=False,
        )

        # Surface geopotential → height [m] on HEALPix faces (same loader as hydro).
        # _load_topography returns [1, F, 1, H, W] in meters when convert_*=True.
        topo = _load_topography(
            dataset_path,
            surface_geopotential_name,
            surface_geopotential_mean,
            surface_geopotential_std,
            self.g0,
            convert_topography_to_meters,
        )
        if topo.dim() != 5 or topo.shape[0] != 1 or topo.shape[2] != 1:
            raise ValueError(
                f"Expected topography [1, F, 1, H, W], got {tuple(topo.shape)}"
            )
        faces, hh, ww = int(topo.shape[1]), int(topo.shape[3]), int(topo.shape[4])
        if faces != 12 or hh != ww or (hh & (hh - 1)) != 0:
            raise ValueError(
                f"Topography must be HEALPix faces with power-of-two nside, "
                f"got F,H,W=({faces},{hh},{ww})"
            )
        self.nside = int(nside) if nside is not None else int(hh)
        if self.nside != hh:
            raise ValueError(
                f"nside={self.nside} does not match topography nside={hh}"
            )
        self.n_pix = 12 * self.nside * self.nside
        self.d_omega = 4.0 * math.pi / float(self.n_pix)

        grid_xy = earth2grid.healpix.Grid(
            level=int(np.log2(self.nside)), pixel_order=HEALPIX_PAD_XY
        )
        cos_phi = torch.tensor(
            np.cos(np.deg2rad(grid_xy.lat)).reshape(12, self.nside, self.nside),
            dtype=torch.float32,
        )
        self.register_buffer(
            "cos_phi",
            cos_phi.view(1, 12, 1, 1, self.nside, self.nside),
            persistent=False,
        )

        # ∂h/∂λ via SHT (built in setup on the trainer device).
        self.lmax = 3 * self.nside - 1
        self.mmax = self.lmax
        self.sht = None
        self.isht = None
        self.reorder_to_ring = None
        self.reorder_from_ring = None
        self.register_buffer(
            "dh_dlambda",
            torch.zeros(1, 12, 1, 1, self.nside, self.nside),
            persistent=False,
        )
        # Face layout [F, H, W] for the SHT path.
        self.register_buffer(
            "_surface_height",
            topo[0, :, 0, :, :].contiguous(),
            persistent=False,
        )

    def setup(self, trainer) -> None:
        device = trainer.device
        self.p_levels_hpa = self.p_levels_hpa.to(device=device)
        self.u_channel_indices = self.u_channel_indices.to(device=device)
        self.u_mean = self.u_mean.to(device=device)
        self.u_std = self.u_std.to(device=device)
        self.ps_mean = self.ps_mean.to(device=device)
        self.ps_std = self.ps_std.to(device=device)
        self.iews_mean = self.iews_mean.to(device=device)
        self.iews_std = self.iews_std.to(device=device)
        self.iegwss_mean = self.iegwss_mean.to(device=device)
        self.iegwss_std = self.iegwss_std.to(device=device)
        self.alpha = self.alpha.to(device=device)
        self.cos_phi = self.cos_phi.to(device=device)
        self._surface_height = self._surface_height.to(device=device)

        if self.needs_input_diagnostics:
            diag_vars = getattr(trainer, "ic_diagnostic_variables", None)
            if diag_vars is None:
                dm = getattr(trainer, "data_module", None)
                diag_vars = getattr(dm, "ic_diagnostic_variables", None) if dm else None
            if diag_vars is not None:
                self.diagnostic_channels = list(diag_vars)
                for name in self._ic_field_names:
                    if self._ic_input_index[name] is None:
                        if name not in self.diagnostic_channels:
                            raise ValueError(
                                f"IC diagnostic field '{name}' not in "
                                f"trainer.ic_diagnostic_variables="
                                f"{self.diagnostic_channels!r}."
                            )
                        self._ic_diag_index[name] = self.diagnostic_channels.index(
                            name
                        )

        # Longitude derivative of surface height (constant); requires CUDA SHT.
        if device.type != "cuda":
            # CPU unit tests exercise Δp / M_r only; mountain torque is zero.
            self.dh_dlambda = torch.zeros_like(self.cos_phi, device=device)
            return

        self.sht = SHTCUDA(
            nside=self.nside,
            lmax=self.lmax,
            mmax=self.mmax,
            quad_weights="ring",
        ).to(device)
        self.isht = iSHTCUDA(
            nside=self.nside,
            lmax=self.lmax,
            mmax=self.mmax,
            quad_weights="ring",
        ).to(device)
        src = earth2grid.healpix.Grid(
            level=int(np.log2(self.nside)), pixel_order=HEALPIX_PAD_XY
        )
        tar = earth2grid.healpix.Grid(
            level=int(np.log2(self.nside)), pixel_order=PixelOrder.RING
        )
        self.reorder_to_ring = earth2grid.get_regridder(src, tar).to(
            dtype=torch.float32, device=device
        )
        self.reorder_from_ring = earth2grid.get_regridder(tar, src).to(
            dtype=torch.float32, device=device
        )
        h = self._surface_height.reshape(1, -1).contiguous()
        h_ring = self.reorder_to_ring(h)
        coeffs = self.sht(h_ring)
        m = torch.arange(self.mmax, device=device, dtype=torch.float32)
        d_coeffs = coeffs * (1j * m[None, None, :])
        dh_ring = self.isht(d_coeffs).real
        dh = self.reorder_from_ring(dh_ring).reshape(
            1, 12, 1, 1, self.nside, self.nside
        )
        self.dh_dlambda = dh.to(device=device)

    def _ic_field(
        self,
        name: str,
        input: torch.Tensor,
        input_diagnostics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        in_idx = self._ic_input_index[name]
        if in_idx is not None:
            return input[:, :, -1:, in_idx : in_idx + 1, :, :]
        if input_diagnostics is None:
            raise ValueError(
                f"AxialAngularMomentumSoftConstraint requires input_diagnostics "
                f"for IC field '{name}'."
            )
        diag_idx = self._ic_diag_index[name]
        return input_diagnostics[:, :, -1:, diag_idx : diag_idx + 1, :, :]

    def _denorm_u(self, u_norm: torch.Tensor) -> torch.Tensor:
        return u_norm * self.u_std + self.u_mean

    def _denorm_sp(self, sp_norm: torch.Tensor) -> torch.Tensor:
        return sp_norm * self.ps_std + self.ps_mean

    def _denorm_stress(self, iews_norm: torch.Tensor, iegwss_norm: torch.Tensor):
        iews = iews_norm * self.iews_std + self.iews_mean
        iegwss = iegwss_norm * self.iegwss_std + self.iegwss_mean
        return iews, iegwss

    def _u_layers_from_levels(self, u_levels: torch.Tensor) -> torch.Tensor:
        """Average adjacent u levels → layer centers ``[B,F,T,L-1,H,W]``."""
        return 0.5 * (u_levels[:, :, :, :-1] + u_levels[:, :, :, 1:])

    def _total_aam_from_physical(
        self,
        u_levels: torch.Tensor,
        sp_pa: torch.Tensor,
    ) -> torch.Tensor:
        """``M_r + M_Ω`` with shape ``[B, T]`` from physical u / sp tensors."""
        dp = layer_delta_p_pa(self.p_levels_hpa, sp_pa)
        u_layer = self._u_layers_from_levels(u_levels)
        m_r = relative_angular_momentum(
            u_layer, dp, self.cos_phi, self.d_omega, a=self.a, g=self.g0
        )
        m_omega = earth_angular_momentum(
            sp_pa,
            self.cos_phi,
            self.d_omega,
            a=self.a,
            omega=self.omega,
            g=self.g0,
        )
        return m_r + m_omega

    def _aam_from_prediction(self, tensor: torch.Tensor) -> torch.Tensor:
        u_norm = tensor[:, :, :, self.u_channel_indices, :, :]
        u = self._denorm_u(u_norm)
        sp = self._denorm_sp(
            tensor[:, :, :, self.sp_channel_index : self.sp_channel_index + 1, :, :]
        )
        return self._total_aam_from_physical(u, sp)

    def _aam_from_ic(
        self,
        input: torch.Tensor,
        input_diagnostics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        sp = self._denorm_sp(self._ic_field("sp", input, input_diagnostics))
        u_list = []
        for pl in self.pressure_levels:
            u_list.append(self._ic_field(f"u{int(pl)}", input, input_diagnostics))
        u_norm = torch.cat(u_list, dim=3)
        u = self._denorm_u(u_norm)
        return self._total_aam_from_physical(u, sp)

    def _torques_from_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Friction + mountain torque at each predicted time; shape ``[B, T]``."""
        sp = self._denorm_sp(
            prediction[
                :, :, :, self.sp_channel_index : self.sp_channel_index + 1, :, :
            ]
        )
        iews_n = prediction[
            :, :, :, self.iews_channel_index : self.iews_channel_index + 1, :, :
        ]
        iegwss_n = prediction[
            :,
            :,
            :,
            self.iegwss_channel_index : self.iegwss_channel_index + 1,
            :,
            :,
        ]
        iews, iegwss = self._denorm_stress(iews_n, iegwss_n)
        # ERA5 eastward stress is on the surface → torque on atmosphere is −a³ Σ τ cosφ ΔΩ.
        t_f = -(self.a ** 3) * ((iews + iegwss) * self.cos_phi * self.d_omega).sum(
            dim=(1, 3, 4, 5)
        )
        t_m = (self.a ** 2) * (
            sp * self.dh_dlambda * self.d_omega
        ).sum(dim=(1, 3, 4, 5))
        return t_f + t_m

    def _budget_residuals(
        self,
        prediction: torch.Tensor,
        input: torch.Tensor,
        input_diagnostics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Budget residual ``(ΔM/Δt) - (T_f+T_m)`` per transition; ``[B, T]``."""
        m_pred = self._aam_from_prediction(prediction)
        m_ic = self._aam_from_ic(input, input_diagnostics)[:, -1]
        torques = self._torques_from_prediction(prediction)
        T = m_pred.shape[1]
        prev = m_ic
        residuals = []
        for t in range(T):
            curr = m_pred[:, t]
            dmdt = (curr - prev) / self.delta_t_seconds
            residuals.append(dmdt - torques[:, t])
            prev = curr
        return torch.stack(residuals, dim=1)

    def constraint_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        input: torch.Tensor,
        input_diagnostics: Optional[torch.Tensor] = None,
        average_channels: bool = True,
    ) -> torch.Tensor:
        del target
        if input is None:
            raise ValueError(
                "AxialAngularMomentumSoftConstraint requires prognostic input."
            )
        if self.needs_input_diagnostics and input_diagnostics is None:
            raise ValueError(
                "AxialAngularMomentumSoftConstraint requires input_diagnostics."
            )
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            input = input.float()
            if input_diagnostics is not None:
                input_diagnostics = input_diagnostics.float()
            if prediction.ndim != 6:
                raise AssertionError("Expected predictions to have 6 dimensions")
            residual = self._budget_residuals(prediction, input, input_diagnostics)
            error = (residual / self.alpha) ** 2
            loss = self.weight * _error_tolerant(error).mean()
            if average_channels:
                return loss
            return loss.reshape(())

    def physical_rmse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor = None,
        *,
        input: torch.Tensor,
        input_diagnostics: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """RMSE of the global AAM budget residual in N·m.

        Same transitions and underground Δp masking as the loss; no alpha,
        tolerant map, or loss weight.
        """
        del target, kwargs
        if input is None:
            raise ValueError(
                "AxialAngularMomentumSoftConstraint.physical_rmse requires input."
            )
        if self.needs_input_diagnostics and input_diagnostics is None:
            raise ValueError(
                "AxialAngularMomentumSoftConstraint.physical_rmse requires "
                "input_diagnostics."
            )
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            input = input.float()
            if input_diagnostics is not None:
                input_diagnostics = input_diagnostics.float()
            residual = self._budget_residuals(prediction, input, input_diagnostics)
            # A realistic budget residual is ~1e19 N·m. Squaring that in fp32
            # overflows to inf (3e19**2 > 3.4e38), so the RMSE is accumulated
            # in fp64 and only the result is cast back.
            residual64 = residual.double()
            return torch.sqrt((residual64 * residual64).mean()).to(
                dtype=torch.float32
            )


class LossWithSoftConstraints(torch.nn.Module):
    """
    Loss-agnostic wrapper that adds zero or more soft constraints to a data loss.

    Compatible with the DLWP trainer ``setup`` / ``average_channels`` conventions.

    When ``average_channels=True``, the training scalar is
    ``sum(per-term losses) / n_data_variables`` (see :func:`reduce_per_term_loss`),
    not the older sum of block means. ``average_channels=False`` returns the
    concatenated per-term vector ``[data channels | constraint terms…]``.

    ``needs_input`` is True iff any child soft constraint requires prognostic
    input. ``needs_input_diagnostics`` is True iff any child needs IC diagnostic
    channels. Trainers should gate passing tensors on those flags.
    """

    def __init__(
        self,
        data_loss: torch.nn.Module,
        constraints: Optional[Sequence[SoftConstraint]] = None,
    ):
        super().__init__()
        self.data_loss = data_loss
        if constraints is None:
            constraints = []
        self.constraints = torch.nn.ModuleList(list(constraints))
        # Instance attributes (not properties) so trainers can use getattr(..., False).
        self.needs_input = any(
            getattr(c, "needs_input", False) for c in self.constraints
        )
        self.needs_input_diagnostics = any(
            getattr(c, "needs_input_diagnostics", False) for c in self.constraints
        )
        # Bind a forward whose signature matches the needs_* flags.
        if self.needs_input or self.needs_input_diagnostics:
            self.forward = self._forward_with_input
        else:
            self.forward = self._forward_without_input

    def setup(self, trainer) -> None:
        if hasattr(self.data_loss, "setup"):
            self.data_loss.setup(trainer)
        for constraint in self.constraints:
            constraint.setup(trainer)

    def _constraint_parts(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        average_channels: bool,
        input: Optional[torch.Tensor],
        input_diagnostics: Optional[torch.Tensor],
    ) -> list[torch.Tensor]:
        parts = []
        for constraint in self.constraints:
            kwargs = {"average_channels": average_channels}
            if constraint.needs_input:
                kwargs["input"] = input
            if getattr(constraint, "needs_input_diagnostics", False):
                kwargs["input_diagnostics"] = input_diagnostics
            parts.append(
                constraint.constraint_loss(prediction, target, **kwargs)
            )
        return parts

    def _combine(
        self,
        data: torch.Tensor,
        parts: list[torch.Tensor],
        average_channels: bool,
    ) -> torch.Tensor:
        """Concatenate per-term losses; optionally reduce with ``sum / C``.

        ``data`` must be the per-channel data-loss vector (``average_channels``
        already False at the call site). ``C = data.shape[0]``.
        """
        pieces = [data if data.dim() > 0 else data.unsqueeze(0)]
        for p in parts:
            pieces.append(p if p.dim() > 0 else p.unsqueeze(0))
        terms = torch.cat(pieces) if parts else pieces[0]
        if not average_channels:
            return terms
        n_data = int(pieces[0].shape[0])
        return reduce_per_term_loss(terms, n_data)

    def _forward_without_input(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        average_channels: bool = True,
    ) -> torch.Tensor:
        # Always collect the per-term vector so the scalar path can use sum/C.
        data = self.data_loss(
            prediction,
            target,
            average_channels=False,
        )
        parts = self._constraint_parts(
            prediction,
            target,
            average_channels=False,
            input=None,
            input_diagnostics=None,
        )
        return self._combine(data, parts, average_channels)

    def _forward_with_input(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        average_channels: bool = True,
        input: Optional[torch.Tensor] = None,
        input_diagnostics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.needs_input and input is None:
            raise ValueError(
                "LossWithSoftConstraints requires prognostic input because a "
                "soft constraint has needs_input=True (pass input=inputs[0])."
            )
        if self.needs_input_diagnostics and input_diagnostics is None:
            raise ValueError(
                "LossWithSoftConstraints requires input_diagnostics because a "
                "soft constraint has needs_input_diagnostics=True (pass the "
                "IC diagnostics tensor from the datapipe)."
            )
        data = self.data_loss(
            prediction,
            target,
            average_channels=False,
        )
        parts = self._constraint_parts(
            prediction,
            target,
            average_channels=False,
            input=input,
            input_diagnostics=input_diagnostics,
        )
        return self._combine(data, parts, average_channels)
