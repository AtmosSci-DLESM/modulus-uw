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
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import xarray as xr

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import RankZeroLoggingWrapper
from physicsnemo.metrics.climate.hydrostasy import DifferentialHydrostaticBalanceConstraint

logger = logging.getLogger(__name__)
if DistributedManager.is_initialized():
    logger = RankZeroLoggingWrapper(logger, DistributedManager())


def _error_tolerant(error: torch.Tensor) -> torch.Tensor:
    """Error-tolerant map used by hydrostasy soft losses: e / (1 + exp(1 - e))."""
    return error / (1.0 + torch.exp(1.0 - error))


class SoftConstraint(torch.nn.Module):
    """Base class for soft constraints composed by :class:`LossWithSoftConstraints`.

    Subclasses set ``needs_input`` and must match that flag in ``constraint_loss``:
    if ``needs_input`` is False, do not accept an ``input`` argument; if True,
    require ``input`` as a keyword-only argument.
    """

    needs_input: bool = False

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


class HydrostasySoftConstraint(SoftConstraint):
    """
    Soft differential hydrostatic-balance constraint.

    Logic mirrors :class:`~physicsnemo.metrics.climate.hydrostasy.LossWithHydrostasy`
    (Tv-only weights), so it can wrap an arbitrary data loss.
    """

    needs_input = False

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

        ds = xr.open_zarr(dataset_path)
        self.topography = (
            surface_geopotential_std
            * ds["constants"].sel(channel_c=surface_geopotential_name).values
            + surface_geopotential_mean
        )
        if self.convert_topography_to_meters:
            self.topography /= self.g0
        self.topography = torch.tensor(
            self.topography[np.newaxis, :, np.newaxis, :, :], dtype=torch.float
        )
        logger.info(
            f"Min/Max topography (m): {self.topography.min()}/{self.topography.max()}"
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


class DryAirMassSoftConstraint(SoftConstraint):
    """
    Soft dry-air-mass conservation constraint.

    Penalizes step-to-step changes in the global-mean dry surface pressure
    ``sp_dry = sp - g0 * tcwv``. The prognostic input anchors the first
    predicted timestep; later terms compare consecutive predicted timesteps.
    """

    needs_input = True

    def __init__(
        self,
        channels: Sequence[str],
        scaling: Dict[str, Dict[str, float]],
        weight: float = 0.001,
        alpha: float = 0.383431,
        g0: float = 9.81,
    ):
        super().__init__()
        self.channels = list(channels)
        self.sp_channel_index = self.channels.index("sp")
        self.tcwv_channel_index = self.channels.index("tcwv")
        self.weight = float(weight)
        self.g0 = float(g0)

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

    def _global_mean_sp_dry(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Global-mean dry surface pressure.

        Parameters
        ----------
        tensor : torch.Tensor
            Shape ``[B, F, T, C, H, W]`` (normalized).

        Returns
        -------
        torch.Tensor
            Shape ``[B, F, T, 1, 1, 1]`` in Pa.
        """
        sp = tensor[
            :, :, :, self.sp_channel_index : self.sp_channel_index + 1, :, :
        ]
        sp = sp * self.ps_std + self.ps_mean
        tcwv = tensor[
            :, :, :, self.tcwv_channel_index : self.tcwv_channel_index + 1, :, :
        ]
        tcwv = tcwv * self.tcwv_std + self.tcwv_mean
        sp_dry = sp - self.g0 * tcwv
        return sp_dry.mean(dim=(1, 4, 5), keepdim=True)

    def constraint_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        input: torch.Tensor,
        average_channels: bool = True,
    ) -> torch.Tensor:
        if input is None:
            raise ValueError(
                "DryAirMassSoftConstraint requires prognostic input "
                "(pass input=inputs[0] from the trainer)."
            )
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            input = input.float()
            if prediction.ndim != 6:
                raise AssertionError("Expected predictions to have 6 dimensions")

            sp_dry_pred = self._global_mean_sp_dry(prediction)
            sp_dry_input = self._global_mean_sp_dry(input[:, :, -1:])

            # [B, F, T, 1, 1, 1] transitions: t=0 vs input, then consecutive preds
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


class LossWithSoftConstraints(torch.nn.Module):
    """
    Loss-agnostic wrapper that adds zero or more soft constraints to a data loss.

    Compatible with the DLWP trainer ``setup`` / ``average_channels`` conventions.

    ``needs_input`` is True iff any child soft constraint requires prognostic
    input. When True, ``forward`` accepts required keyword ``input=``; when
    False, ``forward`` does not accept ``input``. Trainers should gate passing
    prognostic tensors solely on ``needs_input``.
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
        # Instance attribute (not a property) so trainers can use getattr(..., False).
        self.needs_input = any(
            getattr(c, "needs_input", False) for c in self.constraints
        )
        # Bind a forward whose signature matches needs_input: include ``input``
        # only when at least one soft constraint requires it.
        if self.needs_input:
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
    ) -> list[torch.Tensor]:
        parts = []
        for constraint in self.constraints:
            if constraint.needs_input:
                parts.append(
                    constraint.constraint_loss(
                        prediction,
                        target,
                        input=input,
                        average_channels=average_channels,
                    )
                )
            else:
                parts.append(
                    constraint.constraint_loss(
                        prediction,
                        target,
                        average_channels=average_channels,
                    )
                )
        return parts

    def _combine(
        self,
        data: torch.Tensor,
        parts: list[torch.Tensor],
        average_channels: bool,
    ) -> torch.Tensor:
        if not parts:
            return data
        if average_channels:
            total = data
            for p in parts:
                total = total + p
            return total
        pieces = [data]
        for p in parts:
            pieces.append(p if p.dim() > 0 else p.unsqueeze(0))
        return torch.cat(pieces)

    def _forward_without_input(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        average_channels: bool = True,
    ) -> torch.Tensor:
        data = self.data_loss(
            prediction,
            target,
            average_channels=average_channels,
        )
        parts = self._constraint_parts(
            prediction, target, average_channels, input=None
        )
        return self._combine(data, parts, average_channels)

    def _forward_with_input(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        average_channels: bool = True,
        input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input is None:
            raise ValueError(
                "LossWithSoftConstraints requires prognostic input because a "
                "soft constraint has needs_input=True (pass input=inputs[0])."
            )
        data = self.data_loss(
            prediction,
            target,
            average_channels=average_channels,
        )
        parts = self._constraint_parts(
            prediction, target, average_channels, input=input
        )
        return self._combine(data, parts, average_channels)
