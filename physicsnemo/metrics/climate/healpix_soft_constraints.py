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
from physicsnemo.metrics.climate.derived_surface_pressure import DerivedSurfacePressure
from physicsnemo.metrics.climate.hydrostasy import (
    DifferentialHydrostaticBalanceConstraint,
    _load_topography,
)

logger = logging.getLogger(__name__)
if DistributedManager.is_initialized():
    logger = RankZeroLoggingWrapper(logger, DistributedManager())


def _constraint_name(name: str) -> str:
    """Stable constraint id used to bind adaptive-loss scales."""
    text = str(name).strip()
    if not text or "/" in text:
        raise ValueError(
            "soft constraint name must be a non-empty string without '/'; "
            f"got {name!r}"
        )
    return text


def _hpa_token(level: float) -> str:
    value = float(level)
    if value.is_integer():
        return str(int(value))
    return format(value, "g")


def hydrostatic_interface_names(hPa_levels: Sequence[float]) -> list[str]:
    """Hydrostatic interface names in ``Tv_loss`` order.

    Layer ``i`` is the interface between pressure levels ``i`` and ``i + 1``
    after sorting ascending, the same order
    ``DifferentialHydrostaticBalanceConstraint`` writes into ``Tv_avg``.
    """
    levels = sorted(float(level) for level in hPa_levels)
    if len(levels) < 2:
        raise ValueError(
            "hydrostasy needs at least two pressure levels to name "
            f"interfaces, got {list(hPa_levels)}"
        )
    return [
        f"{_hpa_token(lo)}-{_hpa_token(hi)}"
        for lo, hi in zip(levels, levels[1:])
    ]


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

    def term_names(self) -> list[str]:
        """Names of the per-term losses this constraint appends, in order."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement term_names() so adaptive "
            "loss scales can bind to its terms"
        )

    def constraint_spec(self) -> tuple[str, list[str], object]:
        """``(name, term_names, relative_loss_scale)``. Term order matches the loss.

        ``relative_loss_scale`` is the value passed at construction (a float, a
        term-name mapping, or None). Adaptive loss weights read it from here.
        ``name`` is only the log tag.
        """
        if not hasattr(self, "name"):
            raise AttributeError(
                f"{type(self).__name__} has no name; pass name= when "
                "constructing it"
            )
        return (
            _constraint_name(self.name),
            [str(term) for term in self.term_names()],
            getattr(self, "relative_loss_scale", None),
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
        name: str = "hydro",
        relative_loss_scale=None,
    ):
        super().__init__()
        self.name = _constraint_name(name)
        self.relative_loss_scale = relative_loss_scale
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

    def term_names(self) -> list[str]:
        return hydrostatic_interface_names(self.pressure_levels)

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
        name: str = "dry_air",
        relative_loss_scale=None,
        derived_surface_pressure: Optional[Dict] = None,
    ):
        super().__init__()
        self.name = _constraint_name(name)
        self.relative_loss_scale = relative_loss_scale
        self.channels = list(channels)
        self.input_channels = (
            list(input_channels) if input_channels is not None else list(channels)
        )
        self.weight = float(weight)
        self.g0 = float(g0)
        self.derived = derived_surface_pressure is not None
        if self.derived:
            # sp and tcwv are not model channels: the dry surface pressure sp - g*tcwv is derived hydrostatically
            # from the predicted (and initial) z, t, q and the orography (see derived_surface_pressure.py).
            self.needs_input_diagnostics = False
            self.diagnostic_channels = []
            self._ic_input_index = {}
            self._ic_diag_index = {}
            self._dsp_pred = DerivedSurfacePressure(channels=self.channels, scaling=scaling, **derived_surface_pressure)
            self._dsp_in = DerivedSurfacePressure(channels=self.input_channels, scaling=scaling, **derived_surface_pressure)
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32), persistent=False)
            return
        self.sp_channel_index = self.channels.index("sp")
        self.tcwv_channel_index = self.channels.index("tcwv")

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
        if self.derived:
            self.alpha = self.alpha.to(device=device)
            self._dsp_pred.to(device)
            self._dsp_in.to(device)
            return
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
        if self.derived:
            return self._dsp_pred.sp_dry(tensor).mean(dim=(1, 4, 5), keepdim=True)
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
        if self.derived:
            return self._dsp_in.sp_dry(input[:, :, -1:]).mean(dim=(1, 4, 5), keepdim=True)
        sp = self._ic_field("sp", input, input_diagnostics)
        tcwv = self._ic_field("tcwv", input, input_diagnostics)
        return self._global_mean_sp_dry_from_channels(sp, tcwv)

    def term_names(self) -> list[str]:
        return [self.name]

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


class DryAirMassShiftSoftConstraint(SoftConstraint):
    """
    Penalty on the shift a hard :class:`DryAirMassConstraint` has to apply.

    The hard constraint subtracts the global-mean change in dry surface pressure
    ``sp - g0 * tcwv`` from ``sp`` every step. The corrected output therefore
    conserves dry mass whatever the network does, and neither the data loss nor
    :class:`DryAirMassSoftConstraint` sees the network's own mass tendency. This
    term penalizes that pre-correction tendency (per integration step, in Pa) with
    the same error-tolerant map, so the network learns to produce mass-neutral
    tendencies instead of relying on a uniform shift that cannot remove their
    latitudinal structure.

    The shifts are recorded by the model's ``DryAirMassConstraint`` during the
    forward pass; :meth:`setup` finds that module on ``trainer.model`` and turns
    recording on. It must run in the same step as the forward pass that produced
    the prediction (train, eval and CUDA-graph capture all do).
    """

    needs_input = False

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 0.383431,
        name: str = "dry_air_shift",
        relative_loss_scale=None,
    ):
        super().__init__()
        self.name = _constraint_name(name)
        self.relative_loss_scale = relative_loss_scale
        self.weight = float(weight)
        self.register_buffer(
            "alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
            persistent=False,
        )
        # Plain list: the hard constraint belongs to the model, not to this loss.
        self._hard: list = []

    def setup(self, trainer) -> None:
        from physicsnemo.models.dlwp_healpix_layers.healpix_constraints import (
            DryAirMassConstraint,
        )

        self.alpha = self.alpha.to(device=trainer.device)
        model = trainer.model
        while True:  # DDP / torch.compile wrappers
            inner = getattr(model, "module", None)
            if inner is None:
                inner = getattr(model, "_orig_mod", None)
            if inner is None:
                break
            model = inner
        hard = [
            c
            for c in (getattr(model, "constraints", None) or [])
            if isinstance(c, DryAirMassConstraint)
        ]
        if len(hard) != 1:
            raise ValueError(
                "DryAirMassShiftSoftConstraint needs exactly one "
                f"DryAirMassConstraint in model.constraints, found {len(hard)}."
            )
        self._hard = hard
        hard[0].enable_shift_record(model.integration_steps)

    def term_names(self) -> list[str]:
        return [self.name]

    def _shifts(self, prediction: torch.Tensor) -> torch.Tensor:
        """Recorded shifts of the latest model call, ``[B, 1, T, 1, 1, 1]`` in Pa."""
        if not self._hard or not self._hard[0].shift_record:
            raise RuntimeError(
                "No recorded dry-air shift: setup(trainer) has not run or the "
                "criterion was called before a model forward pass."
            )
        shifts = torch.cat(list(self._hard[0].shift_record), dim=2)
        if shifts.shape[2] != prediction.shape[2]:
            raise RuntimeError(
                f"Recorded dry-air shifts cover {shifts.shape[2]} time steps but "
                f"the prediction has {prediction.shape[2]}; the criterion has to "
                "be evaluated on the output of the latest model call."
            )
        return shifts

    def constraint_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        average_channels: bool = True,
    ) -> torch.Tensor:
        del target
        with torch.amp.autocast("cuda", enabled=False):
            shifts = self._shifts(prediction).float()
            loss = self.weight * _error_tolerant((shifts / self.alpha) ** 2).mean()
            return loss if average_channels else loss.reshape(())

    def physical_rmse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """RMS of the pre-correction global-mean dry-mass shift [Pa]: the quantity the loss penalizes, without alpha, map or weight."""
        del target, kwargs
        shifts = self._shifts(prediction).double()
        return torch.sqrt((shifts * shifts).mean()).to(dtype=torch.float32)


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
        name: str = "aam",
        relative_loss_scale=None,
        derived_surface_pressure: Optional[Dict] = None,
        mountain_torque_sign: float = 1.0,
        mountain_torque_anomaly: bool = False,
        reference_temperature: float = 288.0,
        reference_pressure: float = 101325.0,
    ):
        super().__init__()
        self.name = _constraint_name(name)
        self.relative_loss_scale = relative_loss_scale
        # +1 keeps the original mountain-torque term T_m = +a^2 sum(ps dh/dlambda dOmega). The torque on the atmosphere
        # is -a^2 int ps dh/dlambda dOmega; on ERA5 the budget residual is smaller with -1 (see calibrate_soft_constraint_alphas.py,
        # AAM_SIGN_CHECK=1), but the default stays +1 so existing configs and checkpoints train as before.
        self.mountain_torque_sign = float(mountain_torque_sign)
        # With ``mountain_torque_anomaly`` the torque uses ps - ps_ref(h), ps_ref = p_ref exp(-g h / (R T_ref)). Any function of h
        # alone gives zero torque analytically (its ring integral against dh/dlambda is a total derivative), but the large
        # hydrostatic part of ps does not cancel numerically on the grid: on ERA5 that artifact is ~ +2e19 N m, as large as the
        # physical torques. The anomaly form removes it and raises the ERA5 budget R^2 from 0.58 to 0.75.
        self.mountain_torque_anomaly = bool(mountain_torque_anomaly)
        self._ref_temperature = float(reference_temperature)
        self._ref_pressure = float(reference_pressure)
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
        # With ``derived_surface_pressure`` the surface pressure is derived hydrostatically from the predicted
        # (and initial) z, t, q and the orography, so the model needs no ``sp`` channel.
        self.derived = derived_surface_pressure is not None
        if self.derived:
            self._dsp_pred = DerivedSurfacePressure(channels=self.channels, scaling=scaling, **derived_surface_pressure)
            self._dsp_in = DerivedSurfacePressure(channels=self.input_channels, scaling=scaling, **derived_surface_pressure)
        else:
            if "sp" not in self.channels:
                raise ValueError("Axial AAM requires 'sp' in prediction channels")
            self.sp_channel_index = self.channels.index("sp")
        self.iews_channel_index = self.channels.index("avg_iews-6h")
        self.iegwss_channel_index = self.channels.index("avg_iegwss-6h")

        # IC: sp and each u level from prognostic input or diagnostics.
        self._ic_field_names = ([] if self.derived else ["sp"]) + [
            f"u{int(pl)}" for pl in self.pressure_levels
        ]
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
        # The sp scaling is only needed when sp is a model channel; the placeholders keep the buffers uniform.
        self.register_buffer(
            "ps_mean",
            torch.tensor(0.0 if self.derived else _stat("sp", "mean"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ps_std",
            torch.tensor(1.0 if self.derived else _stat("sp", "std"), dtype=torch.float32),
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
        self.register_buffer(
            "ps_ref",
            (
                self._ref_pressure
                * torch.exp(-self.g0 * topo / (287.04 * self._ref_temperature))
            ).view(1, 12, 1, 1, self.nside, self.nside),
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
        self.ps_ref = self.ps_ref.to(device=device)
        if self.derived:
            self._dsp_pred.to(device)
            self._dsp_in.to(device)

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

    def _sp_from_prediction(self, tensor: torch.Tensor) -> torch.Tensor:
        """Surface pressure [Pa] ``[B,F,T,1,H,W]``: derived from z/t/q, or the model's ``sp`` channel."""
        if self.derived:
            return self._dsp_pred.sp(tensor)
        return self._denorm_sp(
            tensor[:, :, :, self.sp_channel_index : self.sp_channel_index + 1, :, :]
        )

    def _aam_from_prediction(self, tensor: torch.Tensor) -> torch.Tensor:
        u_norm = tensor[:, :, :, self.u_channel_indices, :, :]
        u = self._denorm_u(u_norm)
        return self._total_aam_from_physical(u, self._sp_from_prediction(tensor))

    def _aam_from_ic(
        self,
        input: torch.Tensor,
        input_diagnostics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.derived:
            sp = self._dsp_in.sp(input[:, :, -1:])
        else:
            sp = self._denorm_sp(self._ic_field("sp", input, input_diagnostics))
        u_list = []
        for pl in self.pressure_levels:
            u_list.append(self._ic_field(f"u{int(pl)}", input, input_diagnostics))
        u_norm = torch.cat(u_list, dim=3)
        u = self._denorm_u(u_norm)
        return self._total_aam_from_physical(u, sp)

    def _torques_from_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Friction + mountain torque at each predicted time; shape ``[B, T]``."""
        sp = self._sp_from_prediction(prediction)
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
        sp_torque = sp - self.ps_ref if self.mountain_torque_anomaly else sp
        t_m = self.mountain_torque_sign * (self.a ** 2) * (
            sp_torque * self.dh_dlambda * self.d_omega
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

    def term_names(self) -> list[str]:
        return [self.name]

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

    Same helpers as in ``healpix_ring_mean_pchip``; kept here so the soft constraints do not depend on the
    ring-mean upsampling module.
    """
    key = np.round(lat, decimals=8)
    uniq = np.unique(key)
    order = np.argsort(uniq)
    rank = np.empty(uniq.size, dtype=np.int64)
    rank[order] = np.arange(uniq.size)
    inv = rank[np.searchsorted(uniq, key)]
    return inv.astype(np.int64), uniq[order]


class CapAxialAngularMomentumSoftConstraint(AxialAngularMomentumSoftConstraint):
    """
    Axial angular-momentum budget of every latitude cap bounded by a HEALPix ring boundary.

    For the cap poleward of a latitude circle ``phi_b`` the planetary angular momentum carried across the circle is
    ``Omega a^2 cos^2(phi_b)`` times the net mass flux into the cap, and exact mass continuity replaces that flux by
    the cap's surface-pressure tendency (no column-integrated meridional flux is needed). With ``M_r`` the cap's
    relative angular momentum the budget per transition is::

        d/dt [ M_r + (Omega a^4 / g) sum_cap ps (cos^2 phi - cos^2 phi_b) dOmega ] = F_rel(phi_b) + T_f + T_m

    ``F_rel = +-(2 pi a^2 cos^2 phi_b / g) [int u v dp]_{phi_b}`` is the relative-momentum flux through the
    boundary (zonal mean of the column ``u v``; + for northern caps, - for southern), ``T_f`` the surface-stress torque and ``T_m``
    the mountain torque (``mountain_torque_sign`` / ``mountain_torque_anomaly`` as in the parent class). The
    residual of every one of the ``4 nside - 2`` ring boundaries is compared with its own ``alpha`` (``alpha[b]``,
    ascending boundary latitude). The loss is the mean over batch, time and boundaries of
    ``phi((R_b / alpha_b)^2)``, so it is a single constraint term whatever the number of boundaries.

    With ``normalize_by_boundary`` a single scalar ``alpha`` in N/m replaces the per-boundary list:
    ``alpha_b = alpha * 2 pi a^2 cos^2(phi_b)``. ``R_b / (2 pi a^2 cos^2 phi_b)`` is the residual expressed as an error in the
    column meridional momentum flux ``[int u v dp] / g`` across the boundary (the form of ``F_rel``), so one tolerance applies to
    every latitude circle. The geometric factor (circumference times lever arm) accounts for most of the latitude dependence of
    the ERA5 residual: its per-boundary median falls within about a factor 2 of one value equatorward of 70 degrees, while it
    spans a factor ~80 in N m. Poleward of ~80 degrees the ERA5 residual is smaller than this scaling, so those tiny caps are
    held more loosely than with per-boundary alphas.

    All reductions run in float64: the residuals are ~1e18 N m while the angular momentum itself is ~1e28. Each
    state's column and ring reductions are recomputed in the backward pass (as in ``DerivedSurfacePressure``) so the
    training graph keeps only ``[B, 5, rings]`` tensors per state. Every index is a registered buffer, so the loss and
    its backward capture in a CUDA graph.
    """

    # Latitudes [deg] whose boundary RMS is reported as per-term values (N and S), for TensorBoard.
    REPORT_LATITUDES = (65.0, 50.0, 35.0, 20.0)

    def __init__(
        self,
        hPa_levels: Sequence[float],
        channels: Sequence[str],
        scaling: Dict[str, Dict[str, float]],
        dataset_path: str,
        surface_geopotential_name: str,
        alpha=1.0,
        name: str = "cap_aam",
        normalize_by_boundary: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("mountain_torque_sign", -1.0)
        kwargs.setdefault("mountain_torque_anomaly", True)
        super().__init__(
            hPa_levels=hPa_levels,
            channels=channels,
            scaling=scaling,
            dataset_path=dataset_path,
            surface_geopotential_name=surface_geopotential_name,
            alpha=1.0,
            name=name,
            **kwargs,
        )
        n = self.nside
        n_rings, n_bnd = 4 * n - 1, 4 * n - 2
        # v at every pressure level: prediction indices and initial condition.
        v_idxs = []
        for pl in self.pressure_levels:
            vname = f"v{int(pl)}"
            if vname not in self.channels:
                raise ValueError(f"Cap AAM requires channel {vname!r} in channels={self.channels!r}")
            v_idxs.append(self.channels.index(vname))
        self.register_buffer("v_channel_indices", torch.tensor(v_idxs, dtype=torch.long), persistent=False)
        if not self.derived:
            u_idxs = [self.channels.index(f"u{int(p)}") for p in self.pressure_levels]
            sel = u_idxs + v_idxs + [self.sp_channel_index, self.iews_channel_index, self.iegwss_channel_index]
            self.register_buffer("selected_channels", torch.tensor(sel, dtype=torch.long), persistent=False)
        for pl in self.pressure_levels:
            vname = f"v{int(pl)}"
            self._ic_field_names.append(vname)
            if vname in self.input_channels:
                self._ic_input_index[vname] = self.input_channels.index(vname)
            else:
                self._ic_input_index[vname] = None
                if not self.needs_input_diagnostics:
                    self.needs_input_diagnostics = True
                    input_set = set(self.input_channels)
                    self.diagnostic_channels = [c for c in self.channels if c not in input_set]
                if vname not in self.diagnostic_channels:
                    raise ValueError(f"Cap AAM requires IC field {vname!r} in input_channels or diagnostic_channels.")
                self._ic_diag_index[vname] = self.diagnostic_channels.index(vname)
        v_stats = [(float(scaling[f"v{int(p)}"]["mean"]), float(scaling[f"v{int(p)}"]["std"])) for p in self.pressure_levels]
        self.register_buffer("v_mean", torch.tensor([m for m, _ in v_stats], dtype=torch.float32).view(1, 1, 1, -1, 1, 1), persistent=False)
        self.register_buffer("v_std", torch.tensor([s_ for _, s_ in v_stats], dtype=torch.float32).view(1, 1, 1, -1, 1, 1), persistent=False)

        # Ring geometry. Ring ids ascend from the south pole; boundary b lies between rings b and b + 1.
        grid = earth2grid.healpix.Grid(level=int(np.log2(n)), pixel_order=HEALPIX_PAD_XY)
        ring, ring_lat = _ring_index(_geographic_latitude(grid))
        if ring_lat.size != n_rings:
            raise ValueError(f"Expected {n_rings} HEALPix rings at nside={n}, found {ring_lat.size}.")
        bnd_lat = 0.5 * (ring_lat[1:] + ring_lat[:-1])
        self.n_rings, self.n_boundaries = n_rings, n_bnd
        self.register_buffer("ring_id", torch.as_tensor(ring, dtype=torch.long), persistent=False)
        self.register_buffer("ring_cos", torch.as_tensor(np.cos(ring_lat), dtype=torch.float64), persistent=False)
        self.register_buffer("ring_count", torch.as_tensor(np.bincount(ring, minlength=n_rings), dtype=torch.float64), persistent=False)
        self.register_buffer("bnd_cos2", torch.as_tensor(np.cos(bnd_lat) ** 2, dtype=torch.float64), persistent=False)
        self.register_buffer("bnd_north", torch.as_tensor(bnd_lat > 0.0), persistent=False)
        self.register_buffer("bnd_lat_deg", torch.as_tensor(np.rad2deg(bnd_lat), dtype=torch.float64), persistent=False)
        report = [int(np.argmin(np.abs(np.rad2deg(bnd_lat) - s_ * lat))) for s_ in (-1.0, 1.0) for lat in self.REPORT_LATITUDES[::(1 if s_ < 0 else -1)]]
        self.register_buffer("report_index", torch.tensor(report, dtype=torch.long), persistent=False)
        self.report_names = [f"{'S' if s_ < 0 else 'N'}{int(lat)}" for s_ in (-1.0, 1.0) for lat in self.REPORT_LATITUDES[::(1 if s_ < 0 else -1)]]

        self.normalize_by_boundary = bool(normalize_by_boundary)
        if self.normalize_by_boundary:
            if not np.isscalar(alpha):
                raise ValueError("normalize_by_boundary=True takes one scalar alpha [N/m], not a per-boundary list.")
            alpha_v = float(alpha) * 2.0 * math.pi * self.a ** 2 * np.cos(bnd_lat) ** 2
        else:
            alpha_v = np.broadcast_to(np.asarray([float(a) for a in alpha] if not np.isscalar(alpha) else float(alpha), dtype=np.float64), (n_bnd,))
        self.alpha = torch.as_tensor(np.array(alpha_v), dtype=torch.float64)

    def setup(self, trainer) -> None:
        super().setup(trainer)
        device = trainer.device
        for name in ("v_channel_indices", "selected_channels", "v_mean", "v_std", "ring_id", "ring_cos", "ring_count", "bnd_cos2", "bnd_north", "bnd_lat_deg", "report_index"):
            if hasattr(self, name):
                setattr(self, name, getattr(self, name).to(device=device))
        self.alpha = self.alpha.to(device=device, dtype=torch.float64)

    # ------------------------------------------------------------------
    def _ring_sums(self, u, v, sp, tau) -> torch.Tensor:
        """Sums over the pixels of every ring of five fields ``[B, 5, rings]`` (float64), for one state.

        ``u``, ``v`` ``[B,F,1,L,H,W]`` physical; ``sp`` ``[B,F,1,1,H,W]``; ``tau`` the eastward surface stress
        ``[B,F,1,1,H,W]`` or ``None`` (initial condition). Fields: column ``int u dp``, column ``int u v dp``,
        ``ps``, ``tau``, mountain-torque integrand ``(ps - ps_ref) dh/dlambda`` (``ps`` without ``ps_ref`` when
        the anomaly form is off).
        """
        # Column products run in float32 (their ~1e-7 relative error is far below the budget noise); the ring sums,
        # which feed differences of ~1e28 totals, accumulate in float64.
        dp = layer_delta_p_pa(self.p_levels_hpa, sp)
        u_l = 0.5 * (u[:, :, :, :-1] + u[:, :, :, 1:])
        uv = u * v
        uv_l = 0.5 * (uv[:, :, :, :-1] + uv[:, :, :, 1:])
        sp_t = sp - self.ps_ref if self.mountain_torque_anomaly else sp
        fields = [
            (u_l * dp).sum(dim=3),
            (uv_l * dp).sum(dim=3),
            sp[:, :, :, 0],
            torch.zeros_like(sp[:, :, :, 0]) if tau is None else tau[:, :, :, 0],
            (sp_t * self.dh_dlambda)[:, :, :, 0],
        ]
        b = sp.shape[0]
        flat = torch.stack([f.reshape(b, 1, -1) for f in fields], dim=1).reshape(b, len(fields), -1).double()
        out = flat.new_zeros(b, len(fields), self.n_rings)
        return out.scatter_add_(2, self.ring_id.view(1, 1, -1).expand(b, len(fields), -1), flat)

    def _step_sums(self, step: torch.Tensor) -> torch.Tensor:
        """Ring sums of one predicted time step ``[B,F,1,C,H,W]`` (all channels)."""
        u = self._denorm_u(step[:, :, :, self.u_channel_indices])
        v = step[:, :, :, self.v_channel_indices] * self.v_std + self.v_mean
        iews, iegwss = self._denorm_stress(
            step[:, :, :, self.iews_channel_index : self.iews_channel_index + 1],
            step[:, :, :, self.iegwss_channel_index : self.iegwss_channel_index + 1],
        )
        return self._ring_sums(u, v, self._sp_from_prediction(step), iews + iegwss)

    def _step_sums_selected(self, x: torch.Tensor) -> torch.Tensor:
        """Ring sums of one time step from the gathered channels ``[B,F,1,2L+3,H,W]`` = u(L), v(L), sp, iews, iegwss."""
        n = len(self.pressure_levels)
        u = self._denorm_u(x[:, :, :, :n])
        v = x[:, :, :, n : 2 * n] * self.v_std + self.v_mean
        sp = self._denorm_sp(x[:, :, :, 2 * n : 2 * n + 1])
        iews, iegwss = self._denorm_stress(x[:, :, :, 2 * n + 1 : 2 * n + 2], x[:, :, :, 2 * n + 2 : 2 * n + 3])
        return self._ring_sums(u, v, sp, iews + iegwss)

    def _ic_sums(self, input, input_diagnostics) -> torch.Tensor:
        if self.derived:
            sp = self._dsp_in.sp(input[:, :, -1:])
        else:
            sp = self._denorm_sp(self._ic_field("sp", input, input_diagnostics))
        u = self._denorm_u(torch.cat([self._ic_field(f"u{int(p)}", input, input_diagnostics) for p in self.pressure_levels], dim=3))
        v = torch.cat([self._ic_field(f"v{int(p)}", input, input_diagnostics) for p in self.pressure_levels], dim=3) * self.v_std + self.v_mean
        return self._ring_sums(u, v, sp, None)

    def _cap_sums(self, x: torch.Tensor):
        """``[..., rings]`` -> (northern caps (rings > b), southern caps (rings <= b)), each ``[..., boundaries]``."""
        south = torch.cumsum(x, dim=-1)[..., :-1]
        north = torch.flip(torch.cumsum(torch.flip(x, dims=[-1]), dim=-1), dims=[-1])[..., 1:]
        return north, south

    def _ring_budget(self, prediction, input, input_diagnostics):
        """Per-ring budget terms ``(a, b, q)``, each ``[B, T, rings]`` (float64).

        ``a`` = d/dt(relative AAM + planetary AAM weighted by cos^2 phi_r) - T_f - T_m of each ring (summing it over all rings
        gives the global budget residual of the parent class); ``b`` = d/dt of the planetary AAM weight per unit
        cos^2 phi_b (the ring's surface-pressure tendency); ``q`` the zonal-mean column ``u v`` at the two time levels.
        """
        grad = torch.is_grad_enabled() and prediction.requires_grad
        states = [self._ic_sums(input, input_diagnostics)]
        if self.derived:
            steps, fn = [prediction[:, :, t : t + 1] for t in range(prediction.shape[2])], self._step_sums
        else:
            # Gather the 2L+3 channels the budget needs once and unbind over time: slicing the full prediction
            # tensor per time step would make autograd allocate a full-size zero gradient for every step.
            sel = prediction[:, :, :, self.selected_channels]
            steps, fn = [x.unsqueeze(2) for x in sel.unbind(dim=2)], self._step_sums_selected
        for step in steps:
            if grad:
                states.append(torch.utils.checkpoint.checkpoint(fn, step, use_reentrant=False))
            else:
                states.append(fn(step))
        s = torch.stack(states, dim=1)  # [B, T+1, 5, rings]
        a, g, om, dt, d_om = self.a, self.g0, self.omega, self.delta_t_seconds, self.d_omega
        cos_r = self.ring_cos
        d_u, d_p = s[:, 1:, 0] - s[:, :-1, 0], s[:, 1:, 2] - s[:, :-1, 2]
        t_f = -(a ** 3) * d_om * cos_r * s[:, 1:, 3]
        t_m = self.mountain_torque_sign * (a ** 2) * d_om * 0.5 * (s[:, :-1, 4] + s[:, 1:, 4])
        per_ring_a = (a ** 3 / g) * d_om * cos_r * d_u / dt + (om * a ** 4 / g) * d_om * cos_r ** 2 * d_p / dt - t_f - t_m
        per_ring_b = (om * a ** 4 / g) * d_om * d_p / dt
        return per_ring_a, per_ring_b, s[:, :, 1] / self.ring_count

    def _boundary_residuals(self, prediction, input, input_diagnostics) -> torch.Tensor:
        """Cap budget residual ``R_b`` [N m] per transition and ring boundary; ``[B, T, boundaries]`` float64."""
        per_ring_a, per_ring_b, q_mean = self._ring_budget(prediction, input, input_diagnostics)
        a_n, a_s = self._cap_sums(per_ring_a)
        b_n, b_s = self._cap_sums(per_ring_b)
        q_hat = 0.5 * (q_mean[..., 1:] + q_mean[..., :-1])  # boundary value from the two adjacent rings
        flux = (2.0 * math.pi * self.a ** 2 * self.bnd_cos2 / self.g0) * 0.5 * (q_hat[:, :-1] + q_hat[:, 1:])
        r_n = a_n - self.bnd_cos2 * b_n - flux
        r_s = a_s - self.bnd_cos2 * b_s + flux
        return torch.where(self.bnd_north, r_n, r_s)

    def term_names(self) -> list[str]:
        return [self.name]

    def constraint_loss(self, prediction, target, *, input, input_diagnostics=None, average_channels: bool = True) -> torch.Tensor:
        del target
        if input is None:
            raise ValueError("CapAxialAngularMomentumSoftConstraint requires prognostic input.")
        if self.needs_input_diagnostics and input_diagnostics is None:
            raise ValueError("CapAxialAngularMomentumSoftConstraint requires input_diagnostics.")
        with torch.amp.autocast("cuda", enabled=False):
            if prediction.ndim != 6:
                raise AssertionError("Expected predictions to have 6 dimensions")
            r = self._boundary_residuals(prediction.float(), input.float(), None if input_diagnostics is None else input_diagnostics.float())
            loss = self.weight * _error_tolerant((r / self.alpha) ** 2).mean()
            return loss.float() if average_channels else loss.float().reshape(())

    def physical_rmse(self, prediction, target=None, *, input, input_diagnostics=None, average_channels: bool = True, **kwargs) -> torch.Tensor:
        """RMS of the cap residual [N m] (all boundaries); per reported latitude when ``average_channels=False``.

        Same transitions as the loss; no alpha, tolerant map or weight. ``report_names`` labels the per-term values.
        """
        del target, kwargs
        if input is None:
            raise ValueError("CapAxialAngularMomentumSoftConstraint.physical_rmse requires input.")
        with torch.amp.autocast("cuda", enabled=False):
            r = self._boundary_residuals(prediction.float(), input.float(), None if input_diagnostics is None else input_diagnostics.float())
            ms = (r * r).mean(dim=(0, 1))
            if average_channels:
                return torch.sqrt(ms.mean()).float()
            return torch.sqrt(ms[self.report_index]).float()


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

    def constraint_specs(self) -> list[tuple[str, list[str], object]]:
        """``(name, term_names, relative_loss_scale)`` in concatenation order.

        Term order inside a constraint matches the vector that constraint
        appends. ``relative_loss_scale`` is whatever that constraint was constructed
        with; adaptive loss weights apply it, and ``name`` is only the log tag.
        """
        specs: list[tuple[str, list[str], object]] = []
        seen: set[str] = set()
        for constraint in self.constraints:
            spec_fn = getattr(constraint, "constraint_spec", None)
            if spec_fn is None:
                raise TypeError(
                    f"{type(constraint).__name__} does not expose constraint_spec()"
                )
            name, term_names, relative_loss_scale = spec_fn()
            name = str(name)
            terms = [str(term) for term in term_names]
            if name in seen:
                raise ValueError(
                    f"duplicate soft-constraint name {name!r}; each constraint "
                    "needs its own name"
                )
            if not terms:
                raise ValueError(f"soft constraint {name!r} exposes no terms")
            if any("/" in term for term in terms):
                raise ValueError(
                    f"soft constraint {name!r} term names must not contain '/': "
                    f"{terms}"
                )
            seen.add(name)
            specs.append((name, terms, relative_loss_scale))
        return specs

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
