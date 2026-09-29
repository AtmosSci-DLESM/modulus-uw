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

"""Hard RecUNet constraint: close the column moisture budget.

ACE ``MoistureBudgetCorrection`` with ``advection_and_precipitation``:

1. Scale ``PRATEsfc`` so the global-mean budget closes, treating global-mean
   moisture advection as zero.
2. Replace ``tendency_of_total_water_path_due_to_advection`` with the column
   residual, which then has zero global mean.
3. Optionally clip ``total_frozen_precipitation_rate`` to the corrected
   ``PRATEsfc``.

Column water is the hybrid-sigma mass integral ``Σ q Δp / g``. ``Δp`` uses the
same ERA5 8-layer ``ak``/``bk`` as the dry-air constraint. Equal-area means are
over faces, H, and W.
"""

from __future__ import annotations

import torch

# Source-zarr scalars (Pa / 1). Same interfaces as HybridSigmaDryAirMassConstraint.
ERA5_ACE_8LAYER_AK = (
    1.0001825094223022,
    5119.89501953125,
    13881.3310546875,
    19343.51171875,
    20087.0859375,
    15596.6953125,
    8880.453125,
    3057.265625,
    0.0,
)
ERA5_ACE_8LAYER_BK = (
    0.0,
    0.0,
    0.005377814639359713,
    0.059728413820266724,
    0.2034912109375,
    0.43839120864868164,
    0.6806430220603943,
    0.8739292621612549,
    1.0,
)
GRAVITY = 9.80665  # m/s^2
LATENT_HEAT_OF_VAPORIZATION = 2.5e6  # J/kg
_PRECIP_MEAN_FLOOR = 1.0e-12


def _mean_std(scaling: dict, name: str) -> tuple[float, float]:
    spec = scaling[name]
    return float(spec["mean"]), float(spec["std"])


class MoistureBudgetConstraint(torch.nn.Module):
    def __init__(
        self,
        in_channels: list[str] | None = None,
        out_channels: list[str] | None = None,
        scaling: dict[str, dict[str, float]] | None = None,
        surface_pressure: str = "PRESsfc",
        water_prefix: str = "specific_total_water_",
        n_layers: int = 8,
        precipitation: str = "PRATEsfc",
        latent_heat_flux: str = "LHTFLsfc",
        moisture_advection: str = "tendency_of_total_water_path_due_to_advection",
        frozen_precipitation: str = "total_frozen_precipitation_rate",
        terms_to_modify: str = "advection_and_precipitation",
        clip_frozen_precipitation: bool = True,
        timestep_seconds: float = 21600.0,
        ak: list[float] | tuple[float, ...] | None = None,
        bk: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        if terms_to_modify != "advection_and_precipitation":
            raise ValueError(
                "only advection_and_precipitation is implemented, "
                f"got {terms_to_modify!r}"
            )
        if scaling is None:
            raise ValueError("scaling is required")
        if in_channels is None or out_channels is None:
            raise ValueError("in_channels and out_channels are required")
        if timestep_seconds <= 0:
            raise ValueError(f"timestep_seconds must be positive, got {timestep_seconds}")
        self.in_names = list(in_channels)
        self.out_names = list(out_channels)
        self.clip_frozen = bool(clip_frozen_precipitation)
        self.timestep_seconds = float(timestep_seconds)
        n_layers = int(n_layers)
        ak = tuple(float(x) for x in (ERA5_ACE_8LAYER_AK if ak is None else ak))
        bk = tuple(float(x) for x in (ERA5_ACE_8LAYER_BK if bk is None else bk))
        if len(ak) != n_layers + 1 or len(bk) != n_layers + 1:
            raise ValueError(
                f"ak/bk must have n_layers+1={n_layers + 1} interfaces, "
                f"got ak={len(ak)} bk={len(bk)}"
            )
        water_names = [f"{water_prefix}{k}" for k in range(n_layers)]
        required_out = (
            surface_pressure,
            *water_names,
            precipitation,
            latent_heat_flux,
            moisture_advection,
        )
        if self.clip_frozen:
            required_out = (*required_out, frozen_precipitation)
        for name in required_out:
            if name not in self.out_names:
                raise ValueError(f"{name!r} is not in out_channels")
            if name not in scaling:
                raise ValueError(f"scaling is missing {name!r}")
        for name in (surface_pressure, *water_names):
            if name not in self.in_names:
                raise ValueError(f"{name!r} is missing from in_channels")
        self._ps_out = self.out_names.index(surface_pressure)
        self._ps_in = self.in_names.index(surface_pressure)
        self._q_out = tuple(self.out_names.index(n) for n in water_names)
        self._q_in = tuple(self.in_names.index(n) for n in water_names)
        self._precip = self.out_names.index(precipitation)
        self._lhf = self.out_names.index(latent_heat_flux)
        self._adv = self.out_names.index(moisture_advection)
        self._frozen = self.out_names.index(frozen_precipitation) if self.clip_frozen else None

        def _buf(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            mean, std = _mean_std(scaling, name)
            if std == 0.0:
                raise ValueError(f"{name} scaling std must be nonzero")
            return (
                torch.tensor(mean, dtype=torch.float32),
                torch.tensor(std, dtype=torch.float32),
            )

        for attr, name in (
            ("ps", surface_pressure),
            ("precip", precipitation),
            ("lhf", latent_heat_flux),
            ("adv", moisture_advection),
        ):
            mean, std = _buf(name)
            self.register_buffer(f"{attr}_mean", mean, persistent=False)
            self.register_buffer(f"{attr}_std", std, persistent=False)
        if self.clip_frozen:
            mean, std = _buf(frozen_precipitation)
            self.register_buffer("frozen_mean", mean, persistent=False)
            self.register_buffer("frozen_std", std, persistent=False)
        q_mean = torch.tensor([_mean_std(scaling, n)[0] for n in water_names], dtype=torch.float32)
        q_std = torch.tensor([_mean_std(scaling, n)[1] for n in water_names], dtype=torch.float32)
        if torch.any(q_std == 0):
            raise ValueError("water scaling std must be nonzero")
        self.register_buffer("q_mean", q_mean, persistent=False)
        self.register_buffer("q_std", q_std, persistent=False)
        self.register_buffer("dak", torch.tensor(ak, dtype=torch.float32).diff(), persistent=False)
        self.register_buffer("dbk", torch.tensor(bk, dtype=torch.float32).diff(), persistent=False)

    def _stack(self, tensor: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        return torch.cat([tensor[:, :, :, i : i + 1] for i in indices], dim=3)

    def _denorm_one(self, tensor: torch.Tensor, idx: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return tensor[:, :, :, idx : idx + 1] * std + mean

    def _denorm_q(self, tensor: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        q = self._stack(tensor, indices)
        mean = self.q_mean.view(1, 1, 1, -1, 1, 1)
        std = self.q_std.view(1, 1, 1, -1, 1, 1)
        return q * std + mean

    def _write(self, tensor: torch.Tensor, idx: int, values: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor[:, :, :, :idx], values, tensor[:, :, :, idx + 1 :]], dim=3)

    def _column_water(self, ps: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """kg m−2. ``ps`` is [B, F, T, 1, H, W], ``q`` stacks layers on dim 3."""
        dak = self.dak.view(1, 1, 1, -1, 1, 1)
        dbk = self.dbk.view(1, 1, 1, -1, 1, 1)
        return (q * (dak + dbk * ps)).sum(dim=3, keepdim=True) / GRAVITY

    def _global_mean(self, field: torch.Tensor) -> torch.Tensor:
        return field.mean(dim=(1, 4, 5), keepdim=True)

    def forward(self, prediction: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        orig_dtype = prediction.dtype
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            orig = input.float()
            if orig.shape[2] != prediction.shape[2]:
                orig = orig[:, :, -prediction.shape[2] :]
            ps = self._denorm_one(prediction, self._ps_out, self.ps_mean, self.ps_std)
            q = self._denorm_q(prediction, self._q_out)
            ps0 = self._denorm_one(orig, self._ps_in, self.ps_mean, self.ps_std)
            q0 = self._denorm_q(orig, self._q_in)
            tendency = (self._column_water(ps, q) - self._column_water(ps0, q0)) / self.timestep_seconds
            evap = self._denorm_one(prediction, self._lhf, self.lhf_mean, self.lhf_std) / LATENT_HEAT_OF_VAPORIZATION
            precip = self._denorm_one(prediction, self._precip, self.precip_mean, self.precip_std)
            precip_mean = self._global_mean(precip)
            # A zero global-mean rate cannot be scaled onto a nonzero target.
            scale = torch.where(
                precip_mean.abs() > _PRECIP_MEAN_FLOOR,
                (self._global_mean(evap) - self._global_mean(tendency)) / precip_mean,
                torch.ones_like(precip_mean),
            )
            precip = precip * scale
            advection = tendency - (evap - precip)
            out = self._write(prediction, self._precip, (precip - self.precip_mean) / self.precip_std)
            out = self._write(out, self._adv, (advection - self.adv_mean) / self.adv_std)
            if self.clip_frozen:
                frozen = self._denorm_one(prediction, self._frozen, self.frozen_mean, self.frozen_std)
                frozen = torch.minimum(frozen, precip)
                out = self._write(out, self._frozen, (frozen - self.frozen_mean) / self.frozen_std)
        return out.to(dtype=orig_dtype)
