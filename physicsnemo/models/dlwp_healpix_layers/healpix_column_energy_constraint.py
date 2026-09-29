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

"""Hard RecUNet constraint: ACE2 column energy with a uniform temperature offset.

``constant_temperature`` adds one spatially uniform offset to every
``air_temperature_*`` level so the global-mean column energy
(``c_v T + L_v q + g z``, kinetic energy omitted) changes by the global-mean
net atmospheric energy flux times the step. Height is hydrostatic. The model
does not carry ERA5 ``DSWRFtoa``; ``aux['insolation']`` is the decoder's
geometric factor and is scaled by ``solar_constant``. ``HGTsfc`` is read from
``aux['constants']`` and denormalized with its scaling.
"""

from __future__ import annotations

import inspect

import torch

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
GRAVITY = 9.80665
LATENT_HEAT_OF_VAPORIZATION = 2.5e6
LATENT_HEAT_OF_FREEZING = 334000.0
RDGAS = 287.05
RVGAS = 461.5
CV = 1004.6 - RDGAS
SOLAR_CONSTANT = 1361.0


def _mean_std(scaling: dict, name: str) -> tuple[float, float]:
    spec = scaling[name]
    return float(spec["mean"]), float(spec["std"])


def accepts_forcing(constraint) -> bool:
    """True when ``constraint.forward`` takes an ``aux`` argument."""
    forward = getattr(constraint, "forward", None)
    if forward is None:
        return False
    try:
        params = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "aux" in params


def forcing_from_folded_input(
    folded: torch.Tensor,
    *,
    input_channels: int,
    input_time_dim: int,
    decoder_input_channels: int,
    n_constants: int,
    num_faces: int = 12,
) -> dict[str, torch.Tensor]:
    """Insolation and constants from the pre-reorder folded encoder input.

    Channel order, before any structural permutation, is prognostics
    (``input_time_dim * input_channels``), decoder inputs
    (``input_time_dim * decoder_input_channels``), then constants.
    ``folded`` is ``[B * F, C, H, W]``.
    """
    n_prog = int(input_channels) * int(input_time_dim)
    n_di = int(decoder_input_channels) * int(input_time_dim)
    insol_flat = folded[:, n_prog : n_prog + n_di]
    const_flat = folded[:, n_prog + n_di : n_prog + n_di + int(n_constants)]
    bf, _, h, w = folded.shape
    batch = bf // int(num_faces)
    insolation = insol_flat.reshape(
        batch, int(num_faces), int(input_time_dim), int(decoder_input_channels), h, w
    )
    constants = const_flat.reshape(batch, int(num_faces), 1, int(n_constants), h, w)
    return {"insolation": insolation, "constants": constants}


class ColumnEnergyConstraint(torch.nn.Module):
    def __init__(
        self,
        in_channels: list[str] | None = None,
        out_channels: list[str] | None = None,
        scaling: dict[str, dict[str, float]] | None = None,
        constant_channels: list[str] | None = None,
        surface_pressure: str = "PRESsfc",
        temperature_prefix: str = "air_temperature_",
        water_prefix: str = "specific_total_water_",
        n_layers: int = 8,
        surface_height: str = "HGTsfc",
        latent_heat_flux: str = "LHTFLsfc",
        sensible_heat_flux: str = "SHTFLsfc",
        surface_down_lw: str = "DLWRFsfc",
        surface_up_lw: str = "ULWRFsfc",
        surface_down_sw: str = "DSWRFsfc",
        surface_up_sw: str = "USWRFsfc",
        toa_up_lw: str = "ULWRFtoa",
        toa_up_sw: str = "USWRFtoa",
        frozen_precipitation: str = "total_frozen_precipitation_rate",
        method: str = "constant_temperature",
        timestep_seconds: float = 21600.0,
        solar_constant: float = SOLAR_CONSTANT,
        ak: list[float] | tuple[float, ...] | None = None,
        bk: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        if method != "constant_temperature":
            raise ValueError(f"only constant_temperature is implemented, got {method!r}")
        if scaling is None or in_channels is None or out_channels is None:
            raise ValueError("in_channels, out_channels, and scaling are required")
        if constant_channels is None or surface_height not in constant_channels:
            raise ValueError(f"{surface_height!r} must be in constant_channels")
        if timestep_seconds <= 0:
            raise ValueError(f"timestep_seconds must be positive, got {timestep_seconds}")
        self.in_names = list(in_channels)
        self.out_names = list(out_channels)
        self.constant_names = list(constant_channels)
        self._h_const = self.constant_names.index(surface_height)
        self.timestep_seconds = float(timestep_seconds)
        self.solar_constant = float(solar_constant)
        n_layers = int(n_layers)
        ak = tuple(float(x) for x in (ERA5_ACE_8LAYER_AK if ak is None else ak))
        bk = tuple(float(x) for x in (ERA5_ACE_8LAYER_BK if bk is None else bk))
        if len(ak) != n_layers + 1 or len(bk) != n_layers + 1:
            raise ValueError(
                f"ak/bk must have n_layers+1={n_layers + 1} interfaces, "
                f"got ak={len(ak)} bk={len(bk)}"
            )
        self.temp_names = [f"{temperature_prefix}{k}" for k in range(n_layers)]
        self.water_names = [f"{water_prefix}{k}" for k in range(n_layers)]
        self.flux_names = (
            latent_heat_flux,
            sensible_heat_flux,
            surface_down_lw,
            surface_up_lw,
            surface_down_sw,
            surface_up_sw,
            toa_up_lw,
            toa_up_sw,
            frozen_precipitation,
        )
        for name in (surface_pressure, *self.temp_names, *self.water_names):
            if name not in self.in_names:
                raise ValueError(f"{name!r} is missing from in_channels")
        for name in (
            surface_pressure,
            *self.temp_names,
            *self.water_names,
            *self.flux_names,
        ):
            if name not in self.out_names:
                raise ValueError(f"{name!r} is not in out_channels")
            if name not in scaling:
                raise ValueError(f"scaling is missing {name!r}")
        if surface_height not in scaling:
            raise ValueError(f"scaling is missing {surface_height!r}")
        self._ps_out = self.out_names.index(surface_pressure)
        self._ps_in = self.in_names.index(surface_pressure)
        self._t_out = tuple(self.out_names.index(n) for n in self.temp_names)
        self._t_in = tuple(self.in_names.index(n) for n in self.temp_names)
        self._q_out = tuple(self.out_names.index(n) for n in self.water_names)
        self._q_in = tuple(self.in_names.index(n) for n in self.water_names)
        self._flux = tuple(self.out_names.index(n) for n in self.flux_names)

        def _pair(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            mean, std = _mean_std(scaling, name)
            if std == 0.0:
                raise ValueError(f"{name} scaling std must be nonzero")
            return (
                torch.tensor(mean, dtype=torch.float32),
                torch.tensor(std, dtype=torch.float32),
            )

        ps_mean, ps_std = _pair(surface_pressure)
        self.register_buffer("ps_mean", ps_mean, persistent=False)
        self.register_buffer("ps_std", ps_std, persistent=False)
        h_mean, h_std = _pair(surface_height)
        self.register_buffer("h_mean", h_mean, persistent=False)
        self.register_buffer("h_std", h_std, persistent=False)
        for prefix, names in (("t", self.temp_names), ("q", self.water_names)):
            mean = torch.tensor([_mean_std(scaling, n)[0] for n in names], dtype=torch.float32)
            std = torch.tensor([_mean_std(scaling, n)[1] for n in names], dtype=torch.float32)
            if torch.any(std == 0):
                raise ValueError(f"{prefix} scaling std must be nonzero")
            self.register_buffer(f"{prefix}_mean", mean, persistent=False)
            self.register_buffer(f"{prefix}_std", std, persistent=False)
        flux_mean = torch.tensor(
            [_mean_std(scaling, n)[0] for n in self.flux_names], dtype=torch.float32
        )
        flux_std = torch.tensor(
            [_mean_std(scaling, n)[1] for n in self.flux_names], dtype=torch.float32
        )
        if torch.any(flux_std == 0):
            raise ValueError("flux scaling std must be nonzero")
        self.register_buffer("flux_mean", flux_mean, persistent=False)
        self.register_buffer("flux_std", flux_std, persistent=False)
        self.register_buffer("ak", torch.tensor(ak, dtype=torch.float32), persistent=False)
        self.register_buffer("bk", torch.tensor(bk, dtype=torch.float32), persistent=False)

    def _stack(self, tensor: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        return torch.cat([tensor[:, :, :, i : i + 1] for i in indices], dim=3)

    def _denorm_one(self, tensor, idx, mean, std):
        return tensor[:, :, :, idx : idx + 1] * std + mean

    def _denorm_levels(self, tensor, indices, mean, std):
        stacked = self._stack(tensor, indices)
        return stacked * std.view(1, 1, 1, -1, 1, 1) + mean.view(1, 1, 1, -1, 1, 1)

    def _write(self, tensor, idx, values):
        return torch.cat([tensor[:, :, :, :idx], values, tensor[:, :, :, idx + 1 :]], dim=3)

    def _align_time(self, tensor: torch.Tensor, n_time: int) -> torch.Tensor:
        if tensor.shape[2] == n_time:
            return tensor
        return tensor[:, :, -n_time:]

    def _interface_pressure(self, ps: torch.Tensor) -> torch.Tensor:
        ak = self.ak.view(1, 1, 1, -1, 1, 1)
        bk = self.bk.view(1, 1, 1, -1, 1, 1)
        return ak + bk * ps

    def _layer_thickness(self, ps, temperature, q):
        tv = temperature * (1.0 + (RVGAS / RDGAS - 1.0) * q)
        dlogp = torch.log(self._interface_pressure(ps).clamp(min=1.0)).diff(dim=3)
        return dlogp * RDGAS * tv / GRAVITY

    def _height_mid(self, thickness, surface_height):
        cumulative = torch.cumsum(thickness.flip(3), dim=3).flip(3)
        hs = torch.where(surface_height < 0, torch.zeros_like(surface_height), surface_height)
        height_if = torch.cat([cumulative + hs, hs], dim=3)
        return 0.5 * (height_if[:, :, :, :-1] + height_if[:, :, :, 1:])

    def _column_integral(self, integrand, ps):
        dp = self._interface_pressure(ps).diff(dim=3)
        return (integrand * dp).sum(dim=3, keepdim=True) / GRAVITY

    def _column_energy(self, temperature, q, ps, surface_height):
        z = self._height_mid(self._layer_thickness(ps, temperature, q), surface_height)
        return self._column_integral(CV * temperature + LATENT_HEAT_OF_VAPORIZATION * q + GRAVITY * z, ps)

    def _correction_factor(self, temperature, q, ps):
        dz = self._layer_thickness(ps, temperature, q)
        q_times_dlogp = dz * GRAVITY / temperature
        cumulative = torch.cumsum(q_times_dlogp.flip(3), dim=3).flip(3)
        integrand = CV - 0.5 * q_times_dlogp + cumulative
        return self._column_integral(integrand, ps)

    def _global_mean(self, field):
        return field.mean(dim=(1, 4, 5), keepdim=True)

    def forward(self, prediction: torch.Tensor, input: torch.Tensor, aux=None) -> torch.Tensor:
        if aux is None or "insolation" not in aux or "constants" not in aux:
            raise ValueError("column energy constraint requires aux insolation and constants")
        orig_dtype = prediction.dtype
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            orig = input.float()
            n_time = prediction.shape[2]
            orig = self._align_time(orig, n_time)
            ps = self._denorm_one(prediction, self._ps_out, self.ps_mean, self.ps_std)
            ps0 = self._denorm_one(orig, self._ps_in, self.ps_mean, self.ps_std)
            temperature = self._denorm_levels(prediction, self._t_out, self.t_mean, self.t_std)
            temperature0 = self._denorm_levels(orig, self._t_in, self.t_mean, self.t_std)
            q = self._denorm_levels(prediction, self._q_out, self.q_mean, self.q_std)
            q0 = self._denorm_levels(orig, self._q_in, self.q_mean, self.q_std)
            constants = aux["constants"].float()
            if constants.shape[2] != 1 and constants.shape[2] != n_time:
                constants = self._align_time(constants, n_time)
            height = constants[:, :, :, self._h_const : self._h_const + 1] * self.h_std + self.h_mean
            insolation = self._align_time(aux["insolation"].float(), n_time)
            toa_down_sw = insolation[:, :, :, :1] * self.solar_constant
            fluxes = self._denorm_levels(prediction, self._flux, self.flux_mean, self.flux_std)
            lhf, shf, dlw, ulw, dsw, usw, ulw_toa, usw_toa, frozen = fluxes.split(1, dim=3)
            surface = (dsw - usw + dlw - ulw) - lhf - shf - frozen * LATENT_HEAT_OF_FREEZING
            into_atmosphere = (toa_down_sw - usw_toa - ulw_toa) - surface
            energy = self._column_energy(temperature, q, ps, height)
            energy0 = self._column_energy(temperature0, q0, ps0, height)
            desired = self._global_mean(energy0) + self._global_mean(into_atmosphere) * self.timestep_seconds
            factor = self._global_mean(self._correction_factor(temperature, q, ps))
            offset = (desired - self._global_mean(energy)) / factor
            corrected = temperature + offset
            norm = (corrected - self.t_mean.view(1, 1, 1, -1, 1, 1)) / self.t_std.view(
                1, 1, 1, -1, 1, 1
            )
            out = prediction
            for k, idx in enumerate(self._t_out):
                out = self._write(out, idx, norm[:, :, :, k : k + 1])
        return out.to(dtype=orig_dtype)
