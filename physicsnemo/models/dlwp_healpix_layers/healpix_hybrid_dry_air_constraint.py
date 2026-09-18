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

"""Hard RecUNet constraint: pin global-mean hybrid-sigma dry-air mass.

ACE AtmosphereCorrector (``_adjust_gen_dry_air_to_target``):

    p_{s,dry} = p_s − Σ_k (Δa_k + Δb_k p_s) q_k
    p_s ← (p_{s,dry}^{target} + Σ Δa_k q_k) / (1 − Σ Δb_k q_k)

``target`` is the HEALPix-area global mean of ``p_{s,dry}`` on RecUNet
``orig_input`` (previous step / IC). Equal-area mean is over faces, H, W.
This pins the monopole only; it does not flatten a planetary dipole.

Default ``ak``/``bk`` are the ERA5 8-layer interfaces from
``/global/cfs/projectdirs/e3sm/yikwill/datasets/2026-03-19-era5-1deg-8layer-1940-2025.zarr``
(``ak_0``…``ak_8``, ``bk_0``…``bk_8``). Those are also baked into the HPX ACE
catalog. Do not substitute Watt-Meyer 2023 Table 2.
"""

from __future__ import annotations

import torch

# Source-zarr scalars (Pa / 1). Copied from the 2026-03-19 ERA5 8-layer store.
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
_DENOM_MIN = 1.0e-6


def _mean_std(scaling: dict, name: str) -> tuple[float, float]:
    spec = scaling[name]
    return float(spec["mean"]), float(spec["std"])


class HybridSigmaDryAirMassConstraint(torch.nn.Module):
    def __init__(
        self,
        in_channels: list[str] | None = None,
        out_channels: list[str] | None = None,
        scaling: dict[str, dict[str, float]] | None = None,
        surface_pressure: str = "PRESsfc",
        water_prefix: str = "specific_total_water_",
        n_layers: int = 8,
        ak: list[float] | tuple[float, ...] | None = None,
        bk: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        if scaling is None:
            raise ValueError("scaling is required")
        if in_channels is None and out_channels is None:
            raise ValueError("in_channels or out_channels is required")
        self.in_names = list(in_channels) if in_channels is not None else list(out_channels)
        self.out_names = list(out_channels) if out_channels is not None else list(in_channels)
        n_layers = int(n_layers)
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        ak = tuple(float(x) for x in (ERA5_ACE_8LAYER_AK if ak is None else ak))
        bk = tuple(float(x) for x in (ERA5_ACE_8LAYER_BK if bk is None else bk))
        if len(ak) != n_layers + 1 or len(bk) != n_layers + 1:
            raise ValueError(
                f"ak/bk must have n_layers+1={n_layers + 1} interfaces, "
                f"got ak={len(ak)} bk={len(bk)}"
            )
        water_names = [f"{water_prefix}{k}" for k in range(n_layers)]
        for name in (surface_pressure, *water_names):
            if name not in self.out_names:
                raise ValueError(f"{name!r} is not in out_channels {self.out_names}")
            if name not in self.in_names:
                raise ValueError(
                    f"{name!r} is diagnostic or missing from in_channels {self.in_names}"
                )
            if name not in scaling:
                raise ValueError(f"scaling is missing {name!r}")
        self._ps_out = self.out_names.index(surface_pressure)
        self._ps_in = self.in_names.index(surface_pressure)
        self._q_out = tuple(self.out_names.index(n) for n in water_names)
        self._q_in = tuple(self.in_names.index(n) for n in water_names)

        ps_mean, ps_std = _mean_std(scaling, surface_pressure)
        if ps_std == 0.0:
            raise ValueError(f"{surface_pressure} scaling std must be nonzero")
        self.register_buffer("ps_mean", torch.tensor(ps_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("ps_std", torch.tensor(ps_std, dtype=torch.float32), persistent=False)
        q_mean = torch.tensor(
            [_mean_std(scaling, n)[0] for n in water_names], dtype=torch.float32
        )
        q_std = torch.tensor(
            [_mean_std(scaling, n)[1] for n in water_names], dtype=torch.float32
        )
        if torch.any(q_std == 0):
            raise ValueError("water scaling std must be nonzero")
        self.register_buffer("q_mean", q_mean, persistent=False)
        self.register_buffer("q_std", q_std, persistent=False)
        dak = torch.tensor(ak, dtype=torch.float32).diff()
        dbk = torch.tensor(bk, dtype=torch.float32).diff()
        self.register_buffer("dak", dak, persistent=False)
        self.register_buffer("dbk", dbk, persistent=False)

    def _stack(self, tensor: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        return torch.cat([tensor[:, :, :, i : i + 1] for i in indices], dim=3)

    def _replace_ps(self, tensor: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        i = self._ps_out
        return torch.cat(
            [tensor[:, :, :, :i], values, tensor[:, :, :, i + 1 :]],
            dim=3,
        )

    def _denorm_ps(self, tensor: torch.Tensor, idx: int) -> torch.Tensor:
        return tensor[:, :, :, idx : idx + 1] * self.ps_std + self.ps_mean

    def _denorm_q(self, tensor: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        q = self._stack(tensor, indices)
        mean = self.q_mean.view(1, 1, 1, -1, 1, 1)
        std = self.q_std.view(1, 1, 1, -1, 1, 1)
        return q * std + mean

    def _ps_dry(self, ps: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        dak = self.dak.view(1, 1, 1, -1, 1, 1)
        dbk = self.dbk.view(1, 1, 1, -1, 1, 1)
        water = (q * (dak + dbk * ps)).sum(dim=3, keepdim=True)
        return ps - water

    def _invert_ps(self, ps_dry: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        dak = self.dak.view(1, 1, 1, -1, 1, 1)
        dbk = self.dbk.view(1, 1, 1, -1, 1, 1)
        numer = ps_dry + (dak * q).sum(dim=3, keepdim=True)
        denom = (1.0 - (dbk * q).sum(dim=3, keepdim=True)).clamp(min=_DENOM_MIN)
        return numer / denom

    def forward(self, prediction: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        orig_dtype = prediction.dtype
        with torch.amp.autocast("cuda", enabled=False):
            prediction = prediction.float()
            orig = input.float()
            if orig.shape[2] != prediction.shape[2]:
                orig = orig[:, :, -1:]
            ps = self._denorm_ps(prediction, self._ps_out)
            q = self._denorm_q(prediction, self._q_out)
            ps0 = self._denorm_ps(orig, self._ps_in)
            q0 = self._denorm_q(orig, self._q_in)
            ps_dry = self._ps_dry(ps, q)
            ps_dry0 = self._ps_dry(ps0, q0)
            target = ps_dry0.mean(dim=(1, 4, 5), keepdim=True)
            error = ps_dry.mean(dim=(1, 4, 5), keepdim=True) - target
            new_ps = torch.clamp(self._invert_ps(ps_dry - error, q), min=0.0)
            new_ps_norm = (new_ps - self.ps_mean) / self.ps_std
            out = self._replace_ps(prediction, new_ps_norm)
        return out.to(dtype=orig_dtype)
