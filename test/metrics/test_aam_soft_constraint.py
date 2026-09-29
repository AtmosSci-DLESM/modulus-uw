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

"""Tests for axial angular-momentum soft constraint helpers."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytest
import torch
import xarray as xr

from physicsnemo.metrics.climate.healpix_soft_constraints import (
    AxialAngularMomentumSoftConstraint,
    _EARTH_RADIUS_M,
    earth_angular_momentum,
    layer_delta_p_pa,
    relative_angular_momentum,
)


@dataclass
class _DummyTrainer:
    device: torch.device
    output_variables: Sequence[str]


def test_layer_delta_p_fully_underground_level_contributes_nothing():
    """Layer entirely below p_s has Δp = 0; straddling layer keeps above-ground only."""
    p = torch.tensor([500.0, 700.0, 850.0])  # hPa
    # p_s = 600 hPa → layer 500–700 keeps 100 hPa; layer 700–850 is underground.
    sp = torch.full((1, 1, 1, 1, 2, 2), 60000.0)  # Pa
    dp = layer_delta_p_pa(p, sp)
    assert dp.shape == (1, 1, 1, 2, 2, 2)
    assert torch.allclose(dp[:, :, :, 0], torch.full_like(dp[:, :, :, 0], 10000.0))
    assert torch.allclose(dp[:, :, :, 1], torch.zeros_like(dp[:, :, :, 1]))


def test_relative_aam_analytic_column():
    """Uniform column: M_r matches (a³/g) u cosφ Δp ΔΩ N_pix analytically."""
    nside = 2
    n_pix = 12 * nside * nside
    d_omega = 4.0 * np.pi / n_pix
    a = _EARTH_RADIUS_M
    g = 9.81
    u0 = 10.0  # m/s
    cos0 = 0.5
    dp0 = 10000.0  # Pa (= 100 hPa)

    B, F, T, L, H, W = 1, 12, 1, 1, nside, nside
    u_layer = torch.full((B, F, T, L, H, W), u0)
    delta_p = torch.full((B, F, T, L, H, W), dp0)
    cos_phi = torch.full((1, F, 1, 1, H, W), cos0)

    m_r = relative_angular_momentum(u_layer, delta_p, cos_phi, d_omega, a=a, g=g)
    expected = (a ** 3 / g) * u0 * cos0 * dp0 * d_omega * n_pix
    assert m_r.shape == (B, T)
    assert float(m_r[0, 0]) == pytest.approx(expected, rel=1e-6)


def test_underground_level_drops_out_of_relative_aam():
    """u in a fully underground layer does not change M_r."""
    p = torch.tensor([500.0, 700.0, 850.0])
    nside = 2
    F, H, W = 12, nside, nside
    d_omega = 4.0 * np.pi / (12 * nside * nside)
    cos_phi = torch.ones(1, F, 1, 1, H, W)

    # p_s = 600 hPa → only first layer (500–700) is active.
    sp = torch.full((1, F, 1, 1, H, W), 60000.0)
    dp = layer_delta_p_pa(p, sp)
    assert torch.count_nonzero(dp[:, :, :, 1]) == 0

    u_active = torch.zeros(1, F, 1, 2, H, W)
    u_active[:, :, :, 0] = 5.0
    m0 = relative_angular_momentum(u_active, dp, cos_phi, d_omega)

    u_with_underground = u_active.clone()
    u_with_underground[:, :, :, 1] = 100.0  # huge wind underground
    m1 = relative_angular_momentum(u_with_underground, dp, cos_phi, d_omega)
    assert torch.allclose(m0, m1)


def _aam_scaling(levels):
    scaling = {
        "sp": {"mean": 100000.0, "std": 5000.0},
        "avg_iews-6h": {"mean": 0.0, "std": 0.1},
        "avg_iegwss-6h": {"mean": 0.0, "std": 0.05},
    }
    for pl in levels:
        scaling[f"u{int(pl)}"] = {"mean": 0.0, "std": 10.0}
    return scaling


def test_aam_soft_constraint_physical_rmse_smoke(tmp_path):
    """CPU smoke: module builds, IC→pred residual path runs (mountain torque=0)."""
    levels = [500.0, 700.0, 850.0]
    nside = 2
    channels = (
        [f"u{int(p)}" for p in levels]
        + ["sp", "avg_iews-6h", "avg_iegwss-6h"]
    )
    topo = np.zeros((12, nside, nside), dtype=np.float32)
    ds = xr.Dataset(
        {
            "constants": (
                ("face", "channel_c", "height", "width"),
                topo[:, None, :, :],
            )
        },
        coords={"channel_c": ["z"]},
    )
    zarr_path = tmp_path / "topo.zarr"
    ds.to_zarr(zarr_path)

    mod = AxialAngularMomentumSoftConstraint(
        hPa_levels=levels,
        channels=channels,
        scaling=_aam_scaling(levels),
        dataset_path=str(zarr_path),
        surface_geopotential_name="z",
        surface_geopotential_mean=0.0,
        surface_geopotential_std=1.0,
        convert_topography_to_meters=True,
        weight=1.0,
        alpha=1.0e20,  # huge alpha → near-zero training loss
    )
    trainer = _DummyTrainer(device=torch.device("cpu"), output_variables=channels)
    mod.setup(trainer)

    B, F, T, H, W = 1, 12, 2, nside, nside
    C = len(channels)
    inp = torch.zeros(B, F, 1, C, H, W)
    pred = torch.zeros(B, F, T, C, H, W)
    loss = mod.constraint_loss(pred, pred.clone(), input=inp)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    rmse = mod.physical_rmse(pred, pred.clone(), input=inp)
    assert rmse.shape == ()
    assert torch.isfinite(rmse)
    assert float(rmse) >= 0.0


def test_aam_physical_rmse_stays_finite_at_q50_scale(tmp_path):
    """fp32 squaring of a ~1e19 N·m residual overflows; the RMSE must not."""
    levels = [500.0, 700.0, 850.0]
    nside = 2
    channels = (
        [f"u{int(p)}" for p in levels]
        + ["sp", "avg_iews-6h", "avg_iegwss-6h"]
    )
    topo = np.zeros((12, nside, nside), dtype=np.float32)
    ds = xr.Dataset(
        {
            "constants": (
                ("face", "channel_c", "height", "width"),
                topo[:, None, :, :],
            )
        },
        coords={"channel_c": ["z"]},
    )
    zarr_path = tmp_path / "topo.zarr"
    ds.to_zarr(zarr_path)
    mod = AxialAngularMomentumSoftConstraint(
        hPa_levels=levels,
        channels=channels,
        scaling=_aam_scaling(levels),
        dataset_path=str(zarr_path),
        surface_geopotential_name="z",
        surface_geopotential_mean=0.0,
        surface_geopotential_std=1.0,
        convert_topography_to_meters=True,
        weight=1.0,
        alpha=2.93e19,
    )
    mod.setup(_DummyTrainer(device=torch.device("cpu"), output_variables=channels))
    residual = torch.tensor([[2.93e19, -1.0e19]], dtype=torch.float32)
    # The naive fp32 reduction is the bug this guards.
    assert not torch.isfinite(torch.sqrt((residual ** 2).mean()))
    mod._budget_residuals = lambda *args, **kwargs: residual
    B, F, T, H, W = 1, 12, 2, nside, nside
    zeros = torch.zeros(B, F, T, len(channels), H, W)
    rmse = mod.physical_rmse(zeros, zeros, input=zeros[:, :, :1])
    expected = torch.sqrt((residual.double() ** 2).mean()).float()
    assert torch.isfinite(rmse)
    assert torch.allclose(rmse, expected)


def test_earth_aam_positive_for_positive_sp():
    nside = 2
    F, H, W = 12, nside, nside
    d_omega = 4.0 * np.pi / (12 * nside * nside)
    sp = torch.full((1, F, 1, 1, H, W), 100000.0)
    cos_phi = torch.full((1, F, 1, 1, H, W), 0.5)
    m = earth_angular_momentum(sp, cos_phi, d_omega)
    assert float(m[0, 0]) > 0.0
