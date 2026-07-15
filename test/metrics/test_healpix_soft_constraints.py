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

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytest
import torch
import xarray as xr

from physicsnemo.metrics.climate.healpix_loss import WeightedMSE
from physicsnemo.metrics.climate.healpix_soft_constraints import (
    DryAirMassSoftConstraint,
    HydrostasySoftConstraint,
    LossWithSoftConstraints,
)
from physicsnemo.metrics.climate.hydrostasy import LossWithHydrostasy


@dataclass
class _DummyTrainer:
    device: torch.device
    output_variables: Sequence[str]


def _dry_air_scaling():
    return {
        "sp": {"mean": 100000.0, "std": 5000.0},
        "tcwv": {"mean": 25.0, "std": 15.0},
    }


def _make_conserved_batch(channels, scaling, B=2, F=1, T=3, H=4, W=4, g0=9.81):
    """Build input/pred with constant global dry-air mass across steps."""
    sp_idx = channels.index("sp")
    tcwv_idx = channels.index("tcwv")
    C = len(channels)
    torch.manual_seed(0)
    inp = torch.randn(B, F, 1, C, H, W)
    sp_phys = (
        inp[:, :, -1:, sp_idx : sp_idx + 1] * scaling["sp"]["std"]
        + scaling["sp"]["mean"]
    )
    tcwv_phys = (
        inp[:, :, -1:, tcwv_idx : tcwv_idx + 1] * scaling["tcwv"]["std"]
        + scaling["tcwv"]["mean"]
    )
    sp_dry = (sp_phys - g0 * tcwv_phys).mean(dim=(1, 4, 5), keepdim=True)

    pred = torch.randn(B, F, T, C, H, W)
    tcwv_const = torch.full(
        (B, F, T, 1, H, W),
        (20.0 - scaling["tcwv"]["mean"]) / scaling["tcwv"]["std"],
    )
    pred[:, :, :, tcwv_idx : tcwv_idx + 1] = tcwv_const
    tcwv_phys_pred = tcwv_const * scaling["tcwv"]["std"] + scaling["tcwv"]["mean"]
    sp_phys_needed = sp_dry + g0 * tcwv_phys_pred.mean(dim=(1, 4, 5), keepdim=True)
    sp_norm = (sp_phys_needed - scaling["sp"]["mean"]) / scaling["sp"]["std"]
    pred[:, :, :, sp_idx : sp_idx + 1] = sp_norm.expand(B, F, T, 1, H, W)
    return inp, pred


def test_dry_air_mass_soft_constraint_zero_when_conserved():
    channels = ["t2m", "tcwv", "sp"]
    scaling = _dry_air_scaling()
    mod = DryAirMassSoftConstraint(
        channels=channels, scaling=scaling, weight=1.0, alpha=0.383431
    )
    trainer = _DummyTrainer(device=torch.device("cpu"), output_variables=channels)
    mod.setup(trainer)
    inp, pred = _make_conserved_batch(channels, scaling)
    target = pred.clone()
    loss = mod.constraint_loss(pred, target, input=inp)
    assert loss.shape == ()
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_dry_air_mass_soft_constraint_step_to_step():
    channels = ["tcwv", "sp"]
    scaling = _dry_air_scaling()
    mod = DryAirMassSoftConstraint(
        channels=channels, scaling=scaling, weight=1.0, alpha=0.383431
    )
    trainer = _DummyTrainer(device=torch.device("cpu"), output_variables=channels)
    mod.setup(trainer)

    B, F, T, H, W = 1, 1, 3, 2, 2
    C = len(channels)
    sp_idx = channels.index("sp")

    inp = torch.zeros(B, F, 1, C, H, W)
    pred = torch.zeros(B, F, T, C, H, W)
    # t=0 matches input; later steps drift in global dry mass
    pred[:, :, 1:, sp_idx] = 1.0
    loss_step = mod.constraint_loss(pred, pred.clone(), input=inp)
    assert float(loss_step) > 0.0

    # Anchor violation: all predicted steps differ from input
    pred2 = torch.zeros(B, F, T, C, H, W)
    pred2[:, :, :, sp_idx] = 1.0
    loss_anchor = mod.constraint_loss(pred2, pred2.clone(), input=inp)
    assert float(loss_anchor) > 0.0


def test_dry_air_mass_soft_constraint_requires_input():
    channels = ["tcwv", "sp"]
    scaling = _dry_air_scaling()
    mod = DryAirMassSoftConstraint(channels=channels, scaling=scaling)
    pred = torch.randn(1, 1, 2, 2, 2, 2)
    with pytest.raises(TypeError):
        mod.constraint_loss(pred, pred)
    with pytest.raises(ValueError, match="requires prognostic input"):
        mod.constraint_loss(pred, pred, input=None)


def test_dry_air_mass_soft_constraint_gradients():
    channels = ["tcwv", "sp"]
    scaling = _dry_air_scaling()
    mod = DryAirMassSoftConstraint(channels=channels, scaling=scaling, weight=1.0)
    trainer = _DummyTrainer(device=torch.device("cpu"), output_variables=channels)
    mod.setup(trainer)
    inp = torch.randn(1, 1, 1, 2, 3, 3)
    pred = torch.randn(1, 1, 2, 2, 3, 3, requires_grad=True)
    loss = mod.constraint_loss(pred, pred.detach(), input=inp)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[..., 0, :, :].abs().sum() > 0
    assert pred.grad[..., 1, :, :].abs().sum() > 0


def test_loss_with_soft_constraints_no_constraints():
    weights = [1.0, 2.0]
    data_loss = WeightedMSE(weights=weights)
    wrapper = LossWithSoftConstraints(data_loss=data_loss, constraints=[])
    assert wrapper.needs_input is False
    trainer = _DummyTrainer(device=torch.device("cpu"), output_variables=["a", "b"])
    wrapper.setup(trainer)
    pred = torch.randn(1, 1, 1, 2, 4, 4)
    target = torch.randn(1, 1, 1, 2, 4, 4)
    assert torch.allclose(wrapper(pred, target), data_loss(pred, target))
    per = wrapper(pred, target, average_channels=False)
    assert per.shape == (2,)
    with pytest.raises(TypeError):
        wrapper(pred, target, input=pred)


def test_hydrostasy_soft_constraint_matches_loss_with_hydrostasy(tmp_path):
    """Tv soft-constraint term matches LossWithHydrostasy Tv contribution."""
    hPa_levels = [500.0, 700.0, 850.0]
    channels = [
        "z500",
        "z700",
        "z850",
        "t500",
        "t700",
        "t850",
        "q500",
        "q700",
        "q850",
    ]
    scaling = {
        f"z{int(p)}": {"mean": 5000.0 * i, "std": 100.0}
        for i, p in enumerate(hPa_levels)
    }
    scaling.update({f"t{int(p)}": {"mean": 250.0, "std": 10.0} for p in hPa_levels})
    scaling.update({f"q{int(p)}": {"mean": 0.001, "std": 0.0005} for p in hPa_levels})
    alpha = [0.2, 0.3]
    tv_weights = [0.001, 0.002]

    faces, hh, ww = 12, 4, 4
    topo = np.zeros((faces, hh, ww), dtype=np.float32)
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

    soft = HydrostasySoftConstraint(
        hPa_levels=hPa_levels,
        channels=channels,
        weights=tv_weights,
        alpha=alpha,
        scaling=scaling,
        dataset_path=str(zarr_path),
        surface_geopotential_name="z",
        surface_geopotential_mean=0.0,
        surface_geopotential_std=1.0,
        convert_topography_to_meters=True,
        topography_masking=False,
    )
    data_loss = WeightedMSE(weights=[1.0] * len(channels))
    legacy = LossWithHydrostasy(
        data_loss=data_loss,
        hPa_levels=hPa_levels,
        channels=channels,
        weights=tv_weights,
        alpha=alpha,
        scaling=scaling,
        dataset_path=str(zarr_path),
        surface_geopotential_name="z",
        surface_geopotential_mean=0.0,
        surface_geopotential_std=1.0,
        convert_topography_to_meters=True,
        topography_masking=False,
    )
    trainer = _DummyTrainer(device=torch.device("cpu"), output_variables=channels)
    soft.setup(trainer)
    legacy.setup(trainer)

    torch.manual_seed(1)
    # Layout [N, F, B, C, H, W] with F = faces (matches hydrostasy scale/topography)
    pred = torch.randn(1, faces, 1, len(channels), hh, ww)
    target = torch.randn_like(pred)

    soft_tv = soft.constraint_loss(pred, target, average_channels=True)
    legacy_total = legacy(pred, target, average_channels=True)
    data_only = data_loss(pred, target, average_channels=True)
    legacy_tv = legacy_total - data_only
    assert torch.allclose(soft_tv, legacy_tv, rtol=1e-5, atol=1e-6)

    soft_per = soft.constraint_loss(pred, target, average_channels=False)
    legacy_per = legacy(pred, target, average_channels=False)
    assert soft_per.shape == (len(hPa_levels) - 1,)
    assert torch.allclose(soft_per, legacy_per[len(channels) :], rtol=1e-5, atol=1e-6)


def test_loss_with_soft_constraints_composes_dry_air():
    channels = ["tcwv", "sp"]
    scaling = _dry_air_scaling()
    data_loss = WeightedMSE(weights=[1.0, 1.0])
    dry = DryAirMassSoftConstraint(channels=channels, scaling=scaling, weight=1.0)
    wrapper = LossWithSoftConstraints(data_loss=data_loss, constraints=[dry])
    assert wrapper.needs_input is True
    trainer = _DummyTrainer(device=torch.device("cpu"), output_variables=channels)
    wrapper.setup(trainer)

    inp, pred = _make_conserved_batch(channels, scaling, T=2, H=3, W=3)
    target = pred + 0.1
    with pytest.raises(ValueError, match="requires prognostic input"):
        wrapper(pred, target)
    out = wrapper(pred, target, input=inp)
    assert out.shape == ()
    assert torch.allclose(out, data_loss(pred, target), rtol=1e-3, atol=1e-4)

    per = wrapper(pred, target, average_channels=False, input=inp)
    assert per.shape[0] == 3  # 2 data channels + 1 dry-air term


def test_loss_with_soft_constraints_hydra(tmp_path):
    """Hydra-style instantiate of LossWithSoftConstraints with dry-air only."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    channels = ["tcwv", "sp"]
    scaling = _dry_air_scaling()
    cfg = OmegaConf.create(
        {
            "_target_": "physicsnemo.metrics.climate.healpix_soft_constraints.LossWithSoftConstraints",
            "data_loss": {
                "_target_": "physicsnemo.metrics.climate.healpix_loss.WeightedMSE",
                "weights": [1.0, 1.0],
            },
            "constraints": [
                {
                    "_target_": "physicsnemo.metrics.climate.healpix_soft_constraints.DryAirMassSoftConstraint",
                    "channels": channels,
                    "scaling": scaling,
                    "weight": 0.001,
                    "alpha": 0.383431,
                    "g0": 9.81,
                }
            ],
        }
    )
    loss = instantiate(cfg)
    trainer = _DummyTrainer(device=torch.device("cpu"), output_variables=channels)
    loss.setup(trainer)
    inp, pred = _make_conserved_batch(channels, scaling, T=2, H=2, W=2)
    out = loss(pred, pred.clone(), input=inp)
    assert out.shape == ()
    assert out.requires_grad is False
    pred_g = pred.clone().requires_grad_(True)
    out_g = loss(pred_g, pred.clone(), input=inp)
    out_g.backward()
    assert pred_g.grad is not None
