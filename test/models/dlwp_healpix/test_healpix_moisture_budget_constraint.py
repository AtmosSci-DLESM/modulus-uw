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
# ruff: noqa: E402
import os
import sys
import time

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import numpy as np
import pytest
import torch

from physicsnemo.models.dlwp_healpix_layers.healpix_moisture_budget_constraint import (
    GRAVITY,
    LATENT_HEAT_OF_VAPORIZATION,
    ERA5_ACE_8LAYER_AK,
    ERA5_ACE_8LAYER_BK,
    MoistureBudgetConstraint,
)

N_LAYERS = 8
WATER = [f"specific_total_water_{k}" for k in range(N_LAYERS)]
IN_CH = ["PRESsfc", *WATER]
DIAG = [
    "LHTFLsfc",
    "PRATEsfc",
    "tendency_of_total_water_path_due_to_advection",
    "total_frozen_precipitation_rate",
]
OUT_CH = [*IN_CH, *DIAG]
SCALING = {n: {"mean": 0.0, "std": 1.0} for n in OUT_CH}
DT = 21600.0
DAK = np.diff(np.asarray(ERA5_ACE_8LAYER_AK))
DBK = np.diff(np.asarray(ERA5_ACE_8LAYER_BK))


def _make(**kwargs):
    defaults = dict(
        in_channels=IN_CH,
        out_channels=OUT_CH,
        scaling=SCALING,
        timestep_seconds=DT,
        clip_frozen_precipitation=True,
    )
    defaults.update(kwargs)
    return MoistureBudgetConstraint(**defaults)


def _twp(ps, q):
    dak = DAK.reshape(1, 1, 1, -1, 1, 1)
    dbk = DBK.reshape(1, 1, 1, -1, 1, 1)
    return (q * (dak + dbk * ps)).sum(axis=3, keepdims=True) / GRAVITY


def test_rejects_other_terms():
    with pytest.raises(ValueError, match="advection_and_precipitation"):
        _make(terms_to_modify="precipitation")


def test_closes_budget_and_clips_frozen():
    rng = np.random.default_rng(0)
    shape = (2, 12, 1, 1, 4, 4)
    ps0 = 98000.0 + 500.0 * rng.standard_normal(shape)
    q0 = np.clip(0.004 + 0.001 * rng.standard_normal((2, 12, 1, N_LAYERS, 4, 4)), 1e-4, 0.02)
    ps = ps0 + 200.0 * rng.standard_normal(shape)
    q = np.clip(q0 + 0.0003 * rng.standard_normal(q0.shape), 1e-4, 0.02)
    lhf = 80.0 + 20.0 * rng.standard_normal(shape)
    precip = np.clip(2.0e-5 + 1.0e-5 * rng.standard_normal(shape), 1e-6, None)
    frozen = precip * 3.0
    adv = np.zeros(shape)
    pred = torch.zeros(2, 12, 1, len(OUT_CH), 4, 4)
    orig = torch.zeros(2, 12, 1, len(IN_CH), 4, 4)
    pred[:, :, :, 0:1] = torch.from_numpy(ps).float()
    pred[:, :, :, 1 : 1 + N_LAYERS] = torch.from_numpy(q).float()
    orig[:, :, :, 0:1] = torch.from_numpy(ps0).float()
    orig[:, :, :, 1 : 1 + N_LAYERS] = torch.from_numpy(q0).float()
    base = 1 + N_LAYERS
    pred[:, :, :, base] = torch.from_numpy(lhf).float().squeeze(3)
    pred[:, :, :, base + 1] = torch.from_numpy(precip).float().squeeze(3)
    pred[:, :, :, base + 2] = torch.from_numpy(adv).float().squeeze(3)
    pred[:, :, :, base + 3] = torch.from_numpy(frozen).float().squeeze(3)
    mod = _make()
    with torch.no_grad():
        out = mod(pred, orig).numpy()
    new_precip = out[:, :, :, base + 1 : base + 2]
    new_adv = out[:, :, :, base + 2 : base + 3]
    new_frozen = out[:, :, :, base + 3 : base + 4]
    tendency = (_twp(ps, q) - _twp(ps0, q0)) / DT
    evap = lhf / LATENT_HEAT_OF_VAPORIZATION
    residual = tendency - (evap - new_precip + new_adv)
    assert np.max(np.abs(residual)) < 1e-5
    assert np.max(np.abs(new_adv.mean(axis=(1, 4, 5)))) < 1e-6
    assert np.all(new_frozen <= new_precip + 1e-7)


def test_nside64_is_fast():
    rng = np.random.default_rng(1)
    h = w = 64
    ps = np.full((1, 12, 1, 1, h, w), 98500.0)
    q = np.full((1, 12, 1, N_LAYERS, h, w), 0.002)
    pred = torch.zeros(1, 12, 1, len(OUT_CH), h, w)
    orig = torch.zeros(1, 12, 1, len(IN_CH), h, w)
    pred[:, :, :, 0:1] = torch.from_numpy(ps)
    pred[:, :, :, 1 : 1 + N_LAYERS] = torch.from_numpy(q)
    orig.copy_(pred[:, :, :, : len(IN_CH)])
    pred[:, :, :, 1 + N_LAYERS] = 100.0
    pred[:, :, :, 2 + N_LAYERS] = 3.0e-5
    pred[:, :, :, 4 + N_LAYERS] = 1.0e-4
    mod = _make()
    with torch.no_grad():
        started = time.perf_counter()
        mod(pred, orig)
        elapsed = time.perf_counter() - started
    assert elapsed < 1.0, elapsed
