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

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import numpy as np
import pytest
import torch

from physicsnemo.models.dlwp_healpix_layers.healpix_hybrid_dry_air_constraint import (
    ERA5_ACE_8LAYER_AK,
    ERA5_ACE_8LAYER_BK,
    HybridSigmaDryAirMassConstraint,
)

N_LAYERS = 8
WATER = [f"specific_total_water_{k}" for k in range(N_LAYERS)]
IN_CH = ["PRESsfc", *WATER]
OUT_CH = ["PRESsfc", *WATER, "PRATEsfc"]
IDENTITY_SCALING = {n: {"mean": 0.0, "std": 1.0} for n in OUT_CH}
ACE_AK = np.asarray(ERA5_ACE_8LAYER_AK, dtype=np.float64)
ACE_BK = np.asarray(ERA5_ACE_8LAYER_BK, dtype=np.float64)
ACE_DAK = np.diff(ACE_AK)
ACE_DBK = np.diff(ACE_BK)


def _make_mod(**kwargs):
    defaults = dict(
        in_channels=IN_CH,
        out_channels=OUT_CH,
        scaling=IDENTITY_SCALING,
        n_layers=N_LAYERS,
        ak=ERA5_ACE_8LAYER_AK,
        bk=ERA5_ACE_8LAYER_BK,
    )
    defaults.update(kwargs)
    return HybridSigmaDryAirMassConstraint(**defaults)


def _ace_ps_dry(ps: np.ndarray, q: np.ndarray) -> np.ndarray:
    dak = ACE_DAK.reshape(1, 1, 1, -1, 1, 1)
    dbk = ACE_DBK.reshape(1, 1, 1, -1, 1, 1)
    water = (q * (dak + dbk * ps)).sum(axis=3, keepdims=True)
    return ps - water


def _ace_invert(ps_dry: np.ndarray, q: np.ndarray) -> np.ndarray:
    dak = ACE_DAK.reshape(1, 1, 1, -1, 1, 1)
    dbk = ACE_DBK.reshape(1, 1, 1, -1, 1, 1)
    numer = ps_dry + (dak * q).sum(axis=3, keepdims=True)
    denom = 1.0 - (dbk * q).sum(axis=3, keepdims=True)
    return numer / denom


def test_rejects_missing_pressfc():
    with pytest.raises(ValueError, match="PRESsfc"):
        _make_mod(out_channels=WATER)


def test_rejects_missing_water():
    with pytest.raises(ValueError, match="specific_total_water_0"):
        _make_mod(out_channels=["PRESsfc"], in_channels=["PRESsfc"])


def test_rejects_diagnostic_pressfc():
    with pytest.raises(ValueError, match="diagnostic"):
        _make_mod(in_channels=WATER, out_channels=OUT_CH)


def test_rejects_wrong_ak_len():
    with pytest.raises(ValueError, match="interfaces"):
        _make_mod(ak=list(ERA5_ACE_8LAYER_AK)[:-1])


def test_invert_formula_matches_ace():
    rng = np.random.default_rng(0)
    ps = 98000.0 + 2000.0 * rng.standard_normal((2, 12, 1, 1, 4, 4))
    q = np.clip(0.004 + 0.002 * rng.standard_normal((2, 12, 1, N_LAYERS, 4, 4)), 0.0, 0.02)
    ps_dry = _ace_ps_dry(ps, q)
    recovered = _ace_invert(ps_dry, q)
    np.testing.assert_allclose(recovered, ps, rtol=0.0, atol=1e-4)

    pred = torch.zeros(2, 12, 1, len(OUT_CH), 4, 4)
    orig = torch.zeros_like(pred[:, :, :, : len(IN_CH)])
    pred[:, :, :, 0:1] = torch.from_numpy(ps).float()
    pred[:, :, :, 1 : 1 + N_LAYERS] = torch.from_numpy(q).float()
    orig[:, :, :, 0:1] = torch.from_numpy(ps).float()
    orig[:, :, :, 1 : 1 + N_LAYERS] = torch.from_numpy(q).float()
    mod = _make_mod()
    with torch.no_grad():
        out_ps = mod._invert_ps(
            torch.from_numpy(ps_dry).float(),
            torch.from_numpy(q).float(),
        ).numpy()
    np.testing.assert_allclose(out_ps, recovered.astype(np.float32), rtol=0.0, atol=2e-2)


def test_pins_global_mean_dry_air_and_leaves_q():
    rng = np.random.default_rng(1)
    ps0 = 98500.0 + 1500.0 * rng.standard_normal((2, 12, 1, 1, 4, 4))
    q0 = np.clip(0.003 + 0.001 * rng.standard_normal((2, 12, 1, N_LAYERS, 4, 4)), 0.0, 0.02)
    ps = ps0 + 800.0
    q = np.clip(q0 + 0.0005, 0.0, 0.02)
    orig = torch.zeros(2, 12, 1, len(IN_CH), 4, 4)
    pred = torch.zeros(2, 12, 1, len(OUT_CH), 4, 4)
    orig[:, :, :, 0:1] = torch.from_numpy(ps0).float()
    orig[:, :, :, 1:] = torch.from_numpy(q0).float()
    pred[:, :, :, 0:1] = torch.from_numpy(ps).float()
    pred[:, :, :, 1 : 1 + N_LAYERS] = torch.from_numpy(q).float()
    pred[:, :, :, -1] = 3.14
    mod = _make_mod()
    with torch.no_grad():
        out = mod(pred, orig)
    out_np = out.numpy()
    out_ps = out_np[:, :, :, 0:1]
    out_q = out_np[:, :, :, 1 : 1 + N_LAYERS]
    np.testing.assert_allclose(out_q, q.astype(np.float32), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out_np[:, :, :, -1], pred[:, :, :, -1].numpy(), atol=0.0)
    gmean_out = _ace_ps_dry(out_ps.astype(np.float64), out_q.astype(np.float64)).mean(axis=(1, 4, 5))
    gmean_in = _ace_ps_dry(
        orig[:, :, :, 0:1].numpy().astype(np.float64),
        orig[:, :, :, 1:].numpy().astype(np.float64),
    ).mean(axis=(1, 4, 5))
    np.testing.assert_allclose(gmean_out, gmean_in, rtol=0.0, atol=5e-2)


def test_zero_q_pins_mean_ps():
    rng = np.random.default_rng(2)
    ps0 = 99000.0 + 500.0 * rng.standard_normal((1, 12, 1, 1, 4, 4))
    ps = ps0 + 1200.0
    orig = torch.zeros(1, 12, 1, len(IN_CH), 4, 4)
    pred = torch.zeros(1, 12, 1, len(OUT_CH), 4, 4)
    orig[:, :, :, 0:1] = torch.from_numpy(ps0).float()
    pred[:, :, :, 0:1] = torch.from_numpy(ps).float()
    mod = _make_mod()
    with torch.no_grad():
        out_ps = mod(pred, orig)[:, :, :, 0:1]
    np.testing.assert_allclose(
        out_ps.mean(dim=(1, 4, 5)).numpy(),
        orig[:, :, :, 0:1].mean(dim=(1, 4, 5)).numpy(),
        rtol=0.0,
        atol=5e-2,
    )


def test_uses_last_input_time():
    rng = np.random.default_rng(3)
    ps0 = 98000.0 + rng.standard_normal((1, 12, 2, 1, 4, 4))
    ps0[:, :, 0] += 5000.0
    orig = torch.zeros(1, 12, 2, len(IN_CH), 4, 4)
    pred = torch.zeros(1, 12, 1, len(OUT_CH), 4, 4)
    orig[:, :, :, 0:1] = torch.from_numpy(ps0).float()
    pred[:, :, :, 0:1] = 101000.0
    mod = _make_mod()
    with torch.no_grad():
        out_ps = mod(pred, orig)[:, :, :, 0:1].numpy()
    np.testing.assert_allclose(
        out_ps.mean(axis=(1, 4, 5)),
        ps0[:, :, -1:].mean(axis=(1, 4, 5)).astype(np.float32),
        rtol=0.0,
        atol=1e-2,
    )
